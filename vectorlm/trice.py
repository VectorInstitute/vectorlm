"""TRICE implementation in VectorLM vLLM."""

from __future__ import annotations

import collections
import contextlib
import json
import os
import random
import re
import string
import time
from functools import partial
from typing import Any, Callable, Counter, Iterable, NamedTuple, TypeVar

import datasets
import numpy as np
import requests
import torch
import torch.distributed
from tqdm.auto import tqdm
from vllm import SamplingParams

from vectorlm.sampling import batch_process
from vectorlm.trainer import Trainer, _gather

DatasetIndex = str

START_TIME = int(time.time())
SAMPLING_BATCH_SIZE = 8
NUM_DEMONSTRATIONS = {
    # prompt for bootstraping high-quality memory using hinted rationales.
    "few_shot_rationales": 3,
}

QUESTION_TEMPLATE = """\
{input_text}
Options:
{rendered_answer_choices}
"""
ANSWER_OPTION_TEMPLATE = "{answer_key} {label}"


ZERO_SHOT_PROMPT = """\
{question}
Answer: Let's think step-by-step. \
"""

GUIDE_TEMPLATE = """\
Question: {question}
Answer: Let's think step by step. {rationale}{answer}\
"""


FEW_SHOT_DELIMITER = "\n\n---\n"

FEW_SHOT_TEMPLATE_FULL = """\
{few_shot_exemplars}\
{few_shot_delimiter}\
Question: {question}
Answer: Let's think step by step. \
"""


# capture groups: rationale, answer.
RATIONALE_ANSWER_REGEXP = r"(.+the answer is )\(?([A-C])\)?[^\(\)]*$"


class Question(NamedTuple):
    """A question-answer pair."""

    question_text: str
    answer: str


class Rationale(NamedTuple):
    """A question-rationale pair.

    The "parsed_answer" field refers to the answer from model output,
    which might be different from the ground truth reference
    in question.answer.

    "parsed_answer" is None if output isn't parse-able.
    """

    question: Question
    raw_prompt: str
    rationale: str
    parsed_answer: str | None = None

    def serialize(self) -> dict[str, Any]:
        """Produce JSON-friendly representation of self."""
        output = self._asdict()
        output["question"] = self.question._asdict()
        if not bool(os.environ.get("PRINT_VERBOSE_MEMORY", 0)):
            output.pop("raw_prompt")

        return output

    @property
    def is_correct(self) -> bool:
        """Return whether self is correct."""
        return self.parsed_answer == self.question.answer


class _WeightedRationale(NamedTuple):
    """Rationale paired with associated weight.

    Weight can be negative, as in the example of proposed incorrect rationales,
    where the corresponding memory is correct, but this rationale is not.

    Weights for memory entries:
    - 0 if neither this memory nor the proposal is correct.
    - 1 if this memory is correct, but the corresponding proposal is incorrect.
    - (1 - scale) if both are correct, higher if scale is lower, when other
        proposals in this batch are not as accurate as other memories in this
        batch.

    Weights for proposal entries:
    - 0 if this proposal is correct.
    - 0 if neither this proposal nor the corresponding memory is correct.
    - (-scale) if proposal is incorrect, but the corresponding memory is correct
    """

    rationale: Rationale
    weight: float

    @property
    def sign_multiplier(self) -> int:
        """Return sign multiplier of self."""
        if self.weight > 0:
            return 1

        if self.weight < 0:
            return -1

        return 0


class _WeightedRationales(NamedTuple):
    """WeightedRationales from memory and proposal.

    "weights_mean" is for rescaling clm grad values.
    """

    memorized: list[_WeightedRationale]
    proposed: list[_WeightedRationale]
    weights_mean: float


def generate_rationale_answer(
    questions: Iterable[Question],
    batch_generate_fn: Callable[[list[str]], list[str]],
    prompt_template: str = ZERO_SHOT_PROMPT,
    require_match: bool = False,
) -> list[Rationale]:
    """Generate rationale and answer to each of the given questions.

    Params:
    ------
        question_texts: list of Question.
        batch_generate_fn: Generate text continuations for each given query.
        prompt_template: str, where the only placeholder is "question".
        require_match: bool, enable to exclude rationales that don't match
            RATIONALE_ANSWER_REGEXP.

    Returns
    -------
        List of Rationales for parsed examples.

    """
    queries = [
        prompt_template.format(question=question.question_text)
        for question in questions
    ]
    responses = batch_generate_fn(queries)

    output: list[Rationale] = []
    for question, query, response in zip(questions, queries, responses):
        match = re.match(RATIONALE_ANSWER_REGEXP, response, re.DOTALL)
        if match is None:
            if require_match:
                continue
            rationale, answer = response, None
        else:
            rationale, answer = match.groups()
        output.append(Rationale(question, query, rationale, answer))

    return output


V = TypeVar("V")


def _index(items: list[V]) -> dict[DatasetIndex, V]:
    """Convert list of items to a dict mapping index to item."""
    return {str(index): item for index, item in enumerate(items)}


def get_dataset() -> (
    tuple[dict[DatasetIndex, Question], dict[DatasetIndex, Question]]
):
    """Get train and validation datasets.

    Returns
    -------
      train_questions, test_questions

    """
    task = "logical_deduction_three_objects"
    data_url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{task}.json"
    examples = requests.get(data_url, timeout=10).json()["examples"]

    question_texts = [ex["input"] for ex in examples]
    answer_texts = [ex["target"] for ex in examples]
    questions = [
        Question(question, answer)
        for question, answer in zip(question_texts, answer_texts)
    ]

    return _index(questions[:150]), _index(questions[150:])


def enumerate_answer_choices(
    dataset_splits: list[Iterable[dict[str, Any]] | datasets.Dataset],
    label_column_key: str,
    label_lookup: dict[Any, str] | None,
    data_limits: list[int | None],
) -> Counter[str]:
    """Return counter of all answer choice options.

    Args:
    ----
        dataset_splits: list of dataset row iterators, one for each split.
        label_column_key: should be the same across all splits.
        label_lookup: Translate label to alternative, more descriptive form
            before enumeration. If provided, rows where label is not in
            the lookup dictionary would be skipped.
        data_limits: max number of rows to load from each split, one per split.
            Set to None for no limit.

    Returns:
    -------
        Counter of label options across all splits.

    """
    output: Counter[str] = collections.Counter()
    for dataset_split, max_row_count in zip(dataset_splits, data_limits):
        for index, row in enumerate(dataset_split):
            assert isinstance(row, dict)
            label = row.get(label_column_key)

            if label_lookup is not None:
                label = label_lookup.get(label)
                if label is None:
                    continue

            output[str(label)] += 1

            if (max_row_count is not None) and (index + 1 == max_row_count):
                break

    return output


def _render_label_choices(
    label_choices: list[str],
    template: str,
) -> tuple[str, dict[str, str]]:
    """Render label_choices as a multiple-choice question.

    Returns
    -------
        lookup dict mapping label value to assigned answer key e.g., "(A)".

    """
    output_lines = []
    label_lookup: dict[str, str] = {}

    assert len(label_choices) <= len(string.ascii_uppercase)
    for index, label in enumerate(label_choices):
        answer_key = string.ascii_uppercase[index]
        label_lookup[label] = answer_key
        output_line = template.format(answer_key=answer_key, label=label)
        output_lines.append(output_line)

    return "\n".join(output_lines), label_lookup


def load_hf_dataset(
    dataset_path: str,
    text_column_key: str,
    label_column_key: str,
    data_split_names: tuple[str, str] = ("train", "test"),
    label_lookup: dict[str, str] | None = None,
    data_limits: tuple[int | None, int | None] = (1000, 100),
    question_template: str = QUESTION_TEMPLATE,
    answer_option_template: str = ANSWER_OPTION_TEMPLATE,
) -> tuple[dict[DatasetIndex, Question], dict[DatasetIndex, Question]]:
    """Load TRICE-compatible dataset from a HuggingFace dataset.

    Args:
    ----
        dataset_path: path to HF dataset repo or local folder.
        text_column_key: source of input_text.
        label_column_key: source of labels to render as options.
        data_split_names: dataset splits for training and testing.
        label_lookup: Optionally, supply a descriptive label to replace the
            original labels from the dataset. If specified, only rows where
            original labels is in label_lookup would be included. Otherwise,
            all rows would be included and the original label would be used.
        data_limits: max. number of entries to load from train and test split.
        question_template: question template, should include {input_text} and
            {rendered_answer_choices}
        answer_option_template: template for answer choices, should include
            {answer_key} and {label}

    Returns:
    -------
        dict of train questions (index -> Question),
        dict of test questions (index -> Question),

    """
    # prefer loading from local path if exists
    # instead of loading from HF hub
    if os.path.isdir(dataset_path):
        dataset_dict = datasets.load_from_disk(dataset_path)
    else:
        dataset_dict = datasets.load_dataset(dataset_path)

    assert isinstance(dataset_dict, datasets.dataset_dict.DatasetDict)
    assert len(data_split_names) == len(("train", "test"))

    for data_split_name in data_split_names:
        assert data_split_name in dataset_dict, (data_split_names, dataset_dict)

    label_choices = enumerate_answer_choices(
        [dataset_dict[split_name] for split_name in data_split_names],
        label_column_key,
        label_lookup,
        list(data_limits),
    )
    label_choice_str, answer_map = _render_label_choices(
        [str(label) for label in label_choices],
        answer_option_template,
    )
    print("label_choices stats:", label_choices)
    print("label_choice_str:", label_choice_str)
    print("Answer map:", answer_map)

    output: list[dict[DatasetIndex, Question]] = []
    # create one question for each row for each dataset split.
    for data_split_name, max_num_rows in zip(data_split_names, data_limits):
        questions: dict[DatasetIndex, Question] = {}
        dataset = dataset_dict[data_split_name]
        for index, row in enumerate(dataset):
            assert isinstance(row, dict)

            # Translate label and skip rows where label is not in lookup.
            label = row[label_column_key]
            if label_lookup is not None:
                label = label_lookup.get(label)

            if label is None:
                continue

            label = str(label)

            # use str keys to allow json serialization
            questions[str(index)] = Question(
                question_text=question_template.format(
                    input_text=row[text_column_key],
                    rendered_answer_choices=label_choice_str,
                ),
                answer=answer_map[label],
            )

            if (max_num_rows is not None) and (index + 1 == max_num_rows):
                break

        output.append(questions)

    print(output[0]["0"].question_text + "\n")

    assert len(output) == len(("train", "test"))
    return tuple(output)  # type: ignore[tuple length]


def get_n_correct_rationales(
    batch_generate_fn: Callable[[list[str]], list[str]],
    questions: dict[DatasetIndex, Question],
    max_num_correct: int,
) -> list[Rationale]:
    """Return up to N correct rationales.

    Opportunistically stop as soon as number of correct rationales is reached.
    """
    # lazy iterator- generation does not happen until iterated.
    rationale_iterator = batch_process(
        questions.values(),
        partial(generate_rationale_answer, batch_generate_fn=batch_generate_fn),
        SAMPLING_BATCH_SIZE,
    )

    # stop iterating as soon as exactly N rationales are correct.
    correct_rationales = []
    for rationale in tqdm(rationale_iterator, ncols=75, total=len(questions)):
        if str(rationale.question.answer) == rationale.parsed_answer:
            correct_rationales.append(rationale)

        if len(correct_rationales) == max_num_correct:
            break

    return correct_rationales


def filter_rationales(
    proposed_rationales: dict[DatasetIndex, Rationale],
) -> dict[DatasetIndex, Rationale]:
    """Return only valid rationales from the given dict of rationales.

    Params:
    ------
        proposed_rationales: dict mapping dataset index to rationale.

    Returns
    -------
        a subset of the input dict, containing only rationales that are
        correct.

    """
    return {
        index: rationale
        for index, rationale in proposed_rationales.items()
        if rationale.parsed_answer == rationale.question.answer
    }


def few_shot_sample(
    batch_generate_fn: Callable[[list[str]], list[str]],
    few_shot_rationales: list[Rationale],
    questions: dict[DatasetIndex, Question],
) -> dict[DatasetIndex, Rationale]:
    """Generate answers to the given questions using few-shot examples.

    Args:
    ----
        batch_generate_fn: Callable.
        few_shot_rationales: list of correct rationales to demonstrate.
        questions: dict mapping dataset index to Question.

    Returns:
    -------
        dict mapping dataset index to Rationale instance,
        one for each given question.

    """
    few_shot_exemplars = FEW_SHOT_DELIMITER.join(
        GUIDE_TEMPLATE.format(
            question=rationale.question.question_text,
            rationale=rationale.rationale,
            answer=rationale.parsed_answer,
        )
        for rationale in few_shot_rationales
    )

    few_shot_template = FEW_SHOT_TEMPLATE_FULL.format(
        few_shot_exemplars=few_shot_exemplars,
        few_shot_delimiter=FEW_SHOT_DELIMITER,
        question="{question}",
    )

    rationales = generate_rationale_answer(
        batch_generate_fn=batch_generate_fn,
        questions=questions.values(),
        prompt_template=few_shot_template,
    )

    return dict(zip(questions.keys(), rationales))


def get_weighted_rationales(
    memorized_rationales: dict[DatasetIndex, Rationale],
    proposed_rationales: dict[DatasetIndex, Rationale],
) -> _WeightedRationales:
    """Obtain TRICE Control-Variate weights for each rationale.

    Obtain leave-one-out scales
    - Divide number of other correct proposals, excluding self, by total number
        of correct predictions in batch, also excluding self.

    Obtain weights for memory + correct proposal entries
    - 0 if neither original memory nor the new proposal is correct.
    - 1 if original memory is correct, but the corresponding new proposal is
        incorrect.
    - (1 - scale) if new proposal is correct, higher if scale is lower, when
        other proposals are not as accurate as original memories.

    Obtain weights for all proposal entries, negative if incorrect.
    - 0 if this proposal is correct.
    - 0 if neither this proposal nor the corresponding memory is correct.
    - (-scale) if proposal is incorrect, but the corresponding memory is
        correct.

    Params:
    ----
        memorized_rationales: memory rationales.
        proposed_rationales: proposed new rationales.


    Keys of "memorized" must match those of "proposed".

    Returns
    -------
        List of weighted rationales. If there are N memory rationales
        and N proposals, the output would be of length (2 * N).

    """
    assert proposed_rationales.keys() == memorized_rationales.keys()
    num_correct_proposals = len(filter_rationales(proposed_rationales))
    num_correct_all = len(
        {
            **filter_rationales(proposed_rationales),
            **filter_rationales(memorized_rationales),
        },
    )

    # construst weighted list of rationale from both memory and proposal.
    output_memorized: list[_WeightedRationale] = []
    output_proposed: list[_WeightedRationale] = []
    for dataset_index in memorized_rationales:
        memorized = memorized_rationales[dataset_index]
        proposed = proposed_rationales[dataset_index]
        if (proposed.is_correct) and (not memorized.is_correct):
            msg = (
                "Proposal is correct but memory is not. "
                "Did you update memory before trying to compute weights?"
                f"proposal: {proposed}\n"
                f"memory: {memorized}"
            )
            raise ValueError(msg)

        # leave-one-out (leave-self-out) scale
        scale_numerator = num_correct_proposals
        scale_denominator = num_correct_all - 1 + 1e-10
        if proposed.is_correct:
            scale_numerator -= 1

        scale = scale_numerator / scale_denominator

        # weight for memory, which might have been overwritten with the new
        # proposal if that proposal is correct.
        if (not proposed.is_correct) and (not memorized.is_correct):
            weight_memorized = 0.0
        elif (memorized.is_correct) and (not proposed.is_correct):
            weight_memorized = 1.0
        else:
            # proposal is correct, and should have already overwritten memory.
            weight_memorized = 1 - scale
        output_memorized.append(_WeightedRationale(memorized, weight_memorized))

        # weight for proposal
        if proposed.is_correct:
            # Since proposed is correct, after memory update "memorized"
            # would be the same as "proposed". No need to include this again.
            # Hence, set weight to 0.
            weight_proposed = 0.0
        elif (not proposed.is_correct) and (not memorized.is_correct):
            weight_proposed = 0.0
        else:
            # new proposal is incorrect, but original memory is correct
            weight_proposed = -scale

        output_proposed.append(_WeightedRationale(proposed, weight_proposed))

    # sum of all weights, for rescaling grads.
    weight_tally = sum(
        abs(wr.weight) for wr in (output_memorized + output_proposed)
    )

    return _WeightedRationales(
        memorized=output_memorized,
        proposed=output_proposed,
        weights_mean=weight_tally / (num_correct_all + 1e-10),
    )


def _softmax(weights: np.ndarray) -> np.ndarray:
    """Return softmax value given weights."""
    assert len(weights.shape) == 1, weights.shape
    exp_weights = np.exp(weights - np.max(weights))
    return exp_weights / np.sum(exp_weights, axis=0)


def _systematic_resample(
    probabilities: np.ndarray,
    num_selected: int,
    seed: int = 0,
) -> list[int]:
    """Resample systematically.

    Params:
    ------
        probabilties: 1D float array, must sum up to 1.
        num_selected: number of items to select.

    Returns
    -------
        list of index of "num_selected" items that were selected.
        Each item is an index of the probability array.

    """
    assert np.allclose(probabilities.sum(), 1), "Forgot to normalize?"
    assert num_selected > 0

    generator = np.random.Generator(np.random.PCG64(seed))
    randomness = generator.uniform(0, 1 / num_selected)
    selections: list[int] = []

    thresholds = np.cumsum(probabilities).tolist()  # (N,)
    thresholds_low = [0.0, *thresholds]  # (N + 1,)
    thresholds_high = [*thresholds, 1.0]  # (N + 1,)
    for option_index, (threshold_low, threshold_high) in enumerate(
        zip(thresholds_low, thresholds_high),
    ):
        # try assigning each selection to the threshold, starting
        # from the next available one.
        for selection_index in range(len(selections), num_selected):
            selected_pos = selection_index * (1 / num_selected) + randomness

            if (selected_pos >= threshold_low) and (
                selected_pos < threshold_high
            ):
                selections.append(option_index)

    return selections


def subsample_weighted(
    weighted_rationales: list[_WeightedRationale],
    num_items: int,
    seed: int = 0,
) -> list[_WeightedRationale]:
    """Subsample rationales based on absolute value of weights."""
    weights = np.array([abs(wr.weight) for wr in weighted_rationales])
    weights = np.clip(weights, a_min=1e-10, a_max=None)
    probabilities = _softmax(weights)
    selected_index_items = _systematic_resample(probabilities, num_items, seed)

    return [weighted_rationales[index] for index in selected_index_items]


def _serialize_memory(
    memory: dict[DatasetIndex, Rationale],
    extra_info: dict[str, Any] | None = None,
    filename_suffix: str = "",
) -> None:
    """Write memory to disk."""
    output_file_path = os.path.join(
        os.environ.get("MEMORY_PATH", "data/memories"),
        f"{START_TIME}{filename_suffix}.json",
    )
    output: dict[str, Any] = {
        "extra_info": extra_info,
        "valid_rationales": {
            index: rationale.serialize()
            for index, rationale in filter_rationales(memory).items()
        },
        "all_rationales": {
            index: rationale.serialize() for index, rationale in memory.items()
        },
    }

    with open(output_file_path, "a") as output_file:
        output_file.write(json.dumps(output, indent=2))
        output_file.write("\n")


def masked_clm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_multiplier: torch.Tensor,
) -> torch.Tensor:
    """Return partially-masked next-token loss for logits.

    loss_multiplier is applied to each token element-wise.

    Params:
    -------
        logits: Tensor[float] (batch, width, vocab)
        input_ids: Tensor[int] (batch, width)
        loss_multiplier: Tensor[int] (batch, width)

    Returns
    -------
        Tensor[float] (,)

    """
    assert logits.shape[:-1] == input_ids.shape
    assert loss_multiplier.shape == input_ids.shape

    # all logits except the one for the last token.
    logits_sliced = logits[:, :-1, :]

    # all labels except the label for the first token.
    labels_shifted = input_ids[:, 1:]
    loss_multiplier_shifted = loss_multiplier[:, 1:]

    # Torch CrossEntropyLoss allows only one batch dimension,
    # not two (batch, width) as in logits and labels.
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fn(
        logits_sliced.flatten(0, 1),
        labels_shifted.flatten(0, 1),
    ).view_as(labels_shifted)
    assert per_token_loss.shape == loss_multiplier_shifted.shape

    return torch.mean(per_token_loss * loss_multiplier_shifted)


class ICEMTrainer(Trainer):
    """Independence chain expectation maximization trainer for TRICE."""

    train_mini_batch_size: int = 8  # parameter "M" as in paper

    def _batch_tokenize_rationales(
        self,
        rationales: Iterable[Rationale],
        rationale_weights: Iterable[float] | None = None,
        use_raw_prompt: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize rationales to produce training batch.

        In the order of being generated:
        - Prompt: "x"
        - Rationale: "z"
        - Answer: "y"

        Objective: maximize "P(z | x)".

        Note that:
        - Answer ("y") is not included.
        - Loss is not calculated over prompt tokens ("x").

        Params
        ------
        rationale_weights: If provided, rescale loss_multiplier of each item by
            this value. Should be of the same length as rationale.
        use_raw_prompt: Use raw_prompt as context if set to True.
            Otherwise, use zero-shot prompt as context.

        Returns
        -------
            dict:
            - input_ids: int, (batch_size, num_tokens)
            - attention_mask: int, (batch_size, num_tokens)
            - loss_multipliers: float, (batch_size, num_tokens)

        """
        assert self.tokenizer is not None
        input_id_lists: list[list[int]] = []
        attention_masks_list: list[list[int]] = []
        loss_multipliers_list: list[list[int]] = []

        # Tokenize prompt and rationale separately before concatenating.
        for rationale in rationales:
            if use_raw_prompt:
                context = rationale.raw_prompt
            else:
                context = ZERO_SHOT_PROMPT.format(
                    question=rationale.question.question_text,
                )

            prompt_tokens = list(self.tokenizer(context).input_ids)

            continuation_str = rationale.rationale
            if rationale.parsed_answer is not None:
                continuation_str += rationale.parsed_answer
            rationale_tokens = list(self.tokenizer(continuation_str).input_ids)

            attention_mask = [1] * (len(prompt_tokens) + len(rationale_tokens))
            loss_multiplier = [0] * len(prompt_tokens) + [1] * len(
                rationale_tokens,
            )
            assert sum(loss_multiplier) > 0

            input_id_lists.append(prompt_tokens + rationale_tokens)
            attention_masks_list.append(attention_mask)
            loss_multipliers_list.append(loss_multiplier)

        # set max_seq_length to max real number of tokens,
        # capped at max_seq_len from config.
        max_seq_length = min(
            self.config.max_seq_len,  # type: ignore[reportAttributeAccessIssue]
            max(map(len, input_id_lists)),
        )
        batch_size = len(input_id_lists)
        weights = (
            list(rationale_weights)
            if rationale_weights is not None
            else [1.0] * batch_size
        )

        assert batch_size > 0
        input_ids = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        attn_masks = torch.zeros((batch_size, max_seq_length), dtype=torch.int)
        loss_multipliers = torch.zeros(
            (batch_size, max_seq_length),
            dtype=torch.float,
        )

        for index, (
            input_id_list,
            attention_mask,
            loss_multiplier,
            weight,
        ) in enumerate(
            zip(
                input_id_lists,
                attention_masks_list,
                loss_multipliers_list,
                weights,
            ),
        ):
            # skip rationales that exceed max_seq_length
            actual_length = len(input_id_list)
            if actual_length > max_seq_length:
                continue

            _non_pad_len = min(max_seq_length, actual_length)
            input_ids[index, :_non_pad_len] = torch.Tensor(input_id_list)
            attn_masks[index, :_non_pad_len] = torch.Tensor(attention_mask)
            loss_multipliers[index, :_non_pad_len] = (
                torch.Tensor(loss_multiplier) * weight
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "loss_multipliers": loss_multipliers,
        }

    def prepare_trainer(
        self,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize memory and bootstrap prompt."""
        super().prepare_trainer(*args, **kwargs)

        assert self.sampling_engine is not None
        self.batch_generate_fn = partial(
            self.sampling_engine.generate_text_only,
            sampling_params=SamplingParams(
                max_tokens=self.config.max_seq_len,  # type: ignore[reportAttributeAccessIssue]
                temperature=1.0,
            ),
        )
        self.batch_generate_fn_eval = partial(
            self.sampling_engine.generate_text_only,
            sampling_params=SamplingParams(
                max_tokens=self.config.max_seq_len,  # type: ignore[reportAttributeAccessIssue]
                temperature=0.0,
            ),
            use_tqdm=True,
        )

        self.trice_config = self.config.trice_configs  # type: ignore[reportAttributeAccessIssue]
        trice_data_config = self.trice_config.hf_dataset
        self.train_questions, self.test_questions = load_hf_dataset(
            trice_data_config.path,
            text_column_key=trice_data_config.text_column_key,
            label_column_key=trice_data_config.label_column_key,
            answer_option_template=trice_data_config.answer_option_template,
            data_limits=(
                trice_data_config.limits.train,
                trice_data_config.limits.test,
            ),
            label_lookup=trice_data_config.get("label_lookup"),
        )

        # Obtain a number of rationales for bootstraping the prompt for
        # generating explanations given answer hint.
        self.few_shot_rationales = get_n_correct_rationales(
            self.batch_generate_fn,
            self.train_questions,
            NUM_DEMONSTRATIONS["few_shot_rationales"],
        )
        print(
            "Obtained {}/{} few_shot_rationales".format(
                len(self.few_shot_rationales),
                NUM_DEMONSTRATIONS["few_shot_rationales"],
            ),
        )

        # "Memory" is a dict mapping dataset_id to rationales.
        print(f"Initializing memory ({len(self.train_questions)} total).")
        self.memory = few_shot_sample(
            partial(self.batch_generate_fn, use_tqdm=True),
            self.few_shot_rationales,
            self.train_questions,
        )

        valid_rationales = filter_rationales(self.memory)
        _serialize_memory(self.memory, {"step": self.tr_step})
        print(f"Valid memories: {len(valid_rationales)}/{len(self.memory)}")

    def sample_rationales_and_update_memory(
        self,
        num_rationales: int,
    ) -> _WeightedRationales:
        """Sample new rationales and write to memory if correct.

        Params:
        ------
            num_rationales: number of rationales to generate on.

        Returns
        -------
            _WeightedRationales, including "num_rationales" each
                of memory and proposal, as well as weighted scores.

        """
        random.seed(self.tr_step)
        selected_keys = random.sample(self.memory.keys(), num_rationales)
        memorized_rationales = {key: self.memory[key] for key in selected_keys}

        selected_questions = {
            key: rationale.question
            for key, rationale in memorized_rationales.items()
        }

        # "proposed" rationales
        proposed_rationales = few_shot_sample(
            partial(self.batch_generate_fn, use_tqdm=False),
            self.few_shot_rationales,
            selected_questions,
        )
        assert len(proposed_rationales) == len(memorized_rationales)

        # write correct proposed rationales to memories.
        memorized_rationales = {
            **memorized_rationales,
            **filter_rationales(proposed_rationales),
        }
        self.memory = {**self.memory, **filter_rationales(proposed_rationales)}
        _serialize_memory(proposed_rationales, {"step": self.tr_step})

        return get_weighted_rationales(
            memorized_rationales=memorized_rationales,
            proposed_rationales=proposed_rationales,
        )

    def train_step(self, _: dict[str, torch.Tensor], epoch: int) -> float:
        """Apply one TRICE training step.

        See BASIC_GRADIENT_ESTIMATE in the TRICE paper.
        """
        assert self.model is not None
        assert self.optimizer is not None
        assert self.lr_scheduler is not None
        assert self.sampling_engine is not None

        # SAMPLING_SIZE _WeightedRationale instances, not yet subsampled.
        weighted_rationales = self.sample_rationales_and_update_memory(
            self.trice_config.sampling_size,
        )
        subsampled_rationales = subsample_weighted(
            weighted_rationales.memorized + weighted_rationales.proposed,
            self.trice_config.batch_size,
            epoch,
        )

        rationales = [wr.rationale for wr in subsampled_rationales]
        rationale_weights = [
            weighted_rationales.weights_mean * wr.sign_multiplier
            for wr in subsampled_rationales
        ]
        training_batch = self._batch_tokenize_rationales(
            rationales,
            rationale_weights=rationale_weights,
            use_raw_prompt=False,
        )

        _batch = {
            k: v.to(torch.cuda.current_device())
            for k, v in training_batch.items()
        }

        # Sync grad only if about to run update.
        is_update_step = (self.tr_step + 1) % self.gas == 0
        with contextlib.ExitStack() as stack:
            if not is_update_step:
                stack.enter_context(self.model.no_sync())
            else:
                torch.distributed.barrier()

            logits = self.model(
                input_ids=_batch["input_ids"],
                attention_mask=_batch["attention_mask"],
            ).logits
            tr_step_loss = masked_clm_loss(
                logits,
                _batch["input_ids"],
                _batch["loss_multipliers"],
            )
            (tr_step_loss / self.gas).backward()
            self.model.clip_grad_norm_(self.config.max_grad_norm)  # type: ignore[reportAttributeAccessIssue]

            if is_update_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.sampling_engine.update(
                    self.model,
                    self.tr_step,
                    self.tokenizer,
                )

                if isinstance(
                    self.lr_scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.lr_scheduler.step(self.metric)
                else:
                    self.lr_scheduler.step()

        gathered_tr_step_loss = _gather(tr_step_loss.reshape(1)).mean().item()
        if self.wandb_logging:
            self.log(gathered_tr_step_loss, epoch, "train")

        return gathered_tr_step_loss

    def eval_step(self, epoch: int) -> float:
        """Return eval accuracy."""
        rationales = few_shot_sample(
            self.batch_generate_fn_eval,
            self.few_shot_rationales,
            self.test_questions,
        )
        valid_rationales = filter_rationales(rationales)

        accuracy = (
            len(valid_rationales) / len(rationales)
            if len(rationales) > 0
            else 0
        )
        _serialize_memory(self.memory, {"epoch": epoch, "step": self.tr_step})
        _serialize_memory(
            rationales,
            {"epoch": epoch, "step": self.tr_step, "accuracy": accuracy},
            "_eval",
        )
        print(f"Eval accuracy: {accuracy * 100:.1f}%")

        return accuracy
