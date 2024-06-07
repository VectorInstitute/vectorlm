"""TRICE implementation in VectorLM vLLM."""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import time
from functools import partial
from typing import Any, Callable, Iterable, NamedTuple, TypeVar

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

ZERO_SHOT_PROMPT = """\
Question: {question}
Answer: Let's think step-by-step. \
"""

GUIDE_TEMPLATE = """\
Question: {question}
Answer: Let's think step by step. {rationale}{answer}\
"""

FEW_SHOT_DELIMITER = "\n\n---\n"

GUIDE_TEMPLATE_FULL = """\
{few_shot_exemplars}\
{few_shot_delimiter}\
Question: {question}
Answer: Let's think step by step. \
"""


# capture groups: rationale, answer.
RATIONALE_ANSWER_REGEXP = r"(.+the answer is )(\([A-C]\))[^\(\)]*$"


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

        import json

        with open("data/output-20240527-1a.jsonl", "a") as output_file:
            output_file.write(json.dumps(output[-1]._asdict()) + "\n")

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

    few_shot_template = GUIDE_TEMPLATE_FULL.format(
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
        loss_mask: Tensor[int] (batch, width)

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

    def _rationales_to_batch(
        self,
        rationales: Iterable[Rationale],
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

        Returns
        -------
            dict:
            - input_ids: (batch_size, num_tokens)
            - attention_mask: (batch_size, num_tokens)
            - loss_mask: (batch_size, num_tokens)

        """
        assert self.tokenizer is not None
        input_id_lists: list[list[int]] = []
        attention_masks_list: list[list[int]] = []
        loss_masks_list: list[list[int]] = []

        # Tokenize prompt and rationale separately before concatenating.
        for rationale in rationales:
            prompt_tokens = list(self.tokenizer(rationale.raw_prompt).input_ids)

            continuation_str = rationale.rationale
            if rationale.parsed_answer is not None:
                continuation_str += rationale.parsed_answer
            rationale_tokens = list(self.tokenizer(continuation_str).input_ids)

            attention_mask = [1] * (len(prompt_tokens) + len(rationale_tokens))
            loss_mask = [0] * len(prompt_tokens) + [1] * len(rationale_tokens)

            input_id_lists.append(prompt_tokens + rationale_tokens)
            attention_masks_list.append(attention_mask)
            loss_masks_list.append(loss_mask)

        # set max_seq_length to max real number of tokens,
        # capped at max_seq_len from config.
        max_seq_length = min(
            self.config.max_seq_len,  # type: ignore[reportAttributeAccessIssue]
            max(map(len, input_id_lists)),
        )
        batch_size = len(input_id_lists)

        input_ids = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        attn_masks = torch.zeros((batch_size, max_seq_length), dtype=torch.int)
        loss_masks = torch.zeros((batch_size, max_seq_length), dtype=torch.int)

        for index, (input_id_list, attention_mask, loss_mask) in enumerate(
            zip(input_id_lists, attention_masks_list, loss_masks_list),
        ):
            # skip rationales that exceed max_seq_length
            actual_length = len(input_id_list)
            if actual_length > max_seq_length:
                continue

            _non_pad_len = min(max_seq_length, actual_length)
            input_ids[index, :_non_pad_len] = torch.Tensor(input_id_list)
            attn_masks[index, :_non_pad_len] = torch.Tensor(attention_mask)
            loss_masks[index, :_non_pad_len] = torch.Tensor(loss_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "loss_mask": loss_masks,
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

        self.train_questions, self.test_questions = get_dataset()

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
        self.memory = few_shot_sample(
            self.batch_generate_fn,
            self.few_shot_rationales,
            self.train_questions,
        )

        valid_rationales = filter_rationales(self.memory)
        _serialize_memory(self.memory, {"step": self.tr_step})
        print(f"Valid memories: {len(valid_rationales)}/{len(self.memory)}")

    def _get_train_rationales(
        self,
        num_rationales: int,
    ) -> dict[DatasetIndex, Rationale]:
        """Sample new rationales from training questions.

        Return only rationales where answer is correct. Prefer newly
        generated rationales and fall back to ones from memory for the same
        set of randomly-selected questions.
        """
        random.seed(self.tr_step)
        selected_keys = random.sample(self.memory.keys(), num_rationales)
        prev_rationales = {key: self.memory[key] for key in selected_keys}

        selected_questions = {
            key: rationale.question
            for key, rationale in prev_rationales.items()
        }

        # "proposed" rationales
        new_rationales = few_shot_sample(
            self.batch_generate_fn,
            self.few_shot_rationales,
            selected_questions,
        )

        return {
            **filter_rationales(prev_rationales),
            **filter_rationales(new_rationales),
        }

    def train_step(self, _: dict[str, torch.Tensor], epoch: int) -> float:
        """Apply one TRICE training step.

        See BASIC_GRADIENT_ESTIMATE in the TRICE paper.
        """
        assert self.model is not None
        assert self.optimizer is not None
        assert self.lr_scheduler is not None
        assert self.sampling_engine is not None

        # keep sampling until at least one rationale (new or memory) is correct.
        new_correct_rationales: dict[DatasetIndex, Rationale] = {}
        while len(new_correct_rationales) == 0:
            new_correct_rationales = self._get_train_rationales(
                self.config.batch_size,  # type: ignore[reportAttributeAccessIssue]
            )

        # Write newly sampled correct rationales to memory.
        self.memory = {**self.memory, **new_correct_rationales}
        _serialize_memory(self.memory, {"step": self.tr_step})

        training_batch = self._rationales_to_batch(
            new_correct_rationales.values(),
        )
        _batch = {
            k: v.to(torch.cuda.current_device())
            for k, v in training_batch.items()
        }

        # Sync grad only if is about to run update.
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
                _batch["loss_mask"],
            )
            (tr_step_loss / self.gas).backward()
            self.model.clip_grad_norm_(self.config.max_grad_norm)  # type: ignore[reportAttributeAccessIssue]

            if is_update_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.sampling_engine.update(self.model, self.tr_step)

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
            self.batch_generate_fn,
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
