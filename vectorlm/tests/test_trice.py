from __future__ import annotations

import numpy as np
import pytest
import torch

from vectorlm.trice import (
    Question,
    Rationale,
    _index,
    filter_rationales,
    get_weighted_rationales,
    masked_clm_loss,
)


def test_masked_clm_loss(
    batch_size: int = 2,
    seq_length: int = 8,
    vocab_size: int = 12,
) -> None:
    """Test partially masked next-token loss fn."""
    logits = torch.ones((batch_size, seq_length, vocab_size))
    input_ids = torch.ones((batch_size, seq_length), dtype=torch.long)
    loss_multiplier = torch.ones((batch_size, seq_length), dtype=torch.long)

    loss = masked_clm_loss(logits, input_ids, loss_multiplier)
    print(loss)


def get_reference_weights(
    is_proposal_correct_: list[bool],
    is_memory_correct_: list[bool],
) -> tuple[list[float], float]:
    """Return (signed) weights and weights_mean.

    Adapted from reference TRICE logic.
    """
    is_proposal_correct = np.stack(is_proposal_correct_)
    is_memory_correct = np.stack(is_memory_correct_)
    mask = is_proposal_correct | is_memory_correct
    correlation_est = (is_proposal_correct.sum() - is_proposal_correct) / (
        mask.sum() - 1 + 1e-10
    )

    # compute weight contributions of rationales from both memory and proposal.
    weights_memory = mask * (1 - correlation_est * is_proposal_correct)
    weights_proposal = mask * correlation_est * (1 - is_proposal_correct)
    flat_weights = np.concatenate([weights_memory, weights_proposal])
    flat_signs = np.concatenate([mask, -1 * mask])
    flat_weights = np.clip(flat_weights, a_min=1e-10, a_max=None)
    weights_mean = flat_weights.sum() / (mask.sum() + 1e-10)

    output = flat_signs * flat_weights
    assert len(output.shape) == 1
    return output.tolist(), weights_mean


@pytest.fixture()
def example_rationales() -> list[Rationale]:
    """Return rationales: F T T F T F."""
    return [
        Rationale(Question("", "A"), "", "F", "B"),
        Rationale(Question("", "A"), "", "T", "A"),
        Rationale(Question("", "A"), "", "T", "A"),
        Rationale(Question("", "A"), "", "F", "B"),
        Rationale(Question("", "A"), "", "T", "A"),
        Rationale(Question("", "A"), "", "F", "B"),
    ]


@pytest.mark.parametrize("variation", [0, 1, 2])
def test_weight_rationales(
    example_rationales: list[Rationale],
    variation: int,
) -> None:
    """Ensure rationales weights match ones from reference."""
    if variation == 0:
        # proposed: F T T
        # memory: F T F
        proposed_rationales = example_rationales[:3]
        memorized_rationales = example_rationales[3:]
    elif variation == 1:
        # proposed: T,
        # memory: T,
        proposed_rationales = [example_rationales[2]]
        memorized_rationales = [example_rationales[3]]
    else:
        # proposed: F T F
        # memory: F T T
        proposed_rationales = example_rationales[3:]
        memorized_rationales = example_rationales[:3]

    weighted_rationales = get_weighted_rationales(
        memorized_rationales={
            **_index(memorized_rationales),
            **filter_rationales(_index(proposed_rationales)),
        },
        proposed_rationales=_index(proposed_rationales),
    )
    weights = [
        wr.weight
        for wr in (weighted_rationales.memorized + weighted_rationales.proposed)
    ]
    weights_reference, weights_mean_reference = get_reference_weights(
        [r.is_correct for r in proposed_rationales],
        [r.is_correct for r in memorized_rationales],
    )

    print(
        [r.is_correct for r in proposed_rationales],
        [r.is_correct for r in memorized_rationales],
    )
    print(weights)
    print(weights_reference)
    assert np.allclose(weights, weights_reference, 1e-6)
    assert np.allclose(
        weighted_rationales.weights_mean,
        weights_mean_reference,
        1e-6,
    )
