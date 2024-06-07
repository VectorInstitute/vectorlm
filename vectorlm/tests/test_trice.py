import pytest
import torch

from vectorlm.trice import masked_clm_loss


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
