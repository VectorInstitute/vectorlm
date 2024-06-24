from vectorlm.sampling import batch_process

RATIONALE_ANSWER_REGEXP = r"(.+)\s\(([A-C])\)[^\(\)]*$"


def test_batch_process() -> None:
    """Test batch_process."""
    example_input = list("banana")
    output = []
    for output_item in batch_process(example_input, lambda x: x, 5):
        print(output_item)
        output.append(output_item)


    assert output == example_input
