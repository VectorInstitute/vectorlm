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


def test_parsing_rationale() -> None:
    """Test parsing rationales."""
    example_input = [
        "\nThe blue jay is to the right of the quail, and the falcon is to the right of the blue jay. So, the order is (B) The quail is the second from the left.",
        "\n\nThe motorcyle is newer than the limousine. The convertible is newer than the motorcyle. So, the correct option is (C) The convertible is the oldest.",
        "\n\nFor the first paragraph, the object arranged to the right is a blue book. Since the blue book is the rightmost, it must be the second object from the left. \n\nIn the second paragraph about the orange book, it must be the object located two positions left of the blue book, which is indeed the leftmost object. \n\nIn the third paragraph, the object that must be two positions left of the orange book is the red book, making option (A) the correct answer.",
        "\n\nThe robin, crow, and blue Jay form a linear order, so based on the position of these birds, the robin must be the rightmost.",
        "\nThe green book is the first object on the shelf. The red book is the second object on the shelf. The blue book is the third object on the shelf. Therefore, the green book is the rightmost.",
        "\nAccording to the statement, the mangoes are less expensive than the peaches, which are less expensive than the apples. Therefore, the mango should be the third-most expensive. So, the correct answer is **(C) The mangoes are the second-most expensive**.",
        "\n\n**Paragraph 1:** The tractor is older than the truck. So, the tractor should be the third object in the order.\n\n**Paragraph 2:** The minivan is newer than the truck. Therefore, the minivan should be the third object in the order.\n\n**Paragraph 3:** The tractor is older than the truck. So, the tractor should be the first object in the order.\n\nTherefore, the answer is (A) The tractor is the newest.",
        "\nEve's position is fixed. Rob finished below Mel, who finished below Eve. Therefore, Eve finished last. So (A) Eve finished first is the answer.",
    ]
