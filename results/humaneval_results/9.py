from typing import List, Tuple


def sum_product_prompt(numbers: List[int]) -> Tuple[int, int]:
    """For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product_prompt([])
    (0, 1)
    >>> sum_product_prompt([1, 2, 3, 4])
    (10, 24)
    """
    if len(numbers) == 0:
        return (0, 1)
    prod = 1
    for _, x in enumerate(numbers):
        prod *= x
    return (sum(numbers), prod)


def check(candidate):
    assert candidate([]) == (0, 1)
    assert candidate([1, 1, 1]) == (3, 1)
    assert candidate([100, 0]) == (100, 0)
    assert candidate([3, 5, 7]) == (15, 105)
    assert candidate([10]) == (10, 10)


if __name__ == "__main__":
    check(sum_product_prompt)
