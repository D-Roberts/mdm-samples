from typing import List


def intersperse_prompt(numbers: List[int], delimeter: int) -> List[int]:
    """Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse_prompt([], 4)
    []
    >>> intersperse_prompt([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    if len(numbers) == 0:
        return []
    result = [numbers[0]]
    for i, x in enumerate(numbers):
        if i == 0:
            continue
        if numbers[i] == numbers[i - 1] + 1:
            result.append(delimeter)
        result.append(x)
    return result


def check(candidate):
    assert candidate([], 7) == []
    assert candidate([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]
    assert candidate([2, 2, 2], 2) == [2, 2, 2, 2, 2]


if __name__ == "__main__":
    check(intersperse_prompt)
