def intersperse_prompt(numbers: List[int], delimeter: int) -> List[int]:
    """Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse_prompt([], 4)
    []
    >>> intersperse_prompt([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """


from typing import List


def intersperse(numbers: List[int], delimiter: int) -> List[int]:
    """Insert a number 'delimiter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    if not numbers:
        return []

    result = []
    for i in range(len(numbers)):
        if i == 0:
            result.append(numbers[i])
        else:
            result.append(delimiter)
            result.append(numbers[i])

    return result
