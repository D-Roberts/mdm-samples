def rolling_max_prompt(numbers: List[int]) -> List[int]:
    """From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.
    >>> rolling_max_prompt([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """


from typing import List, Tuple


def rolling_max(numbers: List[int]) -> List[int]:
    """From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """
    if not numbers:
        return []

    current_max = numbers[0]
    rolling_max = [current_max]

    for num in numbers[1:]:
        if num > current_max:
            current_max = num
        rolling_max.append(current_max)

    return rolling_max
