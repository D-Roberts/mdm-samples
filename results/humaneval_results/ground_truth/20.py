def sort_numbers_prompt(numbers: str) -> str:
    """Input is a space-delimited string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted from smallest to largest
    >>> sort_numbers_prompt('three one five')
    'one three five'
    """


from typing import List


def sort_numbers(numbers: str) -> str:
    """Input is a space-delimited string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted from smallest to largest
    >>> sort_numbers('three one five')
    'one three five'
    """
    # Split the input string into a list of numbers
    numbers = numbers.split()

    # Convert each number to an integer
    num_list = [int(num) for num in numbers]

    # Sort the list of integers
    num_list.sort()

    # Convert the sorted list back to a string and join with spaces
    sorted_numbers_str = " ".join(str(num) for num in num_list)

    return sorted_numbers_str


# Example usage
if __name__ == "__main__":
    import doctest

    doctest.testmod()
