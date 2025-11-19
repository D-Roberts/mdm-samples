def longest_prompt(strings: List[str]) -> Optional[str]:
    """Out of list of strings, return the longest_prompt one. Return the first one in case of multiple
    strings of the same length. Return None in case the input list is empty.
    >>> longest_prompt([])

    >>> longest_prompt(['a', 'b', 'c'])
    'a'
    >>> longest_prompt(['a', 'bb', 'ccc'])
    'ccc'
    """


from typing import List, Optional


def longest(strings: List[str]) -> Optional[str]:
    """Out of list of strings, return the longest one. Return the first one in case of multiple
    strings of the same length. Return None in case the input list is empty.
    >>> longest([])
    >>> longest(['a', 'b', 'c'])
    'a'
    >>> longest(['a', 'bb', 'ccc'])
    'ccc'
    """
    if not strings:
        return None

    max_length = 0
    longest_string = None

    for s in strings:
        if len(s) > max_length:
            max_length = len(s)
            longest_string = s
        elif len(s) == max_length:
            if longest_string is None:
                longest_string = s

    return longest_string
