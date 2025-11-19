def how_many_times_prompt(string: str, substring: str) -> int:
    """Find how many times a given substring can be found in the original string. Count overlaping cases.
    >>> how_many_times_prompt('', 'a')
    0
    >>> how_many_times_prompt('aaa', 'a')
    3
    >>> how_many_times_prompt('aaaa', 'aa')
    3
    """


def how_many_times(string: str, substring: str) -> int:
    """Find how many times a given substring can be found in the original string. Count overlapping cases.
    >>> how_many_times('', 'a')
    0
    >>> how_many_times('aaa', 'a')
    3
    >>> how_many_times('aaaa', 'aa')
    3
    """
    count = 0
    start = 0
    while start < len(string):
        if string.find(substring, start) == -1:
            break
        count += 1
        start += len(substring)
    return count
