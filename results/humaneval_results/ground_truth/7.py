def parse_nested_parens_prompt(paren_string: str) -> List[int]:
    """Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
    For each of the group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.

    >>> parse_nested_parens_prompt('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """


from typing import List


def parse_nested_parens(paren_string: str) -> List[int]:
    """Input to this function is a string representing multiple groups of nested parentheses separated by spaces.
    For each group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.

    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
    groups = paren_string.split()
    results = []
    for group in groups:
        stack = []
        current_level = 0
        max_level = 0
        for char in group:
            if char == "(":
                stack.append(current_level)
                current_level += 1
            elif char == ")":
                current_level -= 1
                if stack:
                    max_level = max(max_level, stack.pop())
        results.append(max_level)
    return results


# Example usage
if __name__ == "__main__":
    import doctest

    doctest.testmod()
