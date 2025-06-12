
from pubmed_submission import submission_one
from pubmed_submission import overall_max
import re
def check_response(response):
    if "apologies" in response.lower():
        return False
    if "sorry" in response.lower():
        return False
    if "unable to fulfill" in response.lower():
        return False
    if "not able to fulfill" in response.lower():
        return False
    if "cannot fulfill" in response.lower():
        return False
    if "unable to complete" in response.lower():
        return False
    if "can't proceed" in response.lower():
        return False
    if "don't have the capability" in response.lower():
        return False
    response_num_tokens = len(response.split())
    if response_num_tokens > 5000:
        return False
    return True


def check_yes_no(response):
    if "yes" in response.lower():
        return True
    if "no" in response.lower():
        return False
    return False



def check_correct(bool_query, mindate, maxdate):
    # this method checks if a boolean query is correct
    # if the query is correct, it returns True
    # if the query is incorrect, it returns False
    # it needs to be correct in two ways: (1) submit to pubmed and return > 0 results, and < overall_max results
    # (2) bool_query should be logically correct
    # if the query is incorrect, it returns False
    # if the query is correct, it returns True
    if not check_logic(bool_query):
        return False
    if not check_submission(bool_query, mindate, maxdate):
        return False
    return True

import re

def check_submission(query, mindate, maxdate):
    count, current_id_list = submission_one(query, mindate, maxdate)
    print(count)
    if len(current_id_list) == 0 or count > overall_max:
        return False
    return True




def check_logic(bool_query):
    """
    Performs basic logic checks on a boolean query string.
    Returns:
        True if all checks pass (the query is logically correct).
        False otherwise.
    """
    # None check
    if bool_query is None:
        return False
    # --------------------------------------------------------------------
    # 1) Check for balanced parentheses
    # --------------------------------------------------------------------

    stack = []

    for char in bool_query:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                # There's a closing parenthesis without a matching opening one
                return False
            stack.pop()
    if stack:
        # Not all parentheses were closed
        return False

    # --------------------------------------------------------------------
    # 2) Tokenize the query
    # --------------------------------------------------------------------
    # A simple approach is to split on whitespace, but also keep parentheses
    # as separate tokens for checks. One naive way is to replace parentheses
    # with spaced versions, then split:

    spaced_query = bool_query.replace('(', ' ( ').replace(')', ' ) ')
    tokens = spaced_query.split()

    # We'll define a set of recognized operators.
    # If you have more operators or variations, add them here.
    valid_operators = {'AND', 'OR', 'NOT'}

    # --------------------------------------------------------------------
    # 3) Ensure tokens are in a logical order
    # --------------------------------------------------------------------
    # Basic rules:
    #   - Query shouldn't start with AND/OR
    #   - Query shouldn't end with AND/OR/NOT
    #   - Operators shouldn't be directly adjacent (e.g. "AND OR")
    #   - Parentheses should enclose valid sequences (already partially enforced with stack logic)
    #
    # More robust logic can be implemented by building an actual parser
    # but below is a straightforward approach.

    if tokens:
        # Check start token
        if tokens[0] in {'AND', 'OR'}:
            return False

        # Check end token
        if tokens[-1] in valid_operators:
            return False

    previous_token = None
    for i, token in enumerate(tokens):
        # Check for valid operators vs. other tokens
        if token in valid_operators:
            # Consecutive operators check
            if previous_token in valid_operators:
                # e.g., "AND OR" or "OR NOT" -- not logically correct
                return False
        else:
            # If not an operator or parentheses, we treat it as a term/phrase
            # or just a single word. You could add further checks here
            # (e.g., allowed characters, quoted phrases, etc.)
            if token not in {'(', ')'}:
                # Additional checks for the term can go here if needed
                pass

        previous_token = token

    # --------------------------------------------------------------------
    # 4) Check NOT usage
    # --------------------------------------------------------------------
    # This is an optional check depending on how strict you want to be
    # with "NOT" usage. For instance, you might want to ensure that
    # "NOT" always precedes a valid term or parentheses, but never stands alone.
    #
    # Example minimal check: "NOT" is followed by something other than AND/OR.
    for i, token in enumerate(tokens):
        if token == 'NOT':
            # If there's no token after NOT, that's a problem
            if i == len(tokens) - 1:
                return False
            # If the next token is also an operator (AND/OR/NOT) or a closing paren, that's suspicious
            if tokens[i+1] in valid_operators.union({')'}):
                return False

    # If we've passed all checks, we assume the query is logically correct
    return True