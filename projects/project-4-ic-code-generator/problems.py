"""
10 test problems for the Iterative Code Generator (4-I-C).

Each problem specifies the exact function name the generated code must
define, a prompt precise enough to be unambiguous (fibonacci indexing,
tie-breaking, etc. are all pinned down explicitly -- an underspecified
prompt would make a wrong answer look like a generator failure when it
was actually a spec failure), and test cases as (args, expected) pairs
checked by direct equality.
"""

PROBLEMS = [
    {
        "id": "is_palindrome",
        "function_name": "is_palindrome",
        "prompt": (
            "Write a function is_palindrome(s) that returns True if the string s "
            "reads the same forwards and backwards, ignoring case and ignoring any "
            "character that is not a letter or digit, and False otherwise."
        ),
        "test_cases": [
            {"args": ["racecar"], "expected": True},
            {"args": ["hello"], "expected": False},
            {"args": ["A man, a plan, a canal: Panama"], "expected": True},
            {"args": [""], "expected": True},
        ],
    },
    {
        "id": "fizzbuzz",
        "function_name": "fizzbuzz",
        "prompt": (
            "Write a function fizzbuzz(n) that returns a list of strings for the "
            "numbers 1 to n inclusive: 'Fizz' if divisible by 3, 'Buzz' if divisible "
            "by 5, 'FizzBuzz' if divisible by both, otherwise the number itself as a string."
        ),
        "test_cases": [
            {"args": [5], "expected": ["1", "2", "Fizz", "4", "Buzz"]},
            {
                "args": [15],
                "expected": ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz",
                             "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"],
            },
        ],
    },
    {
        "id": "reverse_words",
        "function_name": "reverse_words",
        "prompt": (
            "Write a function reverse_words(s) that reverses the order of words in "
            "the string s (words are separated by single spaces, no leading/trailing "
            "or repeated spaces) and returns the result as a single space-separated string."
        ),
        "test_cases": [
            {"args": ["hello world"], "expected": "world hello"},
            {"args": ["the quick brown fox"], "expected": "fox brown quick the"},
            {"args": ["single"], "expected": "single"},
        ],
    },
    {
        "id": "is_prime",
        "function_name": "is_prime",
        "prompt": (
            "Write a function is_prime(n) that returns True if the non-negative "
            "integer n is a prime number, and False otherwise. Note 0 and 1 are not prime."
        ),
        "test_cases": [
            {"args": [2], "expected": True},
            {"args": [1], "expected": False},
            {"args": [0], "expected": False},
            {"args": [17], "expected": True},
            {"args": [18], "expected": False},
        ],
    },
    {
        "id": "fibonacci",
        "function_name": "fibonacci",
        "prompt": (
            "Write a function fibonacci(n) that returns the nth Fibonacci number, "
            "0-indexed, where fibonacci(0) = 0 and fibonacci(1) = 1."
        ),
        "test_cases": [
            {"args": [0], "expected": 0},
            {"args": [1], "expected": 1},
            {"args": [7], "expected": 13},
            {"args": [10], "expected": 55},
        ],
    },
    {
        "id": "count_vowels",
        "function_name": "count_vowels",
        "prompt": (
            "Write a function count_vowels(s) that returns the count of vowels "
            "(a, e, i, o, u -- case-insensitive) in the string s."
        ),
        "test_cases": [
            {"args": ["hello world"], "expected": 3},
            {"args": ["AEIOU"], "expected": 5},
            {"args": ["xyz"], "expected": 0},
        ],
    },
    {
        "id": "remove_duplicates",
        "function_name": "remove_duplicates",
        "prompt": (
            "Write a function remove_duplicates(lst) that returns a new list with "
            "duplicate values removed, preserving the order of first occurrence."
        ),
        "test_cases": [
            {"args": [[1, 2, 2, 3, 1, 4]], "expected": [1, 2, 3, 4]},
            {"args": [[]], "expected": []},
            {"args": [[5, 5, 5]], "expected": [5]},
        ],
    },
    {
        "id": "binary_search",
        "function_name": "binary_search",
        "prompt": (
            "Write a function binary_search(arr, target) that performs binary "
            "search on the sorted list arr for target, returning the index if "
            "found, or -1 if not found."
        ),
        "test_cases": [
            {"args": [[1, 3, 5, 7, 9, 11], 7], "expected": 3},
            {"args": [[1, 3, 5, 7, 9, 11], 4], "expected": -1},
            {"args": [[], 5], "expected": -1},
            {"args": [[2], 2], "expected": 0},
        ],
    },
    {
        "id": "merge_sorted",
        "function_name": "merge_sorted",
        "prompt": (
            "Write a function merge_sorted(a, b) that merges two sorted lists of "
            "integers a and b into a single sorted list."
        ),
        "test_cases": [
            {"args": [[1, 3, 5], [2, 4, 6]], "expected": [1, 2, 3, 4, 5, 6]},
            {"args": [[], [1, 2, 3]], "expected": [1, 2, 3]},
            {"args": [[1, 1, 2], [1, 3]], "expected": [1, 1, 1, 2, 3]},
        ],
    },
    {
        "id": "most_frequent",
        "function_name": "most_frequent",
        "prompt": (
            "Write a function most_frequent(lst) that returns the element that "
            "appears most frequently in the non-empty list lst. Assume there is "
            "always a unique most-frequent element (no ties)."
        ),
        "test_cases": [
            {"args": [[1, 2, 2, 3, 2]], "expected": 2},
            {"args": [["a", "b", "a", "a", "c"]], "expected": "a"},
            {"args": [[7]], "expected": 7},
        ],
    },
    {
        "id": "string_compress",
        "function_name": "string_compress",
        "prompt": (
            "Write a function string_compress(s) that compresses repeated consecutive "
            "characters into character-count pairs (e.g. 'aabcccccaaa' -> 'a2b1c5a3'). "
            "If the compressed result is not shorter than the original string, return the original string."
        ),
        "test_cases": [
            {"args": ["aabcccccaaa"], "expected": "a2b1c5a3"},
            {"args": ["abcd"], "expected": "abcd"},
            {"args": ["a"], "expected": "a"},
            {"args": ["aabbcc"], "expected": "aabbcc"},
        ],
    },
    {
        "id": "eval_rpn",
        "function_name": "eval_rpn",
        "prompt": (
            "Write a function eval_rpn(tokens) that evaluates Reverse Polish Notation "
            "(a list of string tokens). Valid operators are '+', '-', '*', and '/'. "
            "Division between two integers should truncate toward zero."
        ),
        "test_cases": [
            {"args": [["2", "1", "+", "3", "*"]], "expected": 9},
            {"args": [["4", "13", "5", "/", "+"]], "expected": 6},
            {"args": [["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]], "expected": 22},
            {"args": [["6", "-132", "/"]], "expected": 0},
        ],
    },
    {
        "id": "flatten_list",
        "function_name": "flatten_list",
        "prompt": (
            "Write a function flatten_list(nested) that takes a nested list structure "
            "and returns a single flat list containing all non-list elements in order."
        ),
        "test_cases": [
            {"args": [[1, [2, [3, 4], 5], "hello"]], "expected": [1, 2, 3, 4, 5, "hello"]},
            {"args": [[["a", "b"], []]], "expected": ["a", "b"]},
            {"args": [[]], "expected": []},
        ],
    },
    {
        "id": "longest_consecutive",
        "function_name": "longest_consecutive",
        "prompt": (
            "Write a function longest_consecutive(nums) that returns the length of the longest "
            "consecutive elements sequence in an unsorted list of integers nums."
        ),
        "test_cases": [
            {"args": [[100, 4, 200, 1, 3, 2]], "expected": 4},
            {"args": [[0, 3, 7, 2, 5, 8, 4, 6, 0, 1]], "expected": 9},
            {"args": [[1, 2, 0, 1]], "expected": 3},
            {"args": [[]], "expected": 0},
        ],
    },
    {
        "id": "snake_to_camel",
        "function_name": "snake_to_camel",
        "prompt": (
            "Write a function snake_to_camel(s) that converts a snake_case string to camelCase "
            "(e.g., 'user_id' -> 'userId'). Any leading or trailing underscores must be preserved."
        ),
        "test_cases": [
            {"args": ["foo_bar"], "expected": "fooBar"},
            {"args": ["__foo_bar__"], "expected": "__fooBar__"},
            {"args": ["single"], "expected": "single"},
            {"args": ["user_id_number"], "expected": "userIdNumber"},
        ],
    },
    {
        "id": "parse_query_string",
        "function_name": "parse_query_string",
        "prompt": (
            "Write a function parse_query_string(query) that parses a URL query string "
            "(without the leading '?') into a dictionary of key-value pairs. Multiple values for "
            "the same key should be gathered into a list of string values. Keys with no '=' sign "
            "should have a boolean True value, while keys with an '=' sign but no value (e.g., 'a=') "
            "should have an empty string '' value. An empty query string returns an empty dict."
        ),
        "test_cases": [
            {"args": ["a=1&b=2"], "expected": {"a": "1", "b": "2"}},
            {"args": ["a=1&a=2&b=3"], "expected": {"a": ["1", "2"], "b": "3"}},
            {"args": ["flag&debug=false"], "expected": {"flag": True, "debug": "false"}},
            {"args": ["a="], "expected": {"a": ""}},
            {"args": [""], "expected": {}},
        ],
    },
]

PROBLEMS_BY_ID = {p["id"]: p for p in PROBLEMS}
