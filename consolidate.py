import pickle
from tqdm.contrib import tenumerate
from typing import List


def char_gen(string):
    for ch in string:
        yield ch


def time_decoder(decoded: str, batch_decoded: List) -> List[int]:
    """
    This is a function that takes in two arguments:
    decoded: A string of decoded text
    batch_decoded: A list of strings of decoded text

    It returns a list of integers, which are the indices of the start
    of each word in the decoded string.

    This is done by iterating through the characters in the decoded
    string, and comparing them to the characters in the batch_decoded
    list. If a character matches, it is added to a list of indices.
    Once all characters have been compared, the function returns the
    list of indices.

    For example, if decoded = "hello world", and
    batch_decoded = ["hello", "world","again"],
    then this function would return [0, 1]
    """
    gen_decoded = char_gen(decoded)
    decoded_index = 0
    char = next(gen_decoded)

    add_char = True
    start_list = []
    end_list = []

    for index, token in tenumerate(batch_decoded):
        if token == char:
            if add_char == True:
                start_list.append(index)
                add_char = False

            decoded_index += 1
            if decoded_index == len(decoded):
                break
            else:
                char = next(gen_decoded)
                if char == " ":
                    end_list.append(index + 1)
                    decoded_index += 1
                    if decoded_index == len(decoded):
                        break
                    else:
                        add_char = True
                        char = next(gen_decoded)

    if len(end_list) != len(start_list):
        last_letter = decoded
        list_reversed = batch_decoded[::-1]
        index_last_letter = len(batch_decoded) - list_reversed.index(decoded[-1])
        end_list.append(index_last_letter)

    return start_list, end_list

