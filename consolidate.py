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


if __name__ == "__main__":
    decoded = 'ARE YOU WONDERING IF YOU CAN KNOW EVERYTHING ABOUT HOW TO KNOW GOD AND BE CLOSE TO HIM NOW THIS VIVIO WILL PROVE TO YOU THAT THE BAR'
    batch_decoded = ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'A', 'R', 'E', '<pad>', '', 
                     '', '<pad>', 'Y', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'O', 'U', '<pad>', '<pad>', 
                     '<pad>', '', '', '', '', '', '', '', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', 'W', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'O', 'N', 'N', 
                     '<pad>', '<pad>', 'D', 'D', '<pad>', '<pad>', '<pad>', 'E', '<pad>', 'R', 'R', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'I', 'N', 'N', 'G', 'G', '<pad>', '', 
                     '', '', '', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'I', 
                     'F', 'F', '<pad>', '', '', '<pad>', '<pad>', 'Y', '<pad>', '<pad>', 'O', 'U', '<pad>', 
                     '<pad>', '', '', '<pad>', '<pad>', 'C', '<pad>', '<pad>', '<pad>', 'A', 'N', 'N', '<pad>', 
                     '<pad>', '', '', '<pad>', 'K', 'K', 'N', 'N', '<pad>', '<pad>', '<pad>', 'O', 'W', 'W', 
                     '<pad>', '', '', '', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', 'E', 'V', 'V', '<pad>', '<pad>', '<pad>', 'E', 'R', 'R', 
                     '<pad>', '<pad>', 'Y', '<pad>', '<pad>', 'T', 'H', '<pad>', '<pad>', 'I', 'N', '<pad>', 'G', 
                     '<pad>', '', '', '<pad>', '<pad>', '<pad>', '<pad>', 'A', '<pad>', 'B', 'B', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', 'O', 'U', '<pad>', '<pad>', '<pad>', 'T', 'T', '<pad>', 
                     '', '', '', '', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'H', '<pad>', 
                     '<pad>', '<pad>', '<pad>', 'O', 'W', '<pad>', '<pad>', '', '', '<pad>', 'T', '<pad>', '<pad>', 
                     'O', '<pad>', '', '<pad>', 'K', 'K', 'N', '<pad>', '<pad>', '<pad>', 'O', 'W', '<pad>', '', '', 
                     '', '<pad>', 'G', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'O', 
                     '<pad>', '<pad>', '<pad>', 'D', '<pad>', '', '', '', '', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', 'A', 'N', '<pad>', 'D', 'D', '', '', '', '<pad>', '<pad>', 'B', '<pad>', 
                     '<pad>', '<pad>', '<pad>', 'E', '<pad>', '<pad>', '<pad>', '<pad>', '', '', '', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'C', '<pad>', 'L', 'L', '<pad>', 
                     '<pad>', '<pad>', 'O', '<pad>', '<pad>', '<pad>', 'S', '<pad>', 'E', 'E', '<pad>', '', '', 'T', 
                     '<pad>', '<pad>', 'O', '<pad>', '', '', '', '<pad>', 'H', '<pad>', '<pad>', '<pad>', '<pad>', 'I', 
                     'M', '<pad>', '', '', '', '', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'N', '<pad>', 
                     'O', 'O', 'W', '', '', '', '<pad>', '<pad>', 'T', 'H', '<pad>', '<pad>', '<pad>', 'I', '<pad>', 
                     '<pad>', 'S', '<pad>', '<pad>', '', '', '', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'V', 
                     '<pad>', '<pad>', '<pad>', '<pad>', 'I', '<pad>', '<pad>', '<pad>', '<pad>', 'V', '<pad>', '<pad>', 
                     '<pad>', 'I', '<pad>', '<pad>', '<pad>', '<pad>', 'O', '<pad>', '<pad>', '<pad>', '<pad>', '', '', 
                     '<pad>', '<pad>', '<pad>', '<pad>', 'W', '<pad>', '<pad>', 'I', 'L', '<pad>', '<pad>', 'L', '', '', '', 
                     '<pad>', '<pad>', 'P', 'P', 'R', 'R', '<pad>', '<pad>', '<pad>', '<pad>', 'O', '<pad>', 'V', 'E', 
                     '<pad>', '', '', 'T', '<pad>', '<pad>', 'O', '<pad>', '', '<pad>', 'Y', 'Y', '<pad>', '<pad>', 'O', 'U', 
                     '<pad>', '<pad>', '', '', '', '', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
                     '<pad>', 'T', 'H', '<pad>', '<pad>', 'A', 'T', '<pad>', '', '', '<pad>', 'T', 'H', 'E', '<pad>', '', '', 
                     'B', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'A', 'R', '<pad>']
    
    start_list, end_list = time_decoder(decoded, batch_decoded)
    print(start_list, end_list)
    print(len(start_list) - len(end_list))
