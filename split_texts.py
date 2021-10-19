from typing import List
from make_xml import number_generator
from math import floor


def split_by_words(
    word_list: List[str], word_start: List[int], word_end: List[int], num_words=3
) -> "same":
    assert len(word_list) == len(word_start)
    assert len(word_start) == len(word_end)

    gen_start = number_generator(word_start)
    gen_end = number_generator(word_end)
    gen_word = number_generator(word_list)

    new_word_start = []
    new_word_end = []
    new_word_list = []

    num_iterations = floor(len(word_list) / num_words) * num_words
    new_start = []
    new_end = 0
    new_word = []
    
    for index in range(num_iterations):
        index += 1
        new_start.append(next(gen_start))
        new_end = next(gen_end)
        new_word.append(next(gen_word))
        
        if index % num_words == 0:
            new_word_start.append(new_start[0])
            new_word_end.append(new_end)
            new_word_list.append(" ".join(new_word))
            
            # reinitialize
            new_start = []
            new_end = 0
            new_word = []

    remainder = len(word_list) % num_words
    if remainder != 0:
        index -= 1
        new_start = []
        new_end = 0
        new_word = []
        
        for index in range(remainder):
            new_start.append(next(gen_start))
            new_end = next(gen_end)
            new_word.append(next(gen_word))
        
        new_word_start.append(new_start[0])
        new_word_end.append(new_end)
        new_word_list.append(" ".join(new_word))

    return new_word_list, new_word_start, new_word_end


def split_by_chars():
    pass


if __name__ == "__main__":
    A = split_by_words(["A", "B", "C"], [1, 5, 10], [3, 8, 11])
    print(A)
    A = split_by_words(["A", "B", "C", "D"], [1, 5, 10, 15], [3, 8, 11, 16])
    print(A)
    A = split_by_words(["A", "B", "C", "D", "E"], [1, 5, 10, 15, 20], [3, 8, 11, 16, 24])
    print(A)
    