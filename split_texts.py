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

    
    if (len(word_list) % num_words) != 0:
        index -= 1
        new_start = []
        new_end = 0
        new_word = []
        
        for index in range(index % num_words - 1):
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
    A = split_by_words(["A", "B", "C", "D"], [1, 5, 10, 15], [3, 8, 11, 16])
    print(A)
    A = split_by_words(['ARE', 'YOU', 'WONDERING', 'IF', 'HOU', 'CAN', 'KNOW', 'EVERYTHING', 'ABOUT', 'HOW', 'TO', 'KNOW', 'GOD', 'AND', 'BE', 'CLOSE', 'TO', 'HIM', 'NOW', 'THIS', 'VIVEO', 'WILL', 'PROVE', 'TO', 'YOU', 'THAT', 'THE', 'BIBLE', 'HAS', 'EVERYTHING', 'YOU', 'NEED', 'TO', 'KNOW', 'ABOUT', 'SALVATION', 'AN', 'SPIRITUAL', 'GROWTH', 'YIT', "DON'T", 'NEED', 'TO', 'LOOK', 'FOR', 'OTHER', 'WORDS', 'FROM', 'GOD', 'OR', 'OTHER', 'BOOKS', 'FROM', 'GOD', 'NOW', 'THE', 'WANT', 'ME', 'TO', 'PROVE', 'THISTY', 'GOOD', 'MORNIGHT', 'MY', 'FRIEND', 'WELCOME', 'BACK', 'TO', 'OUR', 'SERIES', 'IS', 'THE', 'BIBLE', 'RELIABLE', 'NOW', 'TO', 'THESE', 'TOPIC', 'IS', 'THE', 'LAST', 'ISTEP', 'IN', 'THE', 'SIX', 'SCARCASE', 'LIKE', 'PROCESSES', 'AND', 'HOW', 'GOD', 'REVEALED', 'HIS', 'INSPIRED', 'AND', 'IN', 'ERRANT', 'WORDS', 'TO', 'YOU', 'AND', 'ME', 'NOW', 'THESE', 'LACISTEP', 'AND', 'AM', 'EXCITED', 'TO', 'PREACHS', 'ABOUT', 'THIS', 'IS', 'CALLED', 'SUFFICIENCY', 'OF', 'SCRIPTURE', 'NOW', 'THERE', 'ARE', 'THREE', 'THINGS', 'IN', 'THIS', 'MESSAES', 'FIRST', "LET'S", 'DEFINE', 'SUFFICIENCY', 'THEN', 'PROVE', 'IT', 'AND', 'APPLI', 'IT', 'SO', 'FIRST', 'WHAT', 'IS', 'SUFFICIENCY', 'OF', 'SCRIPTURE', 'AND', 'WHY', 'IS', 'THIS', 'IMPORTANT', 'NOW', 'THE', 'DEFINITION', 'OF', 'THE', 'SUFFICIENCY', 'OF', 'SCRIPTURE', 'ISTIES', 'THE', 'SCRIPTURE', 'OR', 'THE', 'SIXTY', 'SIX', 'BOOKS', 'OF', 'THE', 'BIBLE', 'HAS', 'ALL', 'THE', 'WORDS', 'OF', 'GOD', 'HE', 'INTENDED', 'HIS', 'PEOPLE', 'TO', 'HAVE', 'FOUR', 'SALVATION', 'AND', 'SPIRITUAL', 'GROWTH', 'THE', 'BIBLE', 'IS', 'ALL', 'YOU', 'NEED', 'THERE', 'ARE', 'NO', 'OTHER', 'WORDS', 'OR', 'BOOKS', 'FROM', 'GOD', 'IN', 'ADDITION', 'TO', 'THE', 'BIBLE', 'IT', 'HAS', 'THE', 'FINALSAY', 'NOW', 'THUS', 'SUFFICIENCY', 'OF', 'SCRIPTURE', 'MEAN', 'THAT', 'WE', 'CANNOT', 'READ', 'BOOKS', 'OUTSIDE', 'THE', 'BIBLE', 'NO', "THAT'S", 'NOT', 'WHAT', 'IT', 'MEANS', 'SUFFICIENCY', 'MEANS', 'THAT', 'THE', 'BIBLE', 'ALONE', 'IS', 'OUR', 'FINAL', 'AUTHORITY', 'AS', 'LONG', 'AS', 'OTHER', 'BOOKS', 'AND', 'AUTHORITIES', 'DO', 'NOT', 'CONTRADICT', 'THE', 'BIBLE', 'THEN', 'THEY', 'ARE', 'HELPFUL', 'SO', "DON'T", 'FEAR', 'THE', 'USE', 'OF', 'EXTRABIBLICASORCES', 'LEARN', 'AND', 'UTILIZE', 'THEM', 'BUT', 'REALIZE', 'THAT', 'THEY', 'ARE', 'ALL', 'SUBORDINATE', 'TO', 'THE', 'BIBLE', 'FOR', 'EXAMPLE', 'THE', 'BIBLE', 'IS', 'NOT', 'EXHAUSTIVE', 'ABOUT', 'SCIENCE', 'SO', 'BY', 'ALL', 'MEANS', 'LEARN', 'EVERYTHING', 'ABOUT', 'SCIENCE', 'LEARN', 'EVERYTHING', 'ABOUT', 'MEDICINE', 'ARCHIOLOGY', 'PHILOSOPHY', 'LICTERATURE', 'MATHEMATICS', 'CULTURE', 'ARTIFICIAL', 'INTELLIGIENS', 'NOW', 'THESE', 'ARE', "GOD'S", 'COMMON', 'GRACE', 'TO', 'HUMANITY', 'BUT', 'WHEN', 'IT', 'COMES', 'TO', 'GODLINESS', 'THERE', 'IS', 'NO', 'NEED', 'TO', 'OD', 'OTHER', 'BOOKS', 'OR', 'AUTHORITIES', 'ONLY', 'THE', 'WORDS', 'OF', 'GOD', 'WHICH', 'WE', 'HAVE', 'IN', 'THE', 'BIBLE', 'ARE', 'ALL', 'THE', 'WORDS', 'OF', 'GOD', 'WE', 'NEED', 'TO', 'BE', 'SAVE'], [36, 44, 80, 126, 133, 145, 157, 182, 212, 251, 263, 270, 283, 330, 341, 363, 381, 389, 453, 463, 483, 513, 526, 541, 548, 580, 589, 597, 618, 641, 683, 698, 741, 751, 772, 788, 855, 878, 908, 960, 975, 994, 1007, 1014, 1027, 1042, 1064, 1096, 1108, 1144, 1161, 1175, 1193, 1205, 1269, 1277, 1288, 1296, 1303, 1310, 1323, 1402, 1418, 1441, 1449, 1487, 1509, 1521, 1525, 1552, 1591, 1603, 1613, 1659, 1726, 1733, 1741, 1765, 1796, 1805, 1821, 1849, 1894, 1899, 1912, 1948, 1991, 2008, 2076, 2084, 2101, 2140, 2163, 2173, 2229, 2236, 2250, 2266, 2296, 2305, 2321, 2329, 2372, 2386, 2411, 2465, 2473, 2481, 2513, 2520, 2533, 2544, 2585, 2594, 2637, 2700, 2707, 2777, 2785, 2795, 2801, 2813, 2829, 2834, 2846, 2906, 2961, 2978, 3022, 3096, 3141, 3159, 3210, 3253, 3274, 3329, 3338, 3398, 3424, 3428, 3510, 3518, 3585, 3595, 3624, 3632, 3664, 3743, 3750, 3758, 3788, 3793, 3800, 3846, 3853, 3880, 3947, 3953, 4024, 4047, 4058, 4077, 4090, 4103, 4109, 4115, 4152, 4185, 4197, 4207, 4230, 4243, 4285, 4306, 4348, 4368, 4387, 4396, 4446, 4462, 4520, 4548, 4576, 4641, 4651, 4693, 4714, 4736, 4746, 4804, 4815, 4826, 4849, 4862, 4896, 4907, 4924, 4937, 4969, 4984, 5022, 5029, 5035, 5097, 5122, 5163, 5170, 5247, 5302, 5311, 5373, 5380, 5410, 5429, 5438, 5447, 5465, 5478, 5500, 5518, 5524, 5590, 5650, 5664, 5672, 5680, 5686, 5697, 5784, 5815, 5825, 5835, 5861, 5908, 5921, 5941, 5994, 6079, 6087, 6098, 6109, 6124, 6144, 6153, 6190, 6207, 6230, 6260, 6271, 6319, 6333, 6343, 6351, 6427, 6440, 6481, 6523, 6536, 6550, 6567, 6679, 6727, 6738, 6765, 6820, 6830, 6899, 6907, 6915, 6938, 6957, 7009, 7016, 7022, 7076, 7086, 7154, 7161, 7180, 7187, 7198, 7248, 7267, 7331, 7338, 7351, 7361, 7398, 7430, 7460, 7482, 7536, 7566, 7588, 7621, 7683, 7757, 7823, 7898, 7969, 8030, 8063, 8127, 8137, 8179, 8189, 8220, 8239, 8282, 8289, 8379, 8391, 8403, 8410, 8428, 8436, 8511, 8529, 8540, 8554, 8592, 8596, 8627, 8640, 8670, 8685, 8755, 8767, 8774, 8791, 8800, 8817, 8830, 8839, 8854, 8860, 8867, 8915, 8938, 8950, 8961, 8986, 8994, 9028, 9043, 9081, 9089, 9096], [40, 53, 113, 128, 140, 151, 166, 205, 227, 258, 267, 278, 297, 334, 347, 376, 385, 396, 458, 472, 502, 521, 538, 545, 555, 586, 593, 614, 624, 673, 690, 713, 745, 762, 783, 820, 858, 903, 922, 967, 986, 1003, 1012, 1022, 1033, 1052, 1087, 1103, 1122, 1147, 1171, 1187, 1200, 1218, 1274, 1281, 1294, 1300, 1306, 1320, 1342, 1412, 1438, 1445, 1462, 1505, 1517, 1525, 1542, 1572, 1595, 1608, 1633, 1690, 1730, 1736, 1755, 1785, 1798, 1810, 1837, 1870, 1896, 1903, 1922, 1985, 2000, 2040, 2080, 2091, 2122, 2160, 2168, 2210, 2232, 2239, 2261, 2281, 2300, 2312, 2325, 2336, 2378, 2397, 2442, 2468, 2475, 2509, 2517, 2530, 2542, 2552, 2588, 2611, 2681, 2702, 2732, 2782, 2792, 2799, 2810, 2824, 2831, 2841, 2868, 2920, 2971, 3001, 3058, 3104, 3155, 3162, 3216, 3265, 3277, 3333, 3352, 3407, 3428, 3493, 3512, 3542, 3589, 3610, 3626, 3642, 3691, 3747, 3754, 3783, 3790, 3797, 3843, 3849, 3876, 3896, 3951, 3979, 4028, 4051, 4074, 4083, 4099, 4105, 4112, 4129, 4159, 4192, 4202, 4223, 4232, 4261, 4294, 4332, 4357, 4383, 4391, 4412, 4455, 4492, 4526, 4571, 4589, 4644, 4673, 4696, 4719, 4742, 4756, 4810, 4819, 4833, 4858, 4877, 4900, 4918, 4931, 4948, 4971, 5004, 5026, 5032, 5050, 5100, 5133, 5167, 5198, 5253, 5311, 5363, 5375, 5404, 5423, 5434, 5442, 5460, 5472, 5490, 5516, 5522, 5538, 5598, 5661, 5669, 5677, 5683, 5697, 5781, 5797, 5821, 5828, 5853, 5882, 5911, 5926, 5965, 6019, 6081, 6095, 6100, 6117, 6137, 6147, 6180, 6195, 6219, 6260, 6268, 6287, 6327, 6338, 6347, 6372, 6431, 6456, 6496, 6528, 6542, 6552, 6639, 6699, 6731, 6759, 6772, 6826, 6853, 6905, 6912, 6920, 6945, 6994, 7013, 7019, 7036, 7081, 7104, 7158, 7175, 7183, 7192, 7239, 7259, 7291, 7335, 7343, 7356, 7372, 7416, 7452, 7471, 7506, 7554, 7583, 7600, 7647, 7721, 7790, 7857, 7934, 7994, 8057, 8092, 8132, 8159, 8183, 8207, 8234, 8256, 8286, 8316, 8384, 8398, 8405, 8421, 8432, 8463, 8523, 8531, 8547, 8566, 8596, 8617, 8635, 8653, 8674, 8713, 8763, 8770, 8786, 8794, 8812, 8826, 8834, 8848, 8856, 8864, 8885, 8919, 8941, 8954, 8978, 8988, 9008, 9035, 9055, 9086, 9092, 9110])
    print(A)