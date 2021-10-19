'''
length_of_audio:
word_chunks: chunks of audio
fps: usually 1/24th of a second
wcps: num of chunks/seconds
'''

import html
from typing import List, Union


def number_generator(num_list):
    for i in num_list:
        yield i

def make_xml_from_words(word_start: List[int], word_list: List[str], wcps: Union[float, int], word_end: List[int]) -> str:
    """
    makes a fcpxml file from word chunks
    """
    # sanity checks
    assert len(word_start) == len(word_list)
    assert len(word_start) == len(word_end)
    start_gen = number_generator(word_start)
    end_gen = number_generator(word_end)
    for i in range(len(word_start)):
        assert (next(end_gen) - next(start_gen)) > 0

    fps = 24
    
    # convert starting chunks to fps
    temp = []
    for start_int in word_start:
        temp.append(int(start_int/wcps*fps))
    word_start = temp
    
    # make duration list
    duration_list = []
    start_gen = number_generator(word_start)
    for end_int in word_end:
        end_int = int(end_int/wcps*fps)
        duration_list.append(end_int - next(start_gen))

    text_list = []

    begin = """<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.9">
        <resources>
            <format height="1080" width="1920" name="FFVideoFormat1080p24" id="r0" frameDuration="1/24s"/>
            <effect name="Basic Title" uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti" id="r1"/>
        </resources>
        <library>
            <event name="Original (Resolve)">
                <project name="Original (Resolve)">
                    <sequence format="r0" tcStart="3600/1s" tcFormat="NDF" duration="20/1s">
                        <spine>
                            <gap start="3600/1s" name="Gap" offset="3600/1s" duration="20/1s">
    """

    text_list.append(begin)

    text = """                            <title start="{}/24s" lane="1" name="Rich" offset="{}/24s" ref="r1" enabled="1" duration="{}/24s">
                                    <text roll-up-height="0">
                                        <text-style ref="ts{}">{}</text-style>
                                    </text>
                                    <text-style-def id="ts{}">
                                        <text-style font="Helvetica" fontSize="96" italic="0" alignment="center" strokeColor="0 0 0 1" lineSpacing="0" fontColor="1 1 1 1" strokeWidth="0" bold="1"/>
                                    </text-style-def>
                                    <adjust-transform position="0 0" scale="1 1" anchor="0 0"/>
                                </title>
    """

    start_gen = number_generator(word_start)
    duration_gen = number_generator(duration_list)
    for word_index, word in enumerate(word_list):
        start = next(start_gen) + 3600*24 # beginning always starts at 3600 seconds
        duration = next(duration_gen)
        text_list.append(text.format(start, start, duration, word_index, html.escape(word), word_index))


    end = """                        </gap>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
    """

    text_list.append(end)

    text = ''.join(text_list)

    return text


if __name__ == "__main__":
    wcps = 16000
    word_start = [5*wcps, 11*wcps, 17*wcps] # seconds * word_chunks
    word_end = [11*wcps, 17*wcps, 21*wcps]  # seconds * word_chunks
    word_list = ["A", "B", "C"]
    with open("test.fcpxml", "w") as file_handler:
        file_handler.write(make_xml_from_words(word_start=word_start, 
                                word_end=word_end,
                                word_list=word_list,
                                wcps=wcps,
        ))
