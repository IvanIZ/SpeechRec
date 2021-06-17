import azure.cognitiveservices.speech as speechsdk
import os
import sys
import atexit
import time
import json
from pydub import AudioSegment
from string import ascii_letters, digits
import VideoToText as VideoToText
import utils as utils

def split_audio(audio_path, start, end):
    """ A function that splits the input audio file based on a start
    time and an end time in seconds """

    start = start * 1000  # Works in milliseconds
    end = end * 1000
    newAudio = AudioSegment.from_wav(audio_path)
    newAudio = newAudio[start:end]
    newAudio.export('newSong.wav', format="wav")  # Exports to a wav file in the current path.

def process_phraseDict(phraseDict):
    """
    A function that cleans up each individual phrase dictionary in scene_text_dict, and leaving only number and char
    """

    # Filtering -- leaving just numbers and letters
    to_remove = []
    for phrase, count in phraseDict.items():
        if set(phrase).difference(ascii_letters + digits):
            to_remove.append(phrase)
    for word in to_remove:
        phraseDict.pop(word)

    return phraseDict


def convert_dict_to_phraseDict(phrase_dict):
    """
    A function that combines the multiple dictionaries from multiple scenes to one dictionary
    """
    phraseDict = {}
    for i in range(len(phrase_dict)):
        bag_of_word = phrase_dict[i]['bag_of_word']
        for word, count in bag_of_word.items():

            # put the word into the combined phrase dictionary
            cur_count = phraseDict.get(word, -1)
            if cur_count == -1:
                phraseDict[word] = count
            else:
                phraseDict[word] = cur_count + count

    return phraseDict

def extract_key_words(file_name):
    """
    Args:
        file_name: an mov file where the key words will be extracted

    Returns: a list of phrases that are extracted from the input video file. These are the extracted keywords

    """

    # cut the lecture videos into different scenes
    scenes = VideoToText.find_scenes(file_name, min_scene_length=1, abs_min=0.75, abs_max=0.98, find_subscenes=True,
                                     max_subscenes_per_minute=12)

    # for each scene, get its dictionary
    scene_text_dict, frequent_patterns = VideoToText.scene_to_text(scenes)
    print(scene_text_dict)
    print("Raw Dictionary ==================================================================================")

    # combine the multiple dictionary into one
    phraseDict = convert_dict_to_phraseDict(scene_text_dict)
    print(phraseDict)
    print("Combined raw dictionary ==================================================================================")

    # clean up the each dictionary by removing non-letters and non-numbers
    phraseDict = process_phraseDict(phraseDict)
    print(phraseDict)
    print("cleaned dictionaries ==================================================================================")

    # process the dictionary by comparing frequency with the brown corpus
    phraseList = utils.process_by_frequency(phraseDict)
    print(phraseList)
    print("initial phrase list ==================================================================================")

    # process the phrase list by removing stop words
    phraseList = utils.remove_stop_words(phraseList)
    print(phraseList)
    print("Final list after removing stop words ====================================================================")
    print()

    # extract frequent patterns and put them into a list
    print(frequent_patterns)
    print("frequent patterns ===============================================================================")
    print()
    frequent_patterns_list = utils.patterns_to_list(frequent_patterns)

    # combine the frequent patterns with single phrase list into one list
    phraseList = phraseList + frequent_patterns_list
    print(phraseList)
    print("Final phrase list ===============================================================================")
    print()

    return phraseList

