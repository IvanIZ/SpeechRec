#!/usr/bin/env python3
# Copyright (c) 2019 Lawrence Angrave

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python
# https://stackoverflow.com/questions/56842391/how-to-get-word-level-timestamps-using-azure-speech-to-text-and-the-python-sdk
# What is "ITN" https://machinelearning.apple.com/2017/08/02/inverse-text-normal.html
# https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/samples/python/console
# https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/samples/python/console/speech_sample.py
# https://docs.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.recognitionresult?view=azure-python
# https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/rest-speech-to-text

# SDK docs - 
# https://docs.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.speechrecognitioneventargs?view=azure-python#result

# Batch (for the future)-
# https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/batch-transcription
# https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/batch-transcription#supported-formats


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



recognizers = []


def shutdown_recognizers():
    global recognizers
    while len(recognizers)>0:
        recognizer = recognizers.pop()
        try:
            # dont waste resources with any long running transcriptions
            recognizer.stop_continuous_recognition()
        except Error as ignored:
              print(ignored)


atexit.register(shutdown_recognizers)


def recognize_pcm_audio_file_to_ms_json(input_pcm_file, phraseList):
    """Performs speech recognition and returns MS-cognitive-services specific json array """
    print("call recognizer")
    global recognizers
    
    # <SpeechContinuousRecognitionWithFile>
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    
    speech_config.request_word_level_timestamps()
    #https://docs.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.profanityoption?view=azure-python
    speech_config.set_profanity(speechsdk.ProfanityOption.Masked)
    
    audio_config = speechsdk.audio.AudioConfig(filename=input_pcm_file)

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(recognizer)

    # add the phrase list for this section
    for phrase in phraseList:
        phrase_list_grammar.addPhrase(phrase)

    recognizers.append(recognizer)
    
    done = False
    json_results = []
    error_messages = []
    
    def stop_cb(event):
        if recognizer in recognizers:
            recognizers.remove(recognizer)
            recognizer.stop_continuous_recognition()
             # SDK docs claims error_details can be None. In practice it is an empty string for EOF cancel event, so using as a boolean treats both of these as false
            if event.cancellation_details.error_details:
                nonlocal error_messages
                error_messages.append( event.cancellation_details.error_details )
            # Do this last
            nonlocal done
            done = True
 
    def recognized_cb(event):
        nonlocal json_results
        print(event)
        # event.result.json is actually a string, so we parse it here to check for validity
        
        json_results.append( json.loads(event.result.json))

    recognizer.recognized.connect(recognized_cb) # Here are the words!

    # The MS SDK registers the stop_cb for both continuous recognition or canceled events. Canceled events may/are be genereated at EOF! :-(
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    if error_messages: 
        raise RuntimeError( ','.join(error_messages))
# truncated 1000byte PCM file - "RuntimeError: Exception with an error code: 0x9 (SPXERR_UNEXPECTED_EOF)
# Bad API key - "WebSocket Upgrade failed with an authentication error (401). Please check for correct subscription key (or authorization token) and region name.
            
    return json_results


def save_json(json_results, filename):
    with open(filename, 'w') as out_file:
        json.dump(json_results, out_file)



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


def main():
    # file_name = "anat_trim.mov"
    # file_name = "neuralScience_Trim_Trim.mov"
    file_name = "CS 412_Lecture12_Trim_Trim.mov"

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

    # pcm_file = "neuralScience_Trim_Trim.wav"
    # pcm_file = "anat_trim.wav"
    # pcm_file = "CS 412_Lecture12_Trim_Trim.wav"
    # json_file = "recognizedspeech.json"
    # json_results = recognize_pcm_audio_file_to_ms_json(pcm_file, phraseList)
    # save_json(json_results, json_file)

# Or put them directly here
speech_key, service_region = "1faf5f464d7844ecbe7c48efffc29993", "eastus"

# Generate json from speech: python ms_recognize_pcm.py myaudio.wav recognizedspeech.json
# Generate caption from json: python ms_json_to_caption.py recognizedspeech.json transcription.txt
if __name__== "__main__":
    # python ms_recognize_pcm.py toy_lecture.wav recognizedspeech.json
    main()

