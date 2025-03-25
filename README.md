### Audio to Text Converter
This repository contains a comprehensive solution for converting audio or video clips to text format. The solution is divided into three main steps:
1. Converting audio to text.
2. Recognizing speaker IDs and the duration of their speech.
3. Recognizing multiple speakers along with their IDs, timestamps, and the spoken text.

##### Features
1. Audio to Text Conversion: Converts any audio or video clip to text using Google's Speech Recognition API.
2. Speaker Diarization: Identifies different speakers in an audio clip and provides their IDs and speech durations.
3. Multispeaker Recognition: Extracts speech information for multiple speakers, including speaker IDs, timestamps, and the actual text spoken. 

##### Requirements
1. Python 3.6+
2. Libraries: speech_recognition, pyautogui, pyperclip, time, pydub, torch, pyannote.audio 

##### Install the required libraries using:
pip install speechrecognition pyautogui pyperclip pydub torch pyannote.audio 

##### Setup
Ensure you have your Hugging Face authentication token ready for the speaker diarization model.  

##### Usage
Step 1: Convert Audio to Text
This script converts any audio or video clip to text format using Google's Speech Recognition API. 

import speech_recognition as sr
import pyautogui
import pyperclip
import time
from pydub import AudioSegment
import os

def convert_to_wav(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    wav_file_path = os.path.splitext(audio_file_path)[0] + ".wav"
    audio.export(wav_file_path, format="wav")
    return wav_file_path

def voice_typing_from_file(audio_file_path):
    recognizer = sr.Recognizer()
    wav_file_path = convert_to_wav(audio_file_path)

    try:
        with sr.AudioFile(wav_file_path) as source:
            print("Processing audio file...")
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language='te-IN')
        print("Recognized:", text)

    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    except Exception as e:
        print("An error occurred:", e)
    finally:
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path) 

if __name__ == "__main__":
    audio_file_path = "path_to_your_audio_file"
    voice_typing_from_file(audio_file_path) 

Step 2: Recognize Speaker IDs and Duration
This script identifies different speakers in an audio clip and provides their IDs and speech durations. 

import os
from pydub import AudioSegment
import speech_recognition as sr
from pyannote.audio import Pipeline
import torch

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="your_huggingface_auth_token"
)

if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

diarization = pipeline("path_to_your_audio_file")

with open("output_audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker} from {turn.start:.1f}s to {turn.end:.1f}s") 

Step 3: Recognize Multiple Speakers with Text
This script extracts speech information for multiple speakers, including speaker IDs, timestamps, and the spoken text. 

import os
from pydub import AudioSegment
import speech_recognition as sr
from pyannote.audio import Pipeline
import torch

def convert_to_wav(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    wav_file_path = os.path.splitext(audio_file_path)[0] + ".wav"
    audio.export(wav_file_path, format="wav")
    return wav_file_path

def recognize_speech_from_audio_segment(segment, language='te-IN'):
    recognizer = sr.Recognizer()
    wav_file_path = "temp_segment.wav"
    segment.export(wav_file_path, format="wav")

    try:
        with sr.AudioFile(wav_file_path) as source:
            audio = recognizer.record(source)
        
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    finally:
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)

def voice_typing_from_file(audio_file_path):
    wav_file_path = convert_to_wav(audio_file_path)
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="your_huggingface_auth_token"
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    diarization = pipeline(wav_file_path)
    
    audio = AudioSegment.from_wav(wav_file_path)
    results = []

    current_speaker = None
    current_start_time = None
    current_text = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start * 1000
        end_time = turn.end * 1000
        segment = audio[start_time:end_time]
        
        recognized_text = recognize_speech_from_audio_segment(segment)

        if speaker == current_speaker:
            current_text.append(recognized_text)
            current_end_time = end_time
        else:
            if current_speaker is not None:
                results.append((current_speaker, current_start_time, current_end_time, ' '.join(current_text)))
            current_speaker = speaker
            current_start_time = start_time
            current_end_time = end_time
            current_text = [recognized_text]
    
    if current_speaker is not None:
        results.append((current_speaker, current_start_time, current_end_time, ' '.join(current_text)))

    for speaker, start_time, end_time, text in results:
        print(f"Speaker {speaker} from {start_time / 1000:.1f}s to {end_time / 1000:.1f}s: {text}")

    if os.path.exists(wav_file_path):
        os.remove(wav_file_path)

if __name__ == "__main__":
    audio_file_path = "path_to_your_audio_file"
    voice_typing_from_file(audio_file_path) 

##### Sample output:

Speaker SPEAKER_01 from 1.3s to 18.4s: ఒక రోజు ఒక అడవిలో తాబేలు నెమ్మదిగా వెళుతూ ఉంటే ఒక కుందేలు తన దగ్గరికి వచ్చి తనను వెక్కిరిస్తుంది నువ్వు ఇంత మెల్లగా వెళ్తున్నావ్ ఏంటి అప్పుడు వాళ్ళిద్దరూ ఒక పోటీ పెట్టుకుంటారు ఎవరు ముందుగా ఒక ఆ ప్రదేశాన్ని చేరుకుంటారు

Speaker SPEAKER_02 from 19.5s to 41.0s: వెంటనే తాబేలు కళ్ళు తన లోపలికి కదలకుండా ఉండిపోయింది నాకు తాబేలు దగ్గరికి వెళ్లి దాన్ని పట్టుకొని చూసింది పైన తప్ప గట్టిగా తగలలేదు తాబేలు తిరిగేసి మూతి దగ్గరికి పెట్టుకుని ఇలా నక్క తనను పరీక్షిస్తూ ఎంతసేపు తాబేలు ప్రాణాలు అరచేతిలో పెట్టుకుని ఊపిరి పట్టుకొని ఉన్నది

Speaker SPEAKER_00 from 41.4s to 59.5s: ఒక అడవిలోని చెరువులో ఒక తాబేలు ఉండేది ఒక రోజు సాయంత్రం అది నీటి నుంచి బయటికి వచ్చి ఒడ్డున చేరుకుంది ఇంతలో అక్కడికి ఒక నక్క వచ్చింది దాన్ని చూసి నీటిలోకి వెళ్లిపోవాల్సి వచ్చింది తాబేలు కానీ ఇంతలో నక్క దాన్ని చూసింది వెంటనే తాబేలు కళ్ళు తల లోపలికి లో




























