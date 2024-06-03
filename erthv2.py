import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import openai
import nltk
from nltk.tokenize import word_tokenize
import json
import os
import wave
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play

nltk.download('punkt')

#Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')

st.image("Erth.png", use_column_width=True)  
st.title("Erth | إرث")


def load_gpt2_model(path):
    model = GPT2LMHeadModel.from_pretrained(path)
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    return model, tokenizer

#AraGPT2 model & Tokenizer
# model = GPT2LMHeadModel.from_pretrained("ft_medium_aragpt2_saudi_traditions")
# tokenizer = GPT2Tokenizer.from_pretrained("ft_medium_aragpt2_saudi_traditions")

gpt2_model, gpt2_tokenizer = load_gpt2_model("rarayayan/testftargpt2")

#openai_api_key = st.secrets["OPENAI_API_KEY"]
#openai.api_key = openai_api_key

# Function to tokenize text using NLTK
def nltk_tokenize(text):
    return word_tokenize(text)

#generate GPT-2 response
def generate_gpt2_response(prompt):
    tokenized_prompt = nltk_tokenize(prompt)
    tokenized_prompt = " ".join(tokenized_prompt)
    input_ids = gpt2_tokenizer.encode(tokenized_prompt, return_tensors='pt')
    response_ids = gpt2_model.generate(
        input_ids,
        max_length=150,
        pad_token_id=gpt2_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
    )
    response = gpt2_tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

#generate OpenAI response
def generate_openai_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

#for the speech-to-speech
CHUNK_SIZE = 1024
XI_API_KEY = "86a2e649df2493b168b33dc4f201378b"
VOICE_ID = "uPnpjwLtM7boX0RTyswZ"
OUTPUT_AUDIO_FILE = "output.mp3"
OPENAI_API_KEY = 'sk-proj-QrddsmWXf1a9RDZGBfy6T3BlbkFJ2JPHcaauPS3FioWhUk8O'
RECORD_SECONDS = 10
INPUT_AUDIO_FILE = "input.wav"

#OpenAI API key
openai.api_key = OPENAI_API_KEY

#record audio
def record_audio(file_path, record_seconds):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=CHUNK_SIZE)
    st.write("Recording...")
    frames = []
    for _ in range(0, int(44100 / CHUNK_SIZE * record_seconds)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
    st.write("Recording complete.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
    st.write(f"Audio saved to {file_path}")

#transcribe audio with whisper
def transcribe_audio_with_whisper(file_path):
    st.write(f"Transcribing audio from {file_path}")
    if not os.path.exists(file_path):
        st.write(f"File {file_path} does not exist.")
        return None
    with open(file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
    return transcription['text']

#generate text AraGPT2
second_gpt2_model, second_gpt2_tokenizer = load_gpt2_model("rarayayan/testftargpt2")

def generate_text(text, max_length=150, num_return_sequences=1, temperature=0.7, top_p=0.9, top_k=50):
    text_generator = pipeline("text-generation", model=second_gpt2_model, tokenizer=second_gpt2_tokenizer)
    generated_text = text_generator(
        text,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        truncation=True
    )
    return generated_text[0]['generated_text']

# convert text to speech 
def convert_text_to_speech(text, output_path):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    querystring = {"output_format":"mp3_22050_32"}
    payload = {
        "model_id": "eleven_multilingual_v2",
        "text": text,
        "voice_settings": {
            "stability": 0.15,
            "similarity_boost": 1,
            "style": 0.35
        }
    }
    headers = {
        "xi-api-key": XI_API_KEY,
        "Content-Type": "application/json"
    }
    # response = requests.post(url, json=payload, headers=headers, params=querystring)
    response = requests.request("POST", url, json=payload, headers=headers, params=querystring)
    if response.ok:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        st.write("Audio stream saved successfully.")
    else:
        st.write(response.text)



#tabs
tab1, tab2 = st.tabs(["FT-AraGPT2 Text-to-text", "FT-AraGPT2 Speech-to-speech"])#, "OpenAI API"])

with tab1:
    st.header("Fine-Tuned AraGPT2 Text-To-Text")
    if "gpt2_messages" not in st.session_state:
        st.session_state.gpt2_messages = []

    for message in st.session_state.gpt2_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("إسألني عن التراث السعودي (AraGPT2)"):
        st.session_state.gpt2_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_gpt2_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.gpt2_messages.append({"role": "assistant", "content": response})

with tab2:
    st.header("Fine-Tuned AraGPT2 Speech-To-Text")

    if st.button("Record Audio"):
        record_audio(INPUT_AUDIO_FILE, RECORD_SECONDS)

    input_audio = os.path.join(".", INPUT_AUDIO_FILE)
    output_audio = os.path.join(".", OUTPUT_AUDIO_FILE)

    if os.path.exists(input_audio):
        transcribed_text = transcribe_audio_with_whisper(input_audio)

        if transcribed_text:
            st.write("Transcribed Text:", transcribed_text)
            generated_text = generate_text(transcribed_text)
            st.write("Generated Text :", generated_text)
            convert_text_to_speech(generated_text, output_audio)
            if os.path.exists(output_audio):
                st.audio(output_audio)
                st.write("Playing the generated audio...")
                audio = AudioSegment.from_file(output_audio)
                play(audio)
            else:
                st.write("Generated audio file does not exist.")
        else:
            st.write("Transcription failed.")
    else:
        st.write("Please record audio first.")

# with tab3:
#     st.header("OpenAI API")
#     if "openai_messages" not in st.session_state:
#         st.session_state.openai_messages = []

#     for message in st.session_state.openai_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if prompt := st.chat_input("إسألني عن التراث السعودي (OpenAI)"):
#         st.session_state.openai_messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         response = generate_openai_response(prompt)
#         with st.chat_message("assistant"):
#             st.markdown(response)

#         st.session_state.openai_messages.append({"role": "assistant", "content": response})
