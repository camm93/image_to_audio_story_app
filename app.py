# This app consists of three main steps.
# First we will use an image to text model to convert our loaded images into text.
# Step two will use the output of step one. A Large Language Model will take the input and create a compelling story around it (varies depending on prompt instructions).
# Lastly, a text to speech model will generate audio output from the written story.

# from IPython.display import Audio

from io import BytesIO
import os
import requests
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
# import getpass
# from langchain.chains import LLMChain
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate

from PIL import Image
import streamlit as st
# from streamlit import UploadedFile
from transformers import pipeline

load_dotenv()


def img2text(image: BytesIO) -> str:
    """Receives an image and converts it into text using a pre-trained model from huggingfaces.

    Parameters
    ----------
    image : BytesIO
        Image to convert into text.

    Returns
    -------
    str
        a text describing the image.
    """
    model_id = "Salesforce/blip-image-captioning-base"
    image_to_text = pipeline(task="image-to-text",
                             model=model_id)

    text = image_to_text(image)[0]["generated_text"]
    return text


def generate_story(scenario: str, client: InferenceClient) -> str:
    """Uses an LLM model to write a story around a given scenario"""
    model_id = "Qwen/QwQ-32B-Preview"  # "meta-llama/Llama-3.2-1B"
    messages = [
        {
            "role": "system", "content": "You are a story teller. Generate a short story based on a simple narrative, the story should be no more than 100 words."
        },
        {
            "role": "user", "content": scenario
        },
    ]
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=100
    )

    story = completion.choices[0].message["content"]
    return story


def text_to_speech(text: str, api_token: str):
    API_URL = "https://api-inference.huggingface.co/models/myshell-ai/MeloTTS-English"
    headers = {"Authorization": f"Bearer {api_token}"}
    payloads = {"inputs": text}

    max_attempts = 5
    i = 0
    response = None
    while i < max_attempts:
    
        response = requests.post(API_URL, headers=headers, json=payloads)
        # print("1", response)
        # print("2", response.text)
        if response.status_code in (500, 503):
            print(type(response))
            print(response)
            error_message = response.json().get("error", "")
            estimated_time = response.json().get("estimated_time", 5)
            print(f"Model is loading. Retrying in {estimated_time} seconds...")
            time.sleep(estimated_time)
        else:
            break
        
        i += 1

    if response and response.status_code == 200:
        # with open("assets/story.mp3", "wb") as file:
            # file.write(response.content)
        # print("Audio file saved successfully!")
        return response.content
    else:
        print(f"Failed to process the request: {response.status_code}")

        print(vars(response))
        return None


# def display_audio(audio_file: bytes):
#     with open("assets/story.mp3", "rb") as file:
#         audio_file = file.read()
#         st.audio(audio_bytes, format="audio/mp3") 


def main():
    st.title("Story Generation")

    uploaded_image = st.file_uploader("Load an image to create a compelling story.", type=['png', 'jpg'])

    if uploaded_image:
        st.image(uploaded_image)

        image = Image.open(uploaded_image)
        raw_scenario = img2text(image)
        st.write(raw_scenario)
        scenario = st.text_area(
            "## Image Description",
            value=raw_scenario,
            help="Modify to enhance the text."
        )

        if st.button("Generate Story"):
            HF_TOKEN = os.getenv('HUGGING_FACE_API_TOKEN')
            hf_client = InferenceClient(api_key=HF_TOKEN)
            story = "Once upon a time there was light in my life, but now it's only falling apart..."
            # story = generate_story(scenario, client=hf_client)
            st.write(story)
            audio = text_to_speech(story, HF_TOKEN)

            if audio:
                st.audio(audio, format="audio/mp3")


if __name__ == "__main__":
    main()
    print("Done Executing")
