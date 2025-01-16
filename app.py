# This app consists of three main steps.
# First we will use an image to text model to convert our loaded images into text.
# Step two will use the output of step one. A Large Language Model will take the input and create a compelling story around it (varies depending on prompt instructions).
# Lastly, a text to speech model will generate audio output from the written story.

from io import BytesIO
import json
import os
import requests
import time

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
import streamlit as st
import torch
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


# def generate_story(scenario: str, client: InferenceClient) -> str:
#     """Uses an LLM model to write a story around a given scenario"""
#     model_id = "Qwen/QwQ-32B-Preview"  # "meta-llama/Llama-3.2-1B"
#     messages = [
#         {
#             "role": "system", "content": "You are a story teller. Generate a short story based on a simple narrative, the story should be no more than 100 words."
#         },
#         {
#             "role": "user", "content": scenario
#         },
#     ]
#     completion = client.chat.completions.create(
#         model=model_id,
#         messages=messages,
#         max_tokens=100
#     )

#     story = completion.choices[0].message["content"]
#     return story


@st.cache_data()
def generate_story(scenario: str, api_token: str, temperature=0.7):
    """Generates a story using the Llama model with reduced repetitiveness.

    Args:
        scenario: The starting text for the story.
        api_token: Your Hugging Face API token.
        temperature: Controls the randomness of the generated text (default 0.7).

    Returns:
        The generated story as a string, or None if generation fails.
    """

    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
    headers = {"Authorization": f"Bearer {api_token}"}

    system_content = """You are a story teller. Generate a short story based on a simple narrative, the story should be 100 words long. Following is your input scenario: """
    payloads = {"inputs": system_content + scenario, "parameters": {"max_new_tokens": 120, "temperature": temperature, "return_full_text": False}}

    max_attempts = 5
    i = 0
    response = None
    while i < max_attempts:
        print("Calling Llama inference API")
        response = requests.post(API_URL, headers=headers, json=payloads)

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
        text = response.content.decode("utf-8")
        data = json.loads(text)
        return data[0]["generated_text"]
    else:
        print(f"Failed to process the request: {response.status_code}")

        print(vars(response))
        return None


def text_to_speech(text: str):
    """Receives a text and converts it into audio/speech using a pre-trained model from huggingfaces.

    Parameters
    ----------
    text : str
        text/story to convert into speech.

    Returns
    -------
    dict
        a speech.
    """
    model_id = "microsoft/speecht5_tts"
    pipe = pipeline(task="text-to-speech", model=model_id)
    
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(
        embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = pipe(text, forward_params={
                  "speaker_embeddings": speaker_embedding})
    return speech


def main():
    st.title("Story Generation")

    uploaded_image = True
    uploaded_image = st.file_uploader(
        "Load an image to create a compelling story.",
        type=['png', 'jpg']
    )

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

            story = generate_story(scenario, HF_TOKEN, temperature=0.8)
            st.write(story)

            speech = text_to_speech(story)  #, HF_TOKEN)
            if speech:
                st.audio(speech["audio"], sample_rate=speech["sampling_rate"])


if __name__ == "__main__":
    main()
    print("Done Executing")
