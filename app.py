import io
import json

import streamlit as st
import soundfile as sf
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

from mic_recorder import record_audio

MODEL_REPO = "yshali28/speech-trained-distilbert"
CONFIDENCE_THRESHOLD = 0.5

st.set_page_config(page_title="Voice Chatbot", page_icon="🎙️", layout="centered")
st.title("Voice-Enabled Chatbot")
st.caption("Record a question below — it gets transcribed, classified, and answered.")


@st.cache_resource
def load_models():
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    classifier.eval()

    label_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_names.json")
    with open(label_path) as f:
        label_names = json.load(f)

    generator = pipeline("text2text-generation", model="google/flan-t5-base")

    return asr, tokenizer, classifier, label_names, generator


asr, tokenizer, classifier, label_names, generator = load_models()


def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        logits = classifier(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs))
    confidence = float(probs[idx])
    if confidence < CONFIDENCE_THRESHOLD:
        return "oos", confidence
    return label_names[idx], confidence


import random

FALLBACK_TEMPLATES = [
    "Sure, I can help you with {intent_label}. Let me look into that.",
    "Got it — checking on {intent_label} for you now.",
    "I can assist with {intent_label}. Here's what I found.",
    "On it. Handling your {intent_label} request now.",
]


def is_echo(output, text, threshold=0.6):
    output_words = set(output.lower().split())
    text_words = set(text.lower().split())
    if not text_words:
        return False
    overlap = len(output_words & text_words) / len(text_words)
    return overlap >= threshold


def generate_response(intent, text):
    if intent == "oos":
        return "Sorry, I did not understand that. Could you rephrase it?"

    intent_label = intent.replace("_", " ")
    prompt = f"Reply helpfully and concisely to this message about {intent_label}: {text}"
    output = generator(
        prompt,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3,
    )[0]["generated_text"].strip()

    if not output or is_echo(output, text):
        return random.choice(FALLBACK_TEMPLATES).format(intent_label=intent_label)
    return output


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if not st.session_state.messages:
    st.info("No messages yet — record a question below to start the conversation.")

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🎙️"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if message["role"] == "assistant":
            st.caption(f"Detected intent: {message['intent']} ({message['confidence']:.2f} confidence)")

st.markdown(
    """
    <style>
    .block-container {
        padding-bottom: 8rem;
    }
    div[data-testid="stCustomComponentV1"] {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        max-width: 736px;
        margin: 0 auto;
        padding: 1rem;
        background-color: var(--background-color);
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

audio_bytes = record_audio()

if audio_bytes is not None:
    audio_id = len(audio_bytes)

    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id

        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        with st.spinner("Transcribing..."):
            transcript = asr(
                {"array": audio_array, "sampling_rate": sample_rate},
                generate_kwargs={"language": "en", "task": "transcribe"},
            )["text"].strip()

        intent, confidence = predict_intent(transcript)

        with st.spinner("Generating response..."):
            reply = generate_response(intent, transcript)

        st.session_state.messages.append({"role": "user", "content": transcript})
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "intent": intent,
            "confidence": confidence,
        })

        st.rerun()