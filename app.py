import json

import streamlit as st
import soundfile as sf
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

MODEL_REPO = "yshali28/speech-trained-distilbert"
CONFIDENCE_THRESHOLD = 0.5

st.set_page_config(page_title="Voice Chatbot")
st.title("Voice-Enabled Chatbot")
st.write("Record a question. The app transcribes it, figures out the intent, and replies.")


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


def generate_response(intent, text):
    if intent == "oos":
        return "Sorry, I did not understand that. Could you rephrase it?"
    intent_label = intent.replace("_", " ")
    prompt = f"Reply helpfully and concisely to this message about {intent_label}: {text}"
    output = generator(
        prompt,
        max_new_tokens=40,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        early_stopping=True,
    )[0]["generated_text"]
    return output.strip()


audio = st.audio_input("Record your question")

if audio is not None:
    audio_array, sample_rate = sf.read(audio)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    with st.spinner("Transcribing..."):
        transcript = asr({"array": audio_array, "sampling_rate": sample_rate})["text"].strip()

    st.subheader("You said")
    st.write(transcript)

    intent, confidence = predict_intent(transcript)

    with st.spinner("Generating response..."):
        reply = generate_response(intent, transcript)

    st.subheader("Chatbot response")
    st.write(reply)

    st.caption(f"Detected intent: {intent} ({confidence:.2f} confidence)")