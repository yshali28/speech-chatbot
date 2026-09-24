import io
import json

import numpy as np
import streamlit as st
import av
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

from mic_recorder import record_audio

MODEL_REPO = "yshali28/speech-trained-distilbert"
CONFIDENCE_THRESHOLD = 0.5

# Real example utterances from the CLINC150 dataset (clinc/clinc_oos), one
# per intent this model was trained on, for the "try saying" sidebar.
SUGGESTION_PROMPTS = [
    "What is the weather like in Sparks right now?",
    "Can you get me a table for 2 at 7pm?",
    "How do I say you're welcome in Chinese?",
    "Find me round trip flights out of LAX to SFO.",
    "Set my alarm for 6am tomorrow.",
    "Tell me something funny about cats.",
    "What did I spend on groceries this month?",
    "How many calories are in a slice of pizza?",
]

# A sample of the intent categories the classifier recognizes.
EXAMPLE_CATEGORIES = [
    "weather", "restaurant_reservation", "translate", "book_flight",
    "alarm", "tell_joke", "spending_history", "calories",
    "todo_list", "directions", "recipe", "calendar",
]

st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="wide")


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


def decode_audio(audio_bytes):
    container = av.open(io.BytesIO(audio_bytes))
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    chunks = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray())
    container.close()

    audio_array = np.concatenate(chunks, axis=1).flatten().astype(np.float32) / 32768.0
    return audio_array, 16000


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

st.markdown(
    """
    <style>
    html, body {
        overflow: hidden !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0.5rem !important;
        max-height: 100vh;
        overflow: hidden;
    }
    .app-header {
        text-align: center;
        margin-bottom: 1.25rem;
    }
    .app-header h1 {
        font-size: 1.9rem;
        margin: 0;
    }
    .app-header p {
        color: rgba(128, 128, 128, 0.9);
        font-size: 0.95rem;
        margin: 0.25rem 0 0 0;
    }
    .side-heading {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: rgba(128, 128, 128, 0.9);
        margin-bottom: 0.75rem;
    }
    .suggestion-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 10px;
        padding: 9px 13px;
        margin-bottom: 8px;
        font-size: 13.5px;
        line-height: 1.4;
        color: inherit;
    }
    .category-pill {
        display: inline-block;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 999px;
        padding: 4px 12px;
        margin: 0 6px 8px 0;
        font-size: 13px;
        color: inherit;
    }
    </style>
    <div class="app-header">
        <h1>🎙️ Voice Assistant</h1>
        <p>Speak your question — I'll transcribe it, figure out what you mean, and reply.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, chat_col, right_col = st.columns([1, 2, 1], gap="medium")

with left_col:
    st.markdown('<div class="side-heading">Try asking</div>', unsafe_allow_html=True)
    for prompt in SUGGESTION_PROMPTS:
        st.markdown(f'<div class="suggestion-card">{prompt}</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="side-heading">What I can help with</div>', unsafe_allow_html=True)
    pills = "".join(f'<span class="category-pill">{c}</span>' for c in EXAMPLE_CATEGORIES)
    st.markdown(pills, unsafe_allow_html=True)

with chat_col:
    chat_box = st.container(height=380)
    with chat_box:
        if not st.session_state.messages:
            st.info("Tap the mic below and ask something to get started.")

        for message in st.session_state.messages:
            avatar = "🧑" if message["role"] == "user" else "🎙️"
            with st.chat_message(message["role"], avatar=avatar):
                st.write(message["content"])
                if message["role"] == "assistant":
                    st.caption(f"Detected intent: {message['intent']} ({message['confidence']:.2f} confidence)")

        st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

    if st.session_state.messages:
        st.components.v1.html(
            """
            <script>
            try {
                var el = window.parent.document.getElementById("chat-bottom");
                if (el) { el.scrollIntoView({ block: "end" }); }
            } catch (e) {}
            </script>
            """,
            height=0,
        )

    audio_bytes = record_audio()

    if audio_bytes is not None:
        audio_id = len(audio_bytes)

        if audio_id != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_id

            audio_array, sample_rate = decode_audio(audio_bytes)

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