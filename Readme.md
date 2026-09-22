# Voice-Enabled Chatbot using Speech Recognition and Deep Learning

A web app that takes a spoken query, transcribes it, classifies the intent using a
fine-tuned deep learning model, and generates a response conditioned on the detected
intent. Built for the Speech and Language Processing course project.

**Live app:** 

## Pipeline

```
Microphone input
      |
      v
Whisper (ASR)  ->  recognized text
      |
      v
Fine-tuned DistilBERT  ->  predicted intent (CLINC150, 150 classes + out-of-scope)
      |
      v
FLAN-T5-small  ->  generated response, conditioned on (intent, text)
      |
      v
Streamlit UI displays: recognized text + response
```

## Why these components

- **Whisper (`base`)** — pretrained ASR, used directly for inference, no fine-tuning
  needed. Chosen over the browser's built-in speech API because it's an explainable
  deep learning model (log-Mel spectrogram -> transformer encoder-decoder) rather than
  an opaque browser call.
- **DistilBERT fine-tuned on CLINC150** — does the actual classification work this
  project is graded on. CLINC150 covers 150 intents across 10 domains plus an
  out-of-scope class, so the bot isn't limited to one narrow domain.
- **FLAN-T5-small** — generates the reply text instead of pulling from a fixed
  dictionary of canned responses per intent, so the same intent doesn't always produce
  identical wording.

## Repository structure

```
.
├── train_classifier.ipynb   # Colab/Kaggle: fine-tune DistilBERT on CLINC150, saves model
├── app.py                   # Streamlit app: ASR -> classifier -> generator -> UI
├── requirements.txt
└── README.md
```

No other folders. The fine-tuned classifier is pushed to the Hugging Face Hub from the
notebook and loaded by `app.py` at runtime — it is not stored in this repo.

## Dataset

[CLINC150](https://huggingface.co/datasets/clinc_oos) — 150 intents across 10 domains
(banking, travel, utility, small talk, etc.), ~150 labeled examples per intent, plus an
out-of-scope class for queries that don't match any intent.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires a browser with microphone access (HTTPS or localhost).

## Training

Open `train_classifier.ipynb` in Colab or Kaggle with a GPU runtime, run all cells.
The notebook fine-tunes DistilBERT on CLINC150 and pushes the resulting model to the
Hugging Face Hub, which `app.py` loads by name.

## Report

Dataset details, model architecture, training methodology, and results (accuracy,
macro F1, confusion matrix, ASR word error rate) are in `report.pdf`, submitted
separately.