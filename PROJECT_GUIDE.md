# Project Guide

## Quick start (3 steps)

1. **Setup**: `python setup_and_run.py`, choose option 1 (installs packages, trains, starts the server)
2. **Server**: later runs only need `python chatbot_app.py`
3. **Chat**: open http://localhost:5001, or use `python demo.py`, or run `python test_chatbot.py`

## What the web interface shows

- Real-time chat with typing indicator
- The top three predicted intents with probability bars for your latest message
- Connection status (checked every 5 seconds)
- Every intent the model knows, as clickable chips that send an example message
- Responsive layout for mobile and desktop, light and dark themes

## Technical highlights

- **Model**: embedding (64) → BiLSTM (96 units) → masked attention → linear, trained with Adam, cosine learning-rate decay, label smoothing and word dropout
- **Data handling**: pure-Python tokenizer, so no NLTK downloads; synthetic unknown-word samples teach the `default` intent
- **Evaluation**: stratified 80/20 held-out split reported before the final model is trained on all data
- **Serving**: Flask, CPU inference, batch endpoint, JSON error responses

## What this project demonstrates

- NLP preprocessing and vocabulary building
- Sequence modelling in PyTorch (packing, masking, attention)
- Honest evaluation on small datasets
- REST API design and a no-framework frontend
- Automated end-to-end testing

## Troubleshooting

See the table in `README.md`.
