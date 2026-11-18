# Intent-Based Chatbot (Deep Learning)

An intent-based chatbot with a **PyTorch BiLSTM + attention** classifier, a Flask REST API, and a browser interface that shows how the model classified each message.

## Features

- **Deep learning classifier**: embedding, bidirectional LSTM, masked attention pooling, linear head

- **Web interface** at `/`: live chat, top-3 intent probabilities, connection status, clickable intents and example messages, responsive layout, light/dark theme

- **REST API**: `/chat`, `/batch_chat`, `/health`, `/intents`, `/examples`, `/stats`

- **No NLTK downloads**: tokenization is pure Python, so setup has no SSL or data-download problems

- **Unknown-input handling**: gibberish and low-confidence messages fall back to the `default` intent

- **Edit replies without retraining**: responses are read from `intents.json` at server start

## Project structure

```
intent-chatbot/
├── intents.json        # Training patterns and responses
├── nlp_utils.py        # Tokenizer + Vocabulary (shared by training and serving)
├── model.py            # BiLSTM + attention network
├── train_model.py      # Training, held-out evaluation, model saving
├── chatbot_app.py      # Flask API + serves the web interface
├── static/
│   ├── index.html      # Web interface
│   └── favicon.svg
├── test_chatbot.py     # End-to-end API tests
├── demo.py             # Terminal chat client
├── setup_and_run.py    # Install, train, and start in one script
├── requirements.txt
└── models/dl_model.pt  # Created by train_model.py
```

## Quick start

```bash
python setup_and_run.py     # choose option 1
```

Or manually:

```bash
pip install -r requirements.txt
python train_model.py       # prints held-out accuracy, saves models/dl_model.pt
python chatbot_app.py       # then open http://localhost:5001
```

In a second terminal:

```bash
python test_chatbot.py      # automated checks
python demo.py              # chat in the terminal
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/api` | GET | API information |
| `/chat` | POST | `{"message": "..."}` returns reply, intent, confidence, top 3 intents |
| `/batch_chat` | POST | `{"messages": ["...", "..."]}` |
| `/health` | GET | 200 if the model is loaded, 503 otherwise |
| `/intents` | GET | Intents the model knows |
| `/examples` | GET | Example messages per intent |
| `/stats` | GET | Vocabulary size, held-out accuracy, patterns per intent |

```bash
curl -X POST http://localhost:5001/chat -H "Content-Type: application/json" -d '{"message": "Hello!"}'
```

## Customizing

**Add an intent**: add a block to `intents.json` with `tag`, `patterns` (10+ varied examples work best) and `responses`, then run `python train_model.py` and restart the server.

**Change replies only**: edit `responses` in `intents.json` and restart the server. No retraining needed.

**Tune the model**: `CONFIG`, `EPOCHS`, `LR` and `WORD_DROPOUT` at the top of `train_model.py`. The confidence threshold is `confidence_threshold` in `chatbot_app.py`.

**Change the port**: set `CHATBOT_PORT` (server) and `CHATBOT_URL` (demo/tests). The default is 5001 because macOS uses port 5000 for AirPlay Receiver.

## How accuracy is measured

`train_model.py` first trains on 80% of the patterns and reports accuracy on the held-out 20%, then retrains on everything for the saved model. The held-out number is the honest one; training accuracy on this small dataset will be near 100%. More varied patterns per intent is the best way to improve it.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model not found` / 503 from `/chat` | Run `python train_model.py` |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Port already in use | `CHATBOT_PORT=5002 python chatbot_app.py` |
| Web page says "Server offline" | Start `python chatbot_app.py` and refresh |
| Bot says "not sure I understand" too often | Add more patterns to that intent, or lower `confidence_threshold` |

## Next steps

- Replace the BiLSTM with a fine-tuned transformer (DistilBERT) for open-ended phrasing
- Add conversation context and session handling
- Log unmatched messages and use them as new training patterns
