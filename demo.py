# demo.py - Interactive terminal chat with the running chatbot server
import os
import requests

URL = os.environ.get("CHATBOT_URL", "http://localhost:5001")


def chat_with_bot():
    print(f"Chatbot demo ({URL}) - type 'quit' to exit")
    print("-" * 40)
    while True:
        try:
            text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not text:
            continue
        try:
            r = requests.post(f"{URL}/chat", json={"message": text}, timeout=10)
            data = r.json()
            if r.ok:
                print(f"Bot: {data['message']}")
                print(f"     [{data['intent']}, confidence {data['confidence']:.0%}]")
            else:
                print(f"Error: {data.get('error', r.status_code)}")
        except requests.exceptions.RequestException:
            print("Cannot reach the server. Start it with: python chatbot_app.py")


if __name__ == "__main__":
    chat_with_bot()
