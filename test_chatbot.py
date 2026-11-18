# test_chatbot.py - End-to-end checks against a running chatbot server
import os
import sys

import requests

BASE_URL = os.environ.get("CHATBOT_URL", "http://localhost:5001")

# (message, expected intent) - phrased differently from the training patterns
CASES = [
    ("Hello there!", "greeting"),
    ("Thanks for the help", "thanks"),
    ("Goodbye, see you", "goodbye"),
    ("Will it rain tomorrow?", "weather"),
    ("Tell me about machine learning", "ai_ml"),
    ("My code is not working", "error_handling"),
    ("I need some motivation", "motivation"),
    ("Any advice for my resume?", "career"),
    ("asdf qwerty zxcv", "default"),
]


class Tester:
    def __init__(self, base=BASE_URL):
        self.base = base
        self.failures = []

    def check(self, name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
        if not ok:
            self.failures.append(name)

    def run(self):
        print(f"Testing {self.base}\n")
        try:
            h = requests.get(f"{self.base}/health", timeout=5)
        except requests.exceptions.RequestException:
            print("Cannot reach the server. Start it first: python chatbot_app.py")
            return False
        self.check("health endpoint reports a loaded model", h.ok and h.json().get("model_loaded"), h.text)

        r = requests.get(f"{self.base}/intents")
        self.check("intents endpoint lists intents", r.ok and r.json()["total"] >= 10, r.text)
        r = requests.get(f"{self.base}/stats")
        self.check("stats endpoint works", r.ok and "intents_breakdown" in r.json(), r.text)
        r = requests.get(f"{self.base}/examples")
        self.check("examples endpoint works", r.ok and r.json().get("examples"), r.text)
        r = requests.get(f"{self.base}/")
        self.check("web interface is served", r.ok and "<html" in r.text.lower())

        print("\nIntent predictions:")
        for message, expected in CASES:
            r = requests.post(f"{self.base}/chat", json={"message": message})
            d = r.json() if r.ok else {}
            self.check(f"'{message}' -> {expected}", d.get("intent") == expected,
                       f"got {d.get('intent')} at {d.get('confidence', 0):.0%}")

        print("\nBatch and error handling:")
        r = requests.post(f"{self.base}/batch_chat", json={"messages": ["Hi", "Thanks", "Bye"]})
        self.check("batch_chat returns 3 responses", r.ok and r.json()["total"] == 3, r.text)
        r = requests.post(f"{self.base}/chat", json={"message": "   "})
        self.check("empty message returns 400", r.status_code == 400)
        r = requests.post(f"{self.base}/chat", data="not json", headers={"Content-Type": "text/plain"})
        self.check("invalid body returns 400", r.status_code == 400)
        r = requests.get(f"{self.base}/nope")
        self.check("unknown route returns 404 JSON", r.status_code == 404 and "error" in r.json())

        print()
        if self.failures:
            print(f"{len(self.failures)} check(s) failed: {', '.join(self.failures)}")
        else:
            print("All checks passed.")
        return not self.failures


if __name__ == "__main__":
    sys.exit(0 if Tester().run() else 1)
