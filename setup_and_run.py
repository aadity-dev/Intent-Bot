#!/usr/bin/env python3
# setup_and_run.py - Install dependencies, train the model, and start the server
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
REQUIRED = ["intents.json", "model.py", "nlp_utils.py", "train_model.py",
            "chatbot_app.py", "requirements.txt", "static/index.html"]
PORT = os.environ.get("CHATBOT_PORT", "5001")


def run(args):
    try:
        subprocess.check_call([sys.executable, *args])
        return True
    except subprocess.CalledProcessError:
        return False


def install_dependencies():
    print("\nInstalling dependencies (PyTorch is a large download)...")
    ok = run(["-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed." if ok else "Install failed. Check your internet connection and pip.")
    return ok


def train_model():
    print("\nTraining the model...")
    ok = run(["train_model.py"])
    print("Training finished." if ok else "Training failed.")
    return ok


def start_server():
    if not os.path.exists("models/dl_model.pt"):
        print("No trained model found. Choose option 3 first.")
        return
    print(f"\nStarting server at http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        subprocess.run([sys.executable, "chatbot_app.py"])
    except KeyboardInterrupt:
        print("\nServer stopped.")


def main():
    print("=" * 56)
    print(" Intent chatbot setup (PyTorch BiLSTM)")
    print("=" * 56)
    if sys.version_info < (3, 9):
        print("Python 3.9+ is required.")
        return
    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing:
        print(f"Missing project files: {missing}")
        return

    print("\n1. Full setup (install + train + start server)")
    print("2. Install dependencies only")
    print("3. Train model only")
    print("4. Start server only")
    try:
        choice = input("\nChoose 1-4: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    if choice == "1":
        if install_dependencies() and train_model():
            start_server()
    elif choice == "2":
        install_dependencies()
    elif choice == "3":
        train_model()
    elif choice == "4":
        start_server()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
