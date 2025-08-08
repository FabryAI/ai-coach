"""
main.py

This is the entry point of the application for testing the CoachEngine class
with both text input and voice input (offline).

Features:
- Loads configuration from YAML
- Creates CoachEngine (life coaching via Llama 3.1 in Ollama)
- Optionally records audio and transcribes with Faster-Whisper (SpeechToText)
- Converts AI replies to speech using Piper (TextToSpeech)
- Simple console-based conversation loop
"""

import yaml
from tts import TextToSpeech
from coach import CoachEngine, DEFAULT_SYSTEM_PROMPT
from stt import SpeechToText


def load_config(path: str = "config/settings.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # ---------------------------
    # Load configuration
    # ---------------------------
    cfg = load_config()

    # Ensure coach section exists
    if "coach" not in cfg:
        cfg["coach"] = {}
    cfg["coach"].setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)

    # ---------------------------
    # Initialize core components
    # ---------------------------
    coach = CoachEngine(cfg)

    # Offline STT
    stt = SpeechToText(
        model_size="small",       # "base" for better accuracy, "tiny" for speed
        sample_rate=16000,
        audio_dir="data/audio_raw",
        compute_type="int8"       # safe for CPU and most GPUs
    )

    # Offline TTS
    tts = TextToSpeech(piper_dir="models/piper", model_name="en_US-amy-medium")

    

    # ---------------------------
    # Conversation loop
    # ---------------------------
    print("\n=== Coach AI (offline) ===")
    print("Press ENTER to record voice (6s) or type your message.")
    print("Type 'quit' to exit.\n")

    while True:
        # Prompt user
        user_input = input("You (press ENTER to speak): ").strip()

        # Exit condition
        if user_input.lower() in ("quit", "exit"):
            print("Session ended.")
            break

        # If ENTER pressed with no text → record voice
        if user_input == "":
            wav_path = stt.record_wav(seconds=6)
            user_input = stt.transcribe(wav_path, language="it")  # None for auto-detect
            print(f"[Transcript] {user_input}")

        # Skip empty input
        if not user_input:
            print("(Got empty input, try again)\n")
            continue

        # Get AI reply
        reply = coach.reply(user_input)
        print(f"Coach: {reply}\n")
        tts.speak(reply)
        # Speak reply
        tts.speak(reply)
