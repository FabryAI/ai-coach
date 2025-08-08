# src/tts.py
from __future__ import annotations
import os, subprocess, tempfile, platform

class TextToSpeech:
    def __init__(self, piper_dir: str, model_name: str = "en_US-amy-medium"):
        self.piper_dir = os.path.abspath(piper_dir)
        self.exe = os.path.join(self.piper_dir, "piper.exe")
        self.model = os.path.join(self.piper_dir, f"{model_name}.onnx")
        self.config = os.path.join(self.piper_dir, f"{model_name}.onnx.json")
        if not os.path.isfile(self.exe):
            raise FileNotFoundError(f"piper.exe non trovato in {self.piper_dir}")
        for p in [self.model, self.config]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Voce non trovata: {p}")

    def speak(self, text: str):
        if not text:
            return
        # scrivo testo su file per passarlo via STDIN (compatibile con tutte le build)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write(text.strip())
            txt_path = tf.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wav_path = wf.name

        cmd = [
            self.exe, "-m", self.model, "-c", self.config,
            "--sentence_silence", "0.4",
            "-f", wav_path
        ]
        completed = subprocess.run(
            cmd,
            cwd=self.piper_dir,                 # IMPORTANT: carica le DLL giuste
            stdin=open(txt_path, "r", encoding="utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False
        )
        os.unlink(txt_path)

        if completed.returncode != 0 or not os.path.isfile(wav_path):
            raise RuntimeError(
                f"Piper fallito (code={completed.returncode}).\nSTDERR:\n{completed.stderr}"
            )

        self._play_wav(wav_path)
        os.remove(wav_path)

    @staticmethod
    def _play_wav(wav_path: str):
        system = platform.system()
        if system == "Windows":
            subprocess.run(["powershell", "-c",
                            f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync();"])
        elif system == "Darwin":
            subprocess.run(["afplay", wav_path])
        else:
            for player in (["aplay", wav_path], ["paplay", wav_path], ["ffplay", "-autoexit", "-nodisp", wav_path]):
                try:
                    subprocess.run(player, check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except Exception:
                    continue
            print(f"WAV creato in: {wav_path} (nessun player disponibile)")
