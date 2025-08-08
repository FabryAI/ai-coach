"""
stt.py

Speech-to-Text utilities for the Coach AI project.
This module provides:
- Microphone recording to a WAV file (16 kHz, mono)
- Offline transcription using Faster-Whisper (Whisper model, quantized and fast)
- Minimal device helpers for debugging audio issues on Windows

Why Faster-Whisper:
- Optimized inference, supports CPU/GPU
- Built-in VAD filter to skip silences
- Easy to switch model size: tiny/base/small/medium/large-v3
"""

from __future__ import annotations

import time
import uuid
import wave
from pathlib import Path
from typing import Optional, List

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class SpeechToText:
    """
    Encapsulates microphone recording and offline transcription.

    Typical usage:
        stt = SpeechToText(
            model_size="small",
            sample_rate=16000,
            audio_dir="data/audio_raw",
            device_index=None,      # use default input device
            compute_type="int8"     # good default for CPU; "float16"/"int8_float16" for GPU
        )
        wav = stt.record_wav(seconds=10)   # record from mic
        text = stt.transcribe(wav, language="it")  # offline transcription
    """

    def __init__(
        self,
        model_size: str = "small",
        sample_rate: int = 16000,
        audio_dir: str = "data/audio_raw",
        device_index: Optional[int] = None,
        compute_type: str = "int8",
    ):
        """
        Args:
            model_size: Whisper checkpoint size ("tiny", "base", "small", "medium", "large-v3").
                        Start with "small" for a good speed/quality balance on CPU.
            sample_rate: Recording sample rate (Whisper works well at 16 kHz).
            audio_dir: Directory where raw recordings are stored.
            device_index: Optional input device index from sounddevice; None = default.
            compute_type: Inference precision. "int8" is fast on CPU. If you have GPU,
                          try "float16" or "int8_float16" for better accuracy.
        """
        self.model_size = model_size
        self.sample_rate = int(sample_rate)
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.device_index = device_index

        # Initialize Faster-Whisper model. The first run will download weights.
        # This object is heavy, so we keep one instance per process.
        self.model = WhisperModel(
            self.model_size,
            device="cuda",        # force GPU
            compute_type=compute_type
        )


    # ---------------------------
    # Microphone utilities
    # ---------------------------

    @staticmethod
    def list_input_devices() -> List[str]:
        """
        Returns a list of human-readable input device lines.
        Useful for picking the correct microphone on Windows.
        """
        lines = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                lines.append(f"[{idx}] {dev['name']}  (in={dev['max_input_channels']})")
        return lines

    def record_wav(self, seconds: float = 10.0) -> str:
        """
        Records audio from the default (or selected) input device and saves it as 16-bit PCM WAV.

        Args:
            seconds: Duration to record.

        Returns:
            Path to the saved WAV file (string).
        """
        # Generate a unique filename (timestamp + short UUID)
        fname = self.audio_dir / f"rec_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"

        # Allocate recording buffer: mono, int16
        frames = int(self.sample_rate * seconds)
        print(f"[STT] Recording {seconds:.1f}s at {self.sample_rate} Hz...")
        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device_index,
        )
        sd.wait()  # block until recording finishes
        print("[STT] Done recording.")

        # Write PCM to WAV
        with wave.open(str(fname), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM -> 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())

        return str(fname)

    # ---------------------------
    # Transcription
    # ---------------------------

    def transcribe(
        self,
        wav_path: str,
        language: Optional[str] = None,
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> str:
        """
        Offline transcription using Faster-Whisper.

        Args:
            wav_path: Path to a WAV/MP3/FLAC file (WAV mono 16 kHz recommended).
            language: ISO code like "it" or "en". If None, the model will try to detect language.
            beam_size: Decoding beams (1 = greedy, faster; >1 can improve quality slightly).
            vad_filter: Whether to use Voice Activity Detection to skip silences.

        Returns:
            The transcribed text (string).
        """
        segments, info = self.model.transcribe(
            wav_path,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text


# ---------------------------
# Quick manual test (optional)
# Run: python src/stt.py
# ---------------------------
if __name__ == "__main__":
    """
    This block allows you to test recording + transcription quickly, without the rest of the app.
    Tip (Windows): If you get "no default input device" or silence,
    run `print('\n'.join(SpeechToText.list_input_devices()))` to pick a device index.
    """
    print("Available input devices:")
    print("\n".join(SpeechToText.list_input_devices()))
    stt = SpeechToText(model_size="small", sample_rate=16000, audio_dir="data/audio_raw")
    wav = stt.record_wav(seconds=6)
    text = stt.transcribe(wav, language="en")  # set None for auto-detect
    print("\n[TRANSCRIPT]")
    print(text or "(empty)")
