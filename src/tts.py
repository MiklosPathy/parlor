"""Platform-aware Kokoro TTS: mlx-audio on Apple Silicon, kokoro-onnx elsewhere."""

import os
import platform
import sys
from pathlib import Path

import numpy as np


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        raise NotImplementedError


class MLXBackend(TTSBackend):
    """mlx-audio backend (Apple Silicon GPU via MLX)."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        # Warmup: triggers pipeline init (phonemizer, spacy, etc.)
        list(self._model.generate(text="Hello", voice="af_heart", speed=1.0))

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        results = list(self._model.generate(text=text, voice=voice, speed=speed))
        return np.concatenate([np.array(r.audio) for r in results])


class ONNXBackend(TTSBackend):
    """kokoro-onnx backend (ONNX Runtime, CPU)."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        pcm, _sr = self._model.create(text, voice=voice, speed=speed)
        return pcm


class PiperBackend(TTSBackend):
    """Piper TTS backend (offline, multilingual)."""

    def __init__(self, voice_name: str):
        from piper.voice import PiperVoice
        onnx_path, config_path = self._resolve(voice_name)
        self._voice = PiperVoice.load(onnx_path, config_path=config_path)
        self.sample_rate = self._voice.config.sample_rate

    @staticmethod
    def _resolve(voice_name: str) -> tuple[str, str]:
        """Return (onnx_path, config_path), downloading from HuggingFace if needed."""
        if os.path.isfile(voice_name):
            return voice_name, voice_name + ".json"
        from huggingface_hub import hf_hub_download
        # voice_name format: "hu_HU-anna-medium"
        # HF path:           "hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx"
        parts = voice_name.split("-")
        locale, name, quality = parts[0], parts[1], parts[2]
        lang = locale.split("_")[0]
        hf_path = f"{lang}/{locale}/{name}/{quality}/{voice_name}"
        print(f"Downloading Piper voice {voice_name} from HuggingFace...")
        config_path = hf_hub_download(repo_id="rhasspy/piper-voices", filename=f"{hf_path}.onnx.json")
        onnx_path = hf_hub_download(repo_id="rhasspy/piper-voices", filename=f"{hf_path}.onnx")
        return onnx_path, config_path

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        sample = chunks[0]
        if isinstance(sample, (bytes, bytearray)):
            audio_bytes = b"".join(bytes(c) for c in chunks)
        elif hasattr(sample, 'audio_int16_bytes'):
            audio_bytes = b"".join(c.audio_int16_bytes for c in chunks)
        elif hasattr(sample, 'audio'):
            audio_bytes = b"".join(c.audio for c in chunks)
        else:
            public_attrs = [a for a in dir(sample) if not a.startswith('_')]
            raise AttributeError(f"AudioChunk has no known audio attribute. Available: {public_attrs}")
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def load() -> TTSBackend:
    """Load the best available TTS backend for this platform."""
    piper_model = os.environ.get("PIPER_MODEL", "")
    if piper_model:
        backend = PiperBackend(piper_model)
        print(f"TTS: Piper ({piper_model}, sample_rate={backend.sample_rate})")
        return backend

    if _is_apple_silicon() and not os.environ.get("KOKORO_ONNX"):
        try:
            backend = MLXBackend()
            print(f"TTS: mlx-audio (Apple GPU, sample_rate={backend.sample_rate})")
            return backend
        except ImportError:
            print("TTS: mlx-audio not installed, falling back to kokoro-onnx")

    backend = ONNXBackend()
    print(f"TTS: kokoro-onnx (CPU, sample_rate={backend.sample_rate})")
    return backend
