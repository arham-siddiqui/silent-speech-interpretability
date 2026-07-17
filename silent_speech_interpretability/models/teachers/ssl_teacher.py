"""Optional HuBERT/Wav2Vec2 hidden-state teacher wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SSLTeacher:
    def __init__(self, model_name: str, device: str = "auto", sample_rate: int = 16_000, local_files_only: bool = False):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.local_files_only = local_files_only
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.processor = None
        self.model = None

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import librosa  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> "SSLTeacher":
        if not self.available():
            raise RuntimeError("SSLTeacher requires optional dependencies: transformers and librosa.")
        from transformers import AutoFeatureExtractor, AutoModel

        self.processor = AutoFeatureExtractor.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self.model = AutoModel.from_pretrained(self.model_name, local_files_only=self.local_files_only).to(self.device)
        self.model.eval()
        return self

    def _ensure_loaded(self) -> None:
        if self.processor is None or self.model is None:
            self.load()

    def load_audio(self, audio_path: str | Path) -> np.ndarray:
        import librosa

        waveform, _sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        return waveform.astype(np.float32)

    @torch.no_grad()
    def extract_hidden_states(self, audio_path: str | Path) -> dict[str, np.ndarray]:
        self._ensure_loaded()
        waveform = self.load_audio(audio_path)
        inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs)
        hidden = output.last_hidden_state.detach().cpu().numpy()[0].astype(np.float32)
        return {
            "hidden_states": hidden,
            "pooled": hidden.mean(axis=0).astype(np.float32),
            "sample_rate": np.asarray(self.sample_rate),
        }
