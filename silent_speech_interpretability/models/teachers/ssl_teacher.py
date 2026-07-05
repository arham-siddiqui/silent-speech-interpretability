"""Optional HuBERT/Wav2Vec2 hidden-state teacher wrapper."""

class SSLTeacher:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import librosa  # noqa: F401
        except Exception:
            return False
        return True

    def extract_hidden_states(self, audio_path: str) -> dict:
        raise NotImplementedError("SSL extraction is scaffolded but not yet implemented.")
