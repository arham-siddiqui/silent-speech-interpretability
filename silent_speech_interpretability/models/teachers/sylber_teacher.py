"""Optional Sylber syllable teacher wrapper."""

class SylberTeacher:
    def __init__(self, device: str = "auto"):
        self.device = device

    def available(self) -> bool:
        return False

    def extract(self, audio_path: str) -> dict:
        raise RuntimeError("Sylber is not installed; use synthetic teacher targets for tests.")
