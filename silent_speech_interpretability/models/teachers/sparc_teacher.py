"""Optional SPARC articulatory teacher wrapper."""

class SPARCTeacher:
    def __init__(self, device: str = "auto"):
        self.device = device

    def available(self) -> bool:
        return False

    def extract(self, audio_path: str) -> dict:
        raise RuntimeError("SPARC is not installed; use synthetic teacher targets for tests.")
