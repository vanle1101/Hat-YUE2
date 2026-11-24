from io import BytesIO
import shutil
from types import SimpleNamespace
import wave

import pytest
from fastapi.testclient import TestClient

from yue2.lyrics import LyricsRecognizer, LyricsUnavailable
from yue2.service import Settings, create_app


def test_lyrics_route_returns_editable_text_and_offline_errors(tmp_path):
    class Recorder:
        def transcribe(self, source, language):
            assert language is None
            if not source.read():
                raise ValueError("Tệp âm thanh trống.")
            return {"lyrics": "Ánh trăng bên thềm", "language": "vi"}

    app = create_app(Settings(data_dir=tmp_path), lambda settings: None, lyrics_recognizer=Recorder())
    client = TestClient(app)
    uploaded = {"audio": ("original.mp3", b"x" * (300 * 1024), "audio/mpeg")}
    response = client.post("/v1/lyrics", files=uploaded, data={"language": "auto"})
    assert response.status_code == 200
    assert response.json() == {"lyrics": "Ánh trăng bên thềm", "language": "vi"}
    response = client.post("/v1/lyrics", files=uploaded, data={"language": "invalid"})
    assert response.status_code == 422
    response = client.post("/v1/lyrics", files={"audio": ("empty.wav", b"")}, data={"language": "auto"})
    assert response.status_code == 422

    class Offline:
        def transcribe(self, source, language):
            raise LyricsUnavailable("Không nạp được Whisper.")

    unavailable = create_app(Settings(data_dir=tmp_path / "other"), lambda settings: None,
                             lyrics_recognizer=Offline())
    response = TestClient(unavailable).post("/v1/lyrics", files=uploaded)
    assert response.status_code == 503
    assert response.json()["detail"] == "Không nạp được Whisper."


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_recognizer_decodes_audio_and_removes_temporary_files(tmp_path):
    np = pytest.importorskip("numpy")
    source = BytesIO()
    with wave.open(source, "wb") as audio:
        audio.setparams((2, 2, 22050, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * 22050 * 4)
    source.seek(0)

    class Model:
        def transcribe(self, samples, **options):
            assert options["language"] == "vi"
            assert samples.shape == (16000,)
            assert samples.dtype == np.float32
            return iter([SimpleNamespace(text=" Lời tiếng Việt "), SimpleNamespace(text=" ")]), SimpleNamespace(language="vi")

    recognizer = LyricsRecognizer("small", local_files_only=True, work_dir=tmp_path)
    recognizer._whisper = Model()
    assert recognizer.transcribe(source, "vi") == {"lyrics": "Lời tiếng Việt", "language": "vi"}
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="Không đọc được"):
        recognizer.transcribe(BytesIO(b"invalid"), "vi")
    assert not list(tmp_path.iterdir())
