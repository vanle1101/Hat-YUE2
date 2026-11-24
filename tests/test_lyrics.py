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


def test_api_is_opt_in_and_providers_does_not_expose_key(tmp_path):
    local = TestClient(create_app(Settings(data_dir=tmp_path), lambda _: None))
    assert local.get("/v1/lyrics/providers").json() == {"api_host": None}
    assert local.post("/v1/lyrics", files={"audio": ("sample.wav", b"sample")},
                      data={"provider": "api"}).status_code == 503
    settings = Settings(data_dir=tmp_path, lyrics_api_key="test-only-key")
    remote = TestClient(create_app(settings, lambda _: None))
    assert remote.get("/v1/lyrics/providers").json() == {"api_host": "codex.vorte.me"}
    assert "test-only-key" not in repr(settings)
    response = remote.post("/v1/lyrics", files={"audio": ("sample.wav", b"sample")},
                           data={"provider": "api"}, headers={"Origin": "https://other.example"})
    assert response.status_code == 403


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_api_recognizer_uploads_normalized_audio_and_handles_errors(tmp_path):
    import httpx

    from yue2.lyrics import ApiLyricsRecognizer

    source = BytesIO()
    with wave.open(source, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * 3200)
    payload = source.getvalue()
    statuses = iter([200, 401, 429])

    def respond(request):
        assert str(request.url) == "https://codex.vorte.me/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-only-key"
        assert b"gpt-4o-transcribe" in request.content
        assert b'name="language"\r\n\r\nvi' in request.content
        assert b'filename="track.wav"' in request.content
        assert b"RIFF" in request.content
        return httpx.Response(next(statuses), json={"text": "Lời nhận được", "asset_pointer": "private"})

    recognizer = ApiLyricsRecognizer("https://codex.vorte.me/v1", "gpt-4o-transcribe", "test-only-key",
                                    work_dir=tmp_path, transport=httpx.MockTransport(respond))
    assert recognizer.transcribe(BytesIO(payload), "vi") == {"lyrics": "Lời nhận được", "language": "vi"}
    with pytest.raises(LyricsUnavailable, match="từ chối khóa"):
        recognizer.transcribe(BytesIO(payload), "vi")
    with pytest.raises(LyricsUnavailable, match="giới hạn"):
        recognizer.transcribe(BytesIO(payload), "vi")
    assert not list(tmp_path.iterdir())
