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
            assert options["condition_on_previous_text"] is False
            assert options["no_speech_threshold"] == 0.55
            assert options["compression_ratio_threshold"] == 2.2
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


@pytest.mark.parametrize("remote_outcome", ["valid", "rewritten", "timeout"])
def test_recognition_formats_sections_and_preserves_every_word(tmp_path, monkeypatch, remote_outcome):
    import httpx

    from yue2.lyrics import ApiLyricsRecognizer
    from yue2.lyric_sections import SYSTEM_PROMPT

    verse_one = "Chiếc đèn bên hiên ngóng đợi\nHàng cây im sau cơn mưa"
    hook = "Một ngày bình yên trên sân ga\nMình cùng đi qua bao mùa hoa"
    verse_two = "Chiếc tàu nghiêng qua con dốc\nSợi nắng vương bên ô cửa"
    bridge = "Rồi ta chợt nghe mùa hè trở lại\nQua những ô cửa nhỏ"
    blocks = [verse_one, hook, verse_two, hook, bridge]
    raw = "\n\n".join(blocks)
    proposed = "\n\n".join(
        f"[{name}]\n{block}"
        for name, block in zip(["Verse 1", "Chorus 1", "Verse 2", "Chorus 2", "Bridge 1"], blocks)
    )

    def transcribe(self, source, language):
        assert source.read() == b"audio"
        return {"lyrics": raw, "language": language}

    def remote(request):
        import json
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-6-luna"
        assert payload["messages"] == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw},
        ]
        if remote_outcome == "timeout":
            raise httpx.ReadTimeout("test timeout", request=request)
        candidate = proposed if remote_outcome == "valid" else proposed.replace("hè", "thu")
        return httpx.Response(200, json={"choices": [{"message": {"content": candidate}}]})

    monkeypatch.setattr(ApiLyricsRecognizer, "transcribe", transcribe)
    settings = Settings(data_dir=tmp_path, lyrics_api_key="test-only-key")
    app = create_app(settings, lambda _: None)
    original_client = httpx.Client

    class MockUpstream(original_client):
        def __init__(self, **kwargs):
            super().__init__(**{**kwargs, "transport": httpx.MockTransport(remote)})

    client = TestClient(app)
    monkeypatch.setattr(httpx, "Client", MockUpstream)
    response = client.post("/v1/lyrics", files={"audio": ("sample.wav", b"audio")},
                           data={"provider": "api", "language": "vi"})
    assert response.status_code == 200
    assert response.json() == {
        "lyrics": proposed, "language": "vi",
        "structure": "api" if remote_outcome == "valid" else "suggested",
    }


def test_local_formatter_finds_refrain_without_paragraph_breaks():
    from yue2.lyric_sections import add_lyric_sections

    refrain = "Up in the air we turn around and count the colours in the fading light."
    raw = f"We see the sunset. {refrain} The road curves east. {refrain} The bus has left."
    result = add_lyric_sections({"lyrics": raw, "language": "en"})
    assert result["structure"] == "suggested"
    assert "[Chorus 1]" in result["lyrics"] and "[Chorus 2]" in result["lyrics"]
    assert " ".join(line for line in result["lyrics"].splitlines() if not line.startswith("[")).split() == raw.split()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_gemini_recognizer_transcribes_with_fallback_and_handles_errors(tmp_path):
    import json
    import httpx
    from yue2.lyrics import GeminiLyricsRecognizer, optimize_style_with_gemini

    source = BytesIO()
    with wave.open(source, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * 3200)
    payload = source.getvalue()

    calls = []

    def respond(request):
        calls.append(request.url.path)
        assert request.headers["x-goog-api-key"] == "fake-gemini-key"
        body = json.loads(request.content)
        assert body["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/mp3"
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        if len(calls) == 2:
            return httpx.Response(200, json={
                "candidates": [{
                    "content": {
                        "parts": [
                            {"text": "thinking...", "thought": True},
                            {"text": "```\n[Verse 1]\nCuộc sống đâu lường\n```"},
                        ]
                    }
                }]
            })
        if len(calls) == 3:
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "NO_VOCALS"}]}}]
            })
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    rec = GeminiLyricsRecognizer(
        "fake-gemini-key", model="gemini-3-flash-preview",
        work_dir=tmp_path, transport=httpx.MockTransport(respond),
    )
    res = rec.transcribe(BytesIO(payload), "vi")
    assert res == {"lyrics": "[Verse 1]\nCuộc sống đâu lường", "language": "vi"}
    assert len(calls) == 2

    with pytest.raises(ValueError, match="Chưa nhận diện"):
        rec.transcribe(BytesIO(payload), "auto")

    with pytest.raises(LyricsUnavailable, match="từ chối khóa"):
        rec.transcribe(BytesIO(payload), "vi")
    assert not list(tmp_path.iterdir())

    settings = Settings(data_dir=tmp_path / "svc", gemini_api_key="fake-gemini-key")
    assert "fake-gemini-key" not in repr(settings)
    client = TestClient(create_app(settings, lambda _: None))
    assert client.get("/v1/lyrics/providers").json() == {"api_host": "generativelanguage.googleapis.com"}

    def style_handler(request):
        assert request.headers["x-goog-api-key"] == "fake-gemini-key"
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "Vietnamese, lounge jazz, solo piano, warm female lead vocal"}]}}]
        })

    polished = optimize_style_with_gemini(
        "nhạc jazz nhẹ chỉ piano", "Cuộc sống đâu lường", "fake-gemini-key",
        transport=httpx.MockTransport(style_handler),
    )
    assert polished == "Vietnamese, lounge jazz, solo piano, warm female lead vocal"

    def fail_handler(request):
        return httpx.Response(503, json={"error": {"message": "busy"}})

    assert optimize_style_with_gemini(
        "nhạc jazz", "Cuộc sống", "fake-gemini-key",
        transport=httpx.MockTransport(fail_handler),
    ) is None


def test_hybrid_gemini_proofreads_whisper_and_guards_against_hallucination(tmp_path):
    import httpx
    from yue2.lyrics import HybridGeminiLyricsRecognizer, proofread_lyrics_with_gemini

    raw_text = "Của sống đau lương trước điều gì, bàn tay rễu đang nắm thịt chặt"
    cleaned_text = "[Verse 1]\nCuộc sống đâu lường trước điều gì\nBàn tay dẫu đang nắm thật chặt"

    class FakeLocal:
        def transcribe(self, source, language):
            return {"lyrics": raw_text, "language": "vi"}

    def good_handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": cleaned_text}]}}]
        })

    hybrid = HybridGeminiLyricsRecognizer(
        FakeLocal(), "fake-gemini-key", work_dir=tmp_path,
        transport=httpx.MockTransport(good_handler),
    )
    assert hybrid.transcribe(BytesIO(b"wav"), "vi") == {"lyrics": cleaned_text, "language": "vi"}

    def hallucinated_handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": " ".join(["fake"] * 60)}]}}]
        })

    assert proofread_lyrics_with_gemini(
        raw_text, "vi", "fake-gemini-key",
        transport=httpx.MockTransport(hallucinated_handler),
    ) is None

    hybrid_failopen = HybridGeminiLyricsRecognizer(
        FakeLocal(), "fake-gemini-key", work_dir=tmp_path,
        transport=httpx.MockTransport(hallucinated_handler),
    )
    assert hybrid_failopen.transcribe(BytesIO(b"wav"), "vi") == {"lyrics": raw_text, "language": "vi"}


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_lyrics_recognizer_cuda_error_falls_back_to_cpu(tmp_path, monkeypatch):
    source = BytesIO()
    with wave.open(source, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * 3200)
    payload = source.getvalue()

    class BrokenCudaModel:
        def transcribe(self, samples, **options):
            def gen():
                raise RuntimeError("libcudnn_ops.so.9 not found")
                yield  # pragma: no cover
            return gen(), SimpleNamespace(language="vi")

    class GoodCpuModel:
        def transcribe(self, samples, **options):
            return iter([
                SimpleNamespace(text="Hãy subscribe cho kênh Ghiền Mì Gõ"),
                SimpleNamespace(text="Để không bỏ lỡ những video hấp dẫn"),
                SimpleNamespace(text="Cuộc sống đâu lường"),
                SimpleNamespace(text="Hẹn gặp lại các bạn trong những video tiếp theo."),
            ]), SimpleNamespace(language="vi")

    recognizer = LyricsRecognizer("small", local_files_only=True, work_dir=tmp_path, device="auto", unload_after=True)
    loads = []

    def fake_load(*, force_cpu=False):
        loads.append(force_cpu)
        if not force_cpu:
            recognizer._loaded_device = "cuda"
            recognizer._whisper = BrokenCudaModel()
        else:
            recognizer._loaded_device = "cpu"
            recognizer._whisper = GoodCpuModel()
        return recognizer._whisper

    monkeypatch.setattr(recognizer, "_load", fake_load)
    assert recognizer.transcribe(BytesIO(payload), "vi") == {"lyrics": "Cuộc sống đâu lường", "language": "vi"}
    assert loads == [False, True]
