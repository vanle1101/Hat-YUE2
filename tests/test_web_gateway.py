import httpx
from fastapi.testclient import TestClient

from yue2.web_gateway import create_app


def replace_upstream(app, handler):
    original = app.state.client
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler))
    return original


def test_gateway_serves_vietnamese_ui_and_proxies_without_api_key():
    app = create_app("http://127.0.0.1:8015")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"items": []})

    with TestClient(app) as client:
        original = replace_upstream(app, handler)
        page = client.get("/")
        assert page.status_code == 200
        assert "Bắt đầu tạo bài hát" in page.text and "YUE2_API_KEY" not in page.text
        assert "/v1/plans" not in page.text and "/artifacts/" not in page.text
        assert client.get("/service.env").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/v1/jobs").status_code == 200
        assert seen[-1].url.path == "/v1/jobs"
        assert client.post("/v1/jobs", content="x" * 262145).status_code == 413
        forwarded = client.post(
            "/v1/covers",
            files={"audio": ("song.mp3", b"y" * (300 * 1024), "audio/mpeg")},
            data={"style": "pop", "lyrics": "la"},
        )
        assert forwarded.status_code == 200
        assert seen[-1].url.path == "/v1/covers"
        assert len(seen[-1].content) > 256 * 1024
        assert client.post("/v1/covers", content=b"z" * (40 * 1024 * 1024 + 1)).status_code == 413
        assert page.headers["x-frame-options"] == "DENY"
        app.state.client = original


def test_gateway_reports_runtime_connects_and_streams_artifacts(tmp_path):
    artifact = tmp_path / "audio.flac"
    artifact.write_bytes(b"fLaC")
    app = create_app("http://127.0.0.1:8015")

    def handler(request):
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/internal/load":
            return httpx.Response(200, json={
                "backend": "torch",
                "device": "cpu",
                "ar_concurrency": 1,
                "nar_batch_size": 1,
                "ar_nar_overlap": False,
            })
        if request.url.path.endswith("/audio"):
            return httpx.Response(
                206,
                content=artifact.read_bytes(),
                headers={
                    "content-type": "audio/flac",
                    "content-range": "bytes 0-3/4",
                },
            )
        raise AssertionError(request.url)

    with TestClient(app) as client:
        original = replace_upstream(app, handler)
        status = client.get("/console/status").json()
        assert status["available"]
        assert status["runtime"] == {
            "backend": "torch",
            "device": "cpu",
            "ar_concurrency": 1,
            "nar_batch_size": 1,
            "ar_nar_overlap": False,
        }
        assert client.get("/console/connect").json() == {"connected": True}
        response = client.get(
            "/v1/jobs/" + "1" * 32 + "/audio",
            headers={"Range": "bytes=0-3"},
        )
        assert response.status_code == 206
        assert response.content == b"fLaC"
        assert response.headers["content-range"] == "bytes 0-3/4"
        app.state.client = original


def test_lyrics_upload_is_forwarded_with_audio_limit_and_long_read_timeout():
    app = create_app("http://127.0.0.1:8015")

    def handler(request):
        assert request.url.path == "/v1/lyrics"
        assert len(request.content) > 256 * 1024
        assert request.extensions["timeout"]["read"] > 30
        assert b"language" in request.content
        return httpx.Response(200, json={"lyrics": "Xin chào"})

    with TestClient(app) as client:
        original = replace_upstream(app, handler)
        response = client.post("/v1/lyrics", files={"audio": ("song.wav", b"x" * (300 * 1024))},
                               data={"language": "vi"})
        assert response.status_code == 200
        assert response.json() == {"lyrics": "Xin chào"}
        app.state.client = original
