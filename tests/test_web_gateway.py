import json
import httpx
from fastapi.testclient import TestClient

from yue2.web_gateway import create_app


def replace_upstream(app, handler):
    original = app.state.client
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler))
    return original


def test_gateway_serves_nomoney_ui_and_proxies_without_api_key():
    app = create_app("http://127.0.0.1:8015")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"items": []})

    with TestClient(app) as client:
        original = replace_upstream(app, handler)
        page = client.get("/")
        assert page.status_code == 200
        assert "NoMoney" in page.text and "YUE2_API_KEY" not in page.text
        assert "/v1/plans" not in page.text and "/artifacts/" not in page.text
        assert client.get("/nomoney-logo.svg").status_code == 200
        assert client.get("/app.js").status_code == 200
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


def test_api_bridge_routes_text_and_cover_flows(tmp_path, monkeypatch):
    monkeypatch.setenv("YUE2_WEB_SOURCE_DIR", str(tmp_path))
    app = create_app("http://127.0.0.1:8015")
    seen = []
    job_id = "a" * 32
    cover_job_id = "b" * 32

    def handler(request):
        seen.append((request.method, request.url.path, request.content))
        if request.url.path == "/v1/jobs" and request.method == "POST":
            payload = json.loads(request.content)
            assert set(payload.keys()) == {"n", "style", "lyrics", "cot", "seed"}
            assert payload["lyrics"] == "[Instrumental]\n"
            assert "instrumental" in payload["style"]
            return httpx.Response(202, json={"id": job_id, "status": "queued", "candidates": []})
        if request.url.path == f"/v1/jobs/{job_id}":
            return httpx.Response(200, json={"id": job_id, "status": "succeeded", "candidates": []})
        if request.url.path == "/v1/lyrics/providers":
            return httpx.Response(200, json={"api_host": None})
        if request.url.path == "/v1/lyrics":
            return httpx.Response(200, json={"lyrics": "[Verse]\nHello world", "structure": "suggested"})
        if request.url.path == "/v1/covers":
            assert b'name="reference"\r\n\r\nclose' in request.content
            return httpx.Response(202, json={
                "id": cover_job_id,
                "status": "running",
                "stage": "transcribing",
                "candidates": [{"id": "c1", "status": "running", "stage": "transcribing"}],
            })
        if request.url.path == f"/v1/jobs/{cover_job_id}":
            return httpx.Response(200, json={
                "id": cover_job_id,
                "status": "succeeded",
                "stage": "finished",
                "candidates": [
                    {"id": "c1", "status": "succeeded", "result": {"audio_url": "/v1/jobs/c1/audio", "audio_seconds": 15}},
                ],
            })
        raise AssertionError(f"Unexpected upstream request: {request.method} {request.url}")

    with TestClient(app) as client:
        original = replace_upstream(app, handler)

        assert client.get("/api/quota").json() == {"remaining": 999999}

        job_resp = client.post("/api/jobs", json={
            "n": 2,
            "style": "piano",
            "instrumental": True,
            "cot": "full",
            "seed": 7,
            "client_input": {"prompt": "piano"},
        })
        assert job_resp.status_code == 202
        assert job_resp.json()["id"] == job_id

        poll_resp = client.get(f"/api/jobs/{job_id}")
        assert poll_resp.status_code == 200
        assert poll_resp.json()["status"] == "succeeded"

        src_resp = client.post(
            "/api/cover/sources",
            content=b"RIFFdemoaudio",
            headers={"X-Audio-Extension": "wav", "Content-Type": "audio/wav"},
        )
        assert src_resp.status_code == 200
        source_id = src_resp.json()["source_id"]

        assert client.get(f"/api/cover/sources/{source_id}/audio").content == b"RIFFdemoaudio"

        lyrics_resp = client.post("/api/cover/recognize-lyrics", json={"source_id": source_id, "language": "vi"})
        assert lyrics_resp.status_code == 200
        assert lyrics_resp.json()["has_vocals"] is True
        assert "[Verse]" in lyrics_resp.json()["lyrics"]

        cover_resp = client.post("/api/cover/jobs", json={
            "source_id": source_id,
            "lyrics": "[Verse]\nHello world",
            "music_description": "acoustic",
            "style": "Pop",
            "melody_adherence": "high",
            "vocal_gender": "female",
        })
        assert cover_resp.status_code == 202
        assert cover_resp.json()["stage"] == "transcribing_score"

        cover_poll = client.get(f"/api/cover/jobs/{cover_job_id}")
        assert cover_poll.status_code == 200
        assert cover_poll.json()["status"] == "completed"
        assert cover_poll.json()["candidates"][0]["result"]["audio_url"] == "/v1/jobs/c1/audio"

        history = client.get("/api/cover/history").json()
        assert len(history["items"]) == 1
        assert history["items"][0]["id"] == cover_job_id

        app.state.client = original
