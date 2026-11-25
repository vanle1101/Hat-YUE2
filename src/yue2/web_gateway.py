"""Cổng web công khai cho dịch vụ tạo nhạc YuE2 (không yêu cầu khóa API)."""
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


FORWARDED_REQUEST_HEADERS = (
    "content-type", "idempotency-key", "range",
)
FORWARDED_RESPONSE_HEADERS = {
    "content-type", "content-disposition", "content-length", "content-range",
    "accept-ranges", "retry-after",
}
JSON_BODY_LIMIT = 256 * 1024
COVER_BODY_LIMIT = 40 * 1024 * 1024
UNLIMITED_QUOTA = 999999
MAX_CACHED_SOURCES = 32
SOURCE_RETENTION_SECONDS = 6 * 3600

STUDIO_DIR = Path(__file__).with_name("studio")
STUDIO_ASSETS = {
    "styles.css": "text/css; charset=utf-8",
    "analytics.js": "application/javascript; charset=utf-8",
    "quota.js": "application/javascript; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
    "cover.js": "application/javascript; charset=utf-8",
    "showcase-data.js": "application/javascript; charset=utf-8",
    "showcase.js": "application/javascript; charset=utf-8",
    "favicon.svg": "image/svg+xml",
    "nomoney-logo.svg": "image/svg+xml",
    "brand-logo.png": "image/png",
}
AUDIO_EXT_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
}
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STAGE_TO_COVER = {
    "queued": "score_pending",
    "claimed_waiting": "score_pending",
    "transcribing": "transcribing_score",
    "planning": "submitting_generation",
    "semantic": "generating",
    "synthesis": "generating",
    "decode": "generating",
    "saving": "generating",
    "finished": "done",
}


def _default_runtime():
    return {
        "backend": os.environ.get("YUE2_BACKEND", "vllm"),
        "device": os.environ.get("YUE2_DEVICE", "cuda"),
        "ar_concurrency": int(os.environ.get("YUE2_AR_CONCURRENCY", "4")),
        "nar_batch_size": int(os.environ.get("YUE2_NAR_BATCH_SIZE", "2")),
        "ar_nar_overlap": os.environ.get("YUE2_AR_NAR_OVERLAP", "true").lower() == "true",
    }


def _source_dir():
    root = Path(os.environ.get(
        "YUE2_WEB_SOURCE_DIR",
        str(Path(tempfile.gettempdir()) / "yue2_web_sources"),
    ))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _mp3_dir():
    root = _source_dir() / "mp3_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prune_mp3_cache(cache_dir: Path, limit: int = 48):
    files = sorted(cache_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    while len(files) > limit:
        oldest = files.pop(0)
        with suppress(OSError):
            oldest.unlink()


def _transcode_flac_to_mp3(flac_bytes: bytes, mp3_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    tmp_flac = mp3_path.with_suffix(".flac.tmp")
    tmp_mp3 = mp3_path.with_suffix(".mp3.tmp")
    try:
        tmp_flac.write_bytes(flac_bytes)
        proc = subprocess.run(
            [ffmpeg, "-y", "-nostdin", "-v", "error", "-i", str(tmp_flac),
             "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(tmp_mp3)],
            capture_output=True, timeout=180,
        )
        if proc.returncode != 0 or not tmp_mp3.is_file() or tmp_mp3.stat().st_size == 0:
            return False
        tmp_mp3.replace(mp3_path)
        _prune_mp3_cache(mp3_path.parent)
        return True
    finally:
        with suppress(OSError):
            tmp_flac.unlink()
        with suppress(OSError):
            tmp_mp3.unlink()


def _cover_history_path():
    return _source_dir() / "cover_jobs.json"


def _load_cover_history() -> dict:
    path = _cover_history_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cover_history(cover_jobs: dict):
    recent = dict(sorted(
        cover_jobs.items(),
        key=lambda kv: kv[1].get("created_at", 0),
        reverse=True,
    )[:20])
    tmp = _cover_history_path().with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(recent), encoding="utf-8")
        tmp.replace(_cover_history_path())
    except OSError:
        with suppress(OSError):
            tmp.unlink()


def _prune_sources(sources: dict):
    now = time.time()
    expired = [
        sid for sid, meta in sources.items()
        if (now - meta["created_at"]) > SOURCE_RETENTION_SECONDS
    ]
    for sid in expired:
        meta = sources.pop(sid, None)
        if meta:
            with suppress(OSError):
                meta["path"].unlink()
    while len(sources) > MAX_CACHED_SOURCES:
        oldest_sid = min(sources, key=lambda k: sources[k]["created_at"])
        meta = sources.pop(oldest_sid, None)
        if meta:
            with suppress(OSError):
                meta["path"].unlink()


async def _read_bounded_body(request: Request, limit: int, limit_message: str) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(status_code=413, detail=limit_message)
    return bytes(body)


async def _read_json_body(request: Request) -> dict:
    raw = await _read_bounded_body(request, JSON_BODY_LIMIT, "Dữ liệu yêu cầu vượt quá 256 KiB")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="JSON không hợp lệ") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Dữ liệu phải là JSON object")
    return data


def _upstream_error_response(exc: httpx.HTTPError) -> JSONResponse:
    return JSONResponse(
        {"detail": "Dịch vụ suy luận tạm thời không khả dụng, vui lòng thử lại sau"},
        status_code=503,
        headers={"Retry-After": "10"},
    )


def _json_or_detail(response: httpx.Response) -> dict:
    try:
        data = response.json()
        if isinstance(data, dict):
            return data
        return {"detail": str(data)}
    except ValueError:
        return {"detail": response.text or response.reason_phrase or f"HTTP {response.status_code}"}


AUDIO_URL_RE = re.compile(r"^/v1/jobs/([A-Za-z0-9_-]{1,64})/audio$")


def _rewrite_audio_urls(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload

    def _rewrite_result(res):
        if isinstance(res, dict):
            url = res.get("audio_url")
            if isinstance(url, str):
                m = AUDIO_URL_RE.match(url)
                if m:
                    res["audio_url"] = f"/api/audio/{m.group(1)}.mp3"

    _rewrite_result(payload.get("result"))
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for cand in candidates:
            if isinstance(cand, dict):
                _rewrite_result(cand.get("result"))
    return payload


def _forward_json_with_quota(response: httpx.Response) -> JSONResponse:
    data = _json_or_detail(response)
    if response.is_success:
        data["remaining"] = UNLIMITED_QUOTA
        _rewrite_audio_urls(data)
    return JSONResponse(data, status_code=response.status_code)


def _extract_cover_stage(job: dict) -> str:
    candidates = job.get("candidates")
    if isinstance(candidates, list) and candidates:
        for cand in candidates:
            if cand.get("status") == "running":
                return STAGE_TO_COVER.get(cand.get("stage", ""), "generating")
        return STAGE_TO_COVER.get(candidates[0].get("stage", ""), "score_pending")
    return STAGE_TO_COVER.get(job.get("stage", ""), "score_pending")


def _format_cover_job(job: dict, meta: dict | None = None) -> dict:
    status = job.get("status", "queued")
    raw_candidates = job.get("candidates") if isinstance(job.get("candidates"), list) else [job]
    ready_candidates = [
        cand for cand in raw_candidates
        if (cand.get("result") or {}).get("audio_url")
    ]
    if status in {"succeeded", "truncated"} or (status == "partial_failed" and ready_candidates):
        cover_status = "completed"
        stage = "done"
        error = None
    elif status in {"failed", "cancelled"} or (status == "partial_failed" and not ready_candidates):
        cover_status = "failed"
        stage = "done"
        err = job.get("error") or next(
            (cand.get("error") for cand in raw_candidates if cand.get("error")),
            None,
        ) or {}
        error = {
            "code": err.get("code", "cover_failed"),
            "message": err.get("message") or "Tạo bản cover thất bại, vui lòng thử lại.",
        }
    else:
        cover_status = status
        stage = _extract_cover_stage(job)
        error = None

    payload = {
        "id": job["id"],
        "created_at": job.get("created_at") or (meta or {}).get("created_at") or time.time(),
        "status": cover_status,
        "stage": stage,
        "remaining": UNLIMITED_QUOTA,
        "candidates": ready_candidates if cover_status == "completed" else raw_candidates,
    }
    if error:
        payload["error"] = error
    if meta:
        payload["request"] = meta.get("request", {})
        if meta.get("source"):
            payload["source"] = meta["source"]
    return _rewrite_audio_urls(payload)


def create_app(upstream=None):
    upstream = (
        upstream or os.environ.get("YUE2_UPSTREAM", "http://127.0.0.1:8015")
    ).rstrip("/")

    @asynccontextmanager
    async def lifespan(app):
        timeout = httpx.Timeout(30, connect=3)
        async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False) as client:
            app.state.client = client
            yield

    app = FastAPI(
        title="YuE2 Studio gateway", docs_url=None, redoc_url=None,
        openapi_url=None, lifespan=lifespan,
    )
    app.state.cover_sources = {}
    app.state.cover_jobs = _load_cover_history()
    app.state.audio_locks = {}

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; media-src 'self' blob: https://storage.googleapis.com; "
            "img-src 'self' data: https://storage.googleapis.com; "
            "connect-src 'self' https://storage.googleapis.com; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response

    @app.get("/")
    async def page():
        studio_index = STUDIO_DIR / "index.html"
        if studio_index.is_file():
            return FileResponse(studio_index, media_type="text/html")
        return FileResponse(
            Path(__file__).with_name("studio.html"), media_type="text/html")

    @app.get("/console/status")
    async def console_status():
        maintenance = Path(os.environ.get(
            "YUE2_MAINTENANCE_FILE",
            "/opt/noiz-yue/web-maintenance.txt",
        ))
        if maintenance.exists():
            return {
                "available": False,
                "message": maintenance.read_text()[:500],
            }
        try:
            response = await app.state.client.get(upstream + "/health/ready")
            available = response.status_code == 200
            runtime = _default_runtime()
            if available:
                try:
                    load_resp = await app.state.client.get(upstream + "/internal/load")
                    if load_resp.status_code == 200:
                        info = load_resp.json()
                        runtime = {
                            "backend": info.get("backend", runtime["backend"]),
                            "device": info.get("device", runtime["device"]),
                            "ar_concurrency": info.get("ar_concurrency", runtime["ar_concurrency"]),
                            "nar_batch_size": info.get("nar_batch_size", runtime["nar_batch_size"]),
                            "ar_nar_overlap": info.get("ar_nar_overlap", runtime["ar_nar_overlap"]),
                        }
                except Exception:
                    pass
            return {
                "available": available,
                "message": (
                    "Mô hình đã sẵn sàng tạo nhạc"
                    if available else "Mô hình đang được nạp, vui lòng thử lại sau"
                ),
                "runtime": runtime,
            }
        except httpx.HTTPError:
            return {
                "available": False,
                "message": "Mô hình tạm thời ngoại tuyến, giao diện vẫn mở và sẽ gửi được tác vụ khi sẵn sàng",
            }

    @app.get("/console/connect")
    async def console_connect():
        try:
            response = await app.state.client.get(upstream + "/health/ready")
        except httpx.HTTPError:
            return JSONResponse(
                {"detail": "Dịch vụ suy luận tạm thời không khả dụng"}, 503,
                headers={"Retry-After": "10"},
            )
        if response.status_code == 200:
            return {"connected": True}
        return JSONResponse(
            {"detail": "Dịch vụ suy luận chưa sẵn sàng"},
            503,
            headers={"Retry-After": "10"},
        )

    @app.get("/api/quota")
    async def api_quota():
        return {"remaining": UNLIMITED_QUOTA}

    @app.post("/api/analytics/events")
    async def api_analytics_events():
        return {"ok": True}

    @app.post("/api/jobs")
    async def api_create_job(request: Request):
        body = await _read_json_body(request)
        instrumental = bool(body.get("instrumental"))
        style = str(body.get("style") or "").strip()
        lyrics = str(body.get("lyrics") or "").strip()
        if instrumental and not lyrics:
            lyrics = "[Instrumental]\n"
            if "instrumental" not in style.lower():
                style = f"{style}, instrumental" if style else "instrumental"
        max_variants = 1 if os.environ.get("YUE2_DEFAULT_VARIANTS", "2").strip() == "1" else 2
        n = 1 if (body.get("n") == 1 or max_variants == 1) else 2
        cot = body.get("cot") if body.get("cot") in {"full", "melody", "off"} else "full"
        raw_seed = body.get("seed")
        seed = raw_seed if isinstance(raw_seed, int) and not isinstance(raw_seed, bool) and raw_seed >= 0 else random.randint(1, 2_147_483_647)

        payload = {"n": n, "style": style, "lyrics": lyrics, "cot": cot, "seed": seed}
        if isinstance(body.get("abc"), str) and body["abc"].strip():
            payload["abc"] = body["abc"]
        if isinstance(body.get("cfg_scale"), (int, float)) and not isinstance(body.get("cfg_scale"), bool):
            payload["cfg_scale"] = body["cfg_scale"]

        headers = {}
        if "idempotency-key" in request.headers:
            headers["idempotency-key"] = request.headers["idempotency-key"]

        try:
            response = await app.state.client.post(
                upstream + "/v1/jobs", json=payload, headers=headers,
            )
        except httpx.HTTPError as exc:
            return _upstream_error_response(exc)

        return _forward_json_with_quota(response)

    @app.get("/api/jobs/{job_id}")
    async def api_get_job(job_id: str):
        if not SAFE_ID_RE.match(job_id):
            raise HTTPException(status_code=400, detail="Mã tác vụ không hợp lệ")
        try:
            response = await app.state.client.get(upstream + f"/v1/jobs/{job_id}")
        except httpx.HTTPError as exc:
            return _upstream_error_response(exc)
        return _forward_json_with_quota(response)

    @app.post("/api/cover/sources")
    async def api_upload_cover_source(request: Request):
        ext = (request.headers.get("x-audio-extension") or "wav").strip().lower()
        if ext not in AUDIO_EXT_MIME:
            raise HTTPException(status_code=400, detail="Định dạng âm thanh không được hỗ trợ")
        raw = await _read_bounded_body(request, COVER_BODY_LIMIT, "Dữ liệu yêu cầu vượt quá 40 MiB")
        if not raw:
            raise HTTPException(status_code=400, detail="Tệp âm thanh trống")
        _prune_sources(app.state.cover_sources)
        source_id = uuid.uuid4().hex
        filename = f"{source_id}.{ext}"
        path = _source_dir() / filename
        path.write_bytes(raw)
        app.state.cover_sources[source_id] = {
            "path": path,
            "filename": filename,
            "mime_type": request.headers.get("content-type") or AUDIO_EXT_MIME[ext],
            "size_bytes": len(raw),
            "created_at": time.time(),
        }
        return {"source_id": source_id}

    @app.get("/api/cover/sources/{source_id}/audio")
    async def api_get_cover_source_audio(source_id: str):
        if not SAFE_ID_RE.match(source_id):
            raise HTTPException(status_code=400, detail="Mã tệp không hợp lệ")
        meta = app.state.cover_sources.get(source_id)
        if not meta or not meta["path"].is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp âm thanh gốc")
        return FileResponse(meta["path"], media_type=meta["mime_type"], filename=meta["filename"])

    @app.post("/api/cover/recognize-lyrics")
    async def api_recognize_lyrics(request: Request):
        body = await _read_json_body(request)
        source_id = str(body.get("source_id") or "")
        meta = app.state.cover_sources.get(source_id)
        if not meta or not meta["path"].is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp âm thanh gốc")
        language = body.get("language") if body.get("language") in {"vi", "en", "auto"} else "auto"

        provider = "local"
        try:
            prov_resp = await app.state.client.get(upstream + "/v1/lyrics/providers")
            if prov_resp.status_code == 200 and prov_resp.json().get("api_host"):
                provider = "api"
        except Exception:
            pass

        audio_bytes = meta["path"].read_bytes()
        try:
            response = await app.state.client.post(
                upstream + "/v1/lyrics",
                files={"audio": (meta["filename"], audio_bytes, meta["mime_type"])},
                data={"language": language, "provider": provider},
                timeout=httpx.Timeout(900.0, connect=3.0),
            )
        except httpx.HTTPError as exc:
            return _upstream_error_response(exc)

        data = _json_or_detail(response)
        if not response.is_success:
            return JSONResponse(data, status_code=response.status_code)
        lyrics_text = str(data.get("lyrics") or "").strip()
        return {
            "has_vocals": bool(lyrics_text),
            "lyrics": lyrics_text,
            "structure": data.get("structure"),
            "remaining": UNLIMITED_QUOTA,
        }

    @app.post("/api/cover/jobs")
    async def api_create_cover_job(request: Request):
        body = await _read_json_body(request)
        source_id = str(body.get("source_id") or "")
        meta = app.state.cover_sources.get(source_id)
        if not meta or not meta["path"].is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp âm thanh gốc")

        lyrics = str(body.get("lyrics") or "").strip()
        description = str(body.get("music_description") or "").strip()
        style = str(body.get("style") or "").strip()
        vocal_gender = str(body.get("vocal_gender") or "auto")
        voice_tag = {"female": "female lead vocal", "male": "male lead vocal"}.get(vocal_gender, "")
        combined_style = ", ".join(part for part in (description, style, voice_tag) if part)
        reference = "close" if body.get("melody_adherence") == "high" else "free"
        seed = random.randint(1, 2_147_483_647)

        headers = {}
        if "idempotency-key" in request.headers:
            headers["idempotency-key"] = request.headers["idempotency-key"]

        audio_bytes = meta["path"].read_bytes()
        try:
            response = await app.state.client.post(
                upstream + "/v1/covers",
                files={"audio": (meta["filename"], audio_bytes, meta["mime_type"])},
                data={
                    "style": combined_style or "pop",
                    "lyrics": lyrics,
                    "reference": reference,
                    "n": "1" if os.environ.get("YUE2_DEFAULT_VARIANTS", "2").strip() == "1" else "2",
                    "seed": str(seed),
                },
                headers=headers,
            )
        except httpx.HTTPError as exc:
            return _upstream_error_response(exc)

        data = _json_or_detail(response)
        if not response.is_success:
            return JSONResponse(data, status_code=response.status_code)

        job_meta = {
            "created_at": data.get("created_at") or time.time(),
            "request": {
                "source_id": source_id,
                "lyrics": lyrics,
                "music_description": description,
                "style": style,
                "melody_adherence": body.get("melody_adherence") or "high",
                "vocal_gender": vocal_gender,
            },
            "source": {
                "source_id": source_id,
                "audio_url": f"/api/cover/sources/{source_id}/audio",
                "filename": meta["filename"],
                "mime_type": meta["mime_type"],
                "size_bytes": meta["size_bytes"],
            },
        }
        app.state.cover_jobs[data["id"]] = job_meta
        _save_cover_history(app.state.cover_jobs)
        return JSONResponse(_format_cover_job(data, job_meta), status_code=response.status_code)

    @app.get("/api/cover/jobs/{job_id}")
    async def api_get_cover_job(job_id: str):
        if not SAFE_ID_RE.match(job_id):
            raise HTTPException(status_code=400, detail="Mã tác vụ không hợp lệ")
        try:
            response = await app.state.client.get(upstream + f"/v1/jobs/{job_id}")
        except httpx.HTTPError as exc:
            return _upstream_error_response(exc)
        data = _json_or_detail(response)
        if not response.is_success:
            return JSONResponse(data, status_code=response.status_code)
        return _format_cover_job(data, app.state.cover_jobs.get(job_id))

    @app.get("/api/audio/{job_id}.mp3")
    async def api_get_audio_mp3(job_id: str):
        if not SAFE_ID_RE.match(job_id):
            raise HTTPException(status_code=400, detail="Mã tác vụ không hợp lệ")
        mp3_path = _mp3_dir() / f"{job_id}.mp3"
        if mp3_path.is_file():
            return FileResponse(mp3_path, media_type="audio/mpeg", filename=f"{job_id}.mp3")
        lock = app.state.audio_locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            if mp3_path.is_file():
                return FileResponse(mp3_path, media_type="audio/mpeg", filename=f"{job_id}.mp3")
            try:
                response = await app.state.client.get(
                    upstream + f"/v1/jobs/{job_id}/audio",
                    timeout=httpx.Timeout(120.0, connect=3.0),
                )
            except httpx.HTTPError as exc:
                return _upstream_error_response(exc)
            if not response.is_success:
                return JSONResponse(_json_or_detail(response), status_code=response.status_code)
            flac_bytes = response.content
            if await asyncio.to_thread(_transcode_flac_to_mp3, flac_bytes, mp3_path):
                return FileResponse(mp3_path, media_type="audio/mpeg", filename=f"{job_id}.mp3")
            return Response(
                content=flac_bytes,
                media_type=response.headers.get("content-type", "audio/flac"),
                headers={"Content-Disposition": f'attachment; filename="{job_id}.flac"'},
            )

    @app.get("/api/cover/history")
    async def api_cover_history():
        items = []
        recent = sorted(
            app.state.cover_jobs.items(),
            key=lambda kv: kv[1]["created_at"],
            reverse=True,
        )[:20]
        for job_id, meta in recent:
            try:
                response = await app.state.client.get(upstream + f"/v1/jobs/{job_id}")
                if response.is_success:
                    items.append(_format_cover_job(response.json(), meta))
            except httpx.HTTPError:
                break
        return {"items": items}

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        if request.method == "GET" and path in STUDIO_ASSETS:
            asset = STUDIO_DIR / path
            if asset.is_file():
                return FileResponse(asset, media_type=STUDIO_ASSETS[path])
        if not (path.startswith("v1/") or path in {
                "health/live", "health/ready"}):
            return JSONResponse({"detail": "Không tìm thấy đường dẫn"}, 404)
        if any(part in {".", ".."} for part in path.split("/")):
            return JSONResponse({"detail": "Đường dẫn không hợp lệ"}, 400)
        if path == "v1/lyrics" and request.method == "POST":
            origin = request.headers.get("origin")
            if origin is not None and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Yêu cầu nhận diện phải đến từ chính Studio."}, 403)
        cover = request.method == "POST" and path in {"v1/covers", "v1/lyrics"}
        limit = COVER_BODY_LIMIT if cover else JSON_BODY_LIMIT
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                detail = "Dữ liệu yêu cầu vượt quá 40 MiB" if cover else "Dữ liệu yêu cầu vượt quá 256 KiB"
                return JSONResponse({"detail": detail}, 413)
        outgoing_headers = {
            key: request.headers[key] for key in FORWARDED_REQUEST_HEADERS
            if key in request.headers
        }
        outgoing_headers["accept-encoding"] = "identity"
        try:
            outgoing = app.state.client.build_request(
                request.method,
                upstream + "/" + path,
                params=request.query_params,
                headers=outgoing_headers,
                content=bytes(body),
            )
            if path == "v1/lyrics" and request.method == "POST":
                outgoing.extensions["timeout"]["read"] = 900.0
            response = await app.state.client.send(outgoing, stream=True)
        except httpx.HTTPError as exc:
            return _upstream_error_response(exc)
        selected = {
            key: value for key, value in response.headers.items()
            if key in FORWARDED_RESPONSE_HEADERS
        }
        if response.is_stream_consumed:
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=selected,
            )
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=selected,
            background=BackgroundTask(response.aclose),
        )

    return app


def main():
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("YUE2_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("YUE2_WEB_PORT", "8016")),
        workers=1,
    )
