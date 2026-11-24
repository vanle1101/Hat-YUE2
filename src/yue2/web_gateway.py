"""Cổng web công khai cho dịch vụ tạo nhạc YuE2 (không yêu cầu khóa API)."""
from contextlib import asynccontextmanager
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
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


def _default_runtime():
    return {
        "backend": os.environ.get("YUE2_BACKEND", "vllm"),
        "device": os.environ.get("YUE2_DEVICE", "cuda"),
        "ar_concurrency": int(os.environ.get("YUE2_AR_CONCURRENCY", "4")),
        "nar_batch_size": int(os.environ.get("YUE2_NAR_BATCH_SIZE", "2")),
        "ar_nar_overlap": os.environ.get("YUE2_AR_NAR_OVERLAP", "true").lower() == "true",
    }


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
            "style-src 'self' 'unsafe-inline'; media-src 'self' blob:; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response

    @app.get("/")
    async def page():
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

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
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
        except httpx.HTTPError:
            return JSONResponse(
                {"detail": "Dịch vụ suy luận tạm thời không khả dụng, vui lòng thử lại sau"},
                503,
                headers={"Retry-After": "10"},
            )
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
        host=os.environ.get("YUE2_WEB_HOST", "0.0.0.0"),
        port=int(os.environ.get("YUE2_WEB_PORT", "8016")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
