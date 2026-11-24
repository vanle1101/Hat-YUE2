"""Local and OpenAI-compatible lyric transcription with bounded audio decoding."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import wave
from urllib.parse import urlsplit


MAX_AUDIO_SECONDS = 12 * 60
MAX_LYRIC_CHARS = 16_000
SAMPLE_RATE = 16_000


class LyricsUnavailable(RuntimeError):
    """A local ASR dependency or model is unavailable."""


def decode_wav(source, directory):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise LyricsUnavailable("Cần FFmpeg trong PATH để nhận diện lời bài hát.")
    input_path = directory / "source.audio"
    output_path = directory / "mono.wav"
    with input_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    if not input_path.stat().st_size:
        raise ValueError("Tệp âm thanh trống.")
    try:
        subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file",
             "-format_whitelist", "mp3,wav,flac,mov,ogg,aac,matroska,webm",
             "-i", str(input_path), "-map", "0:a:0", "-t", str(MAX_AUDIO_SECONDS + 1),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(output_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("Không đọc được tệp âm thanh. Hãy thử xuất lại sang MP3 hoặc WAV.") from error
    with wave.open(str(output_path), "rb") as audio:
        if audio.getnframes() > MAX_AUDIO_SECONDS * SAMPLE_RATE:
            raise ValueError("Nhận diện lời hỗ trợ bài dài tối đa 12 phút.")
        if not audio.getnframes():
            raise ValueError("Tệp không có âm thanh.")
    return output_path


def decode_audio(source, directory):
    import numpy as np

    with wave.open(str(decode_wav(source, directory)), "rb") as audio:
        pcm = audio.readframes(audio.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def lyric_result(text, language):
    lyrics = text.strip()
    if not lyrics:
        raise ValueError("Chưa nhận diện được lời. Bạn có thể tự dán lời bài hát.")
    if len(lyrics) > MAX_LYRIC_CHARS:
        raise ValueError("Lời nhận diện vượt 16.000 ký tự. Hãy dùng đoạn âm thanh ngắn hơn.")
    return {"lyrics": lyrics, "language": language}


class ApiLyricsRecognizer:
    def __init__(self, base_url, model, api_key, *, work_dir, transport=None):
        parts = urlsplit(base_url)
        if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
                or parts.query or parts.fragment):
            raise ValueError("YUE2_LYRICS_API_BASE_URL phải là URL HTTPS không chứa thông tin đăng nhập.")
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.host = parts.hostname
        self.model = model
        self._api_key = api_key
        self.work_dir = Path(work_dir)
        self.transport = transport

    def transcribe(self, source, language):
        import httpx

        data = {"model": self.model, "response_format": "json"}
        if language:
            data["language"] = language
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="lyrics-api-", dir=self.work_dir) as tmp:
            audio_path = decode_wav(source, Path(tmp))
            try:
                with audio_path.open("rb") as audio, httpx.Client(
                        timeout=httpx.Timeout(300, connect=10), follow_redirects=False,
                        transport=self.transport) as client:
                    response = client.post(
                        self.url, data=data, files={"file": ("track.wav", audio, "audio/wav")},
                        headers={"Authorization": "Bearer " + self._api_key},
                    )
            except httpx.TimeoutException:
                raise LyricsUnavailable("API nhận diện quá thời gian chờ. Thử lại hoặc chọn Trên máy.") from None
            except httpx.HTTPError:
                raise LyricsUnavailable("Không kết nối được với API nhận diện. Kiểm tra mạng hoặc chọn Trên máy.") from None
        if response.status_code in (401, 403):
            raise LyricsUnavailable("API từ chối khóa hoặc quyền nhận diện âm thanh. Kiểm tra khóa và quyền của tài khoản.")
        if response.status_code == 429:
            raise LyricsUnavailable("API đang giới hạn lượt nhận diện. Thử lại sau hoặc chọn Trên máy.")
        if not response.is_success:
            raise LyricsUnavailable(f"API nhận diện trả HTTP {response.status_code}. Kiểm tra model và trạng thái proxy hoặc chọn Trên máy.")
        try:
            result = response.json()
            text = result["text"]
            if not isinstance(text, str):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            raise LyricsUnavailable("API trả kết quả không đúng định dạng.") from None
        return lyric_result(text, language)


class LyricsRecognizer:
    def __init__(self, model, *, local_files_only, work_dir):
        self.model = model
        self.local_files_only = local_files_only
        self.work_dir = Path(work_dir)
        self._whisper = None

    def _load(self):
        if self._whisper is None:
            try:
                from faster_whisper import WhisperModel
            except (ImportError, OSError) as error:
                raise LyricsUnavailable(
                    "Chưa cài nhận diện lời. Cài faster-whisper trong môi trường YuE2 rồi khởi động lại."
                ) from error
            try:
                self._whisper = WhisperModel(
                    self.model, device="cpu", compute_type="int8",
                    cpu_threads=min(4, os.cpu_count() or 1),
                    local_files_only=self.local_files_only,
                )
            except Exception as error:
                raise LyricsUnavailable(
                    "Không nạp được Whisper. Kiểm tra model hoặc cấu hình YUE2_LYRICS_MODEL."
                ) from error
        return self._whisper

    def transcribe(self, source, language):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="lyrics-", dir=self.work_dir) as tmp:
            audio = decode_audio(source, Path(tmp))
            segments, info = self._load().transcribe(
                audio, language=language, task="transcribe", beam_size=5,
                condition_on_previous_text=False,
            )
            lines = [segment.text.strip() for segment in segments if segment.text.strip()]
        return lyric_result("\n".join(lines), info.language)
