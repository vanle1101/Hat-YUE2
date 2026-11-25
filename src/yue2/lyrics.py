"""Local and OpenAI-compatible lyric transcription with bounded audio decoding."""
from __future__ import annotations

import base64
import gc
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import wave
from urllib.parse import urlsplit

from .lyric_sections import HEADING, SYSTEM_PROMPT

log = logging.getLogger("yue2.lyrics")


MAX_AUDIO_SECONDS = 12 * 60
MAX_LYRIC_CHARS = 16_000
SAMPLE_RATE = 16_000
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_FALLBACK_MODELS = (
    "gemini-3-flash-preview",
    "gemini-3.6-flash",
    "gemini-flash-lite-latest",
)
GEMINI_STYLE_FALLBACK_MODELS = (
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.6-flash",
)


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


def encode_mp3(source, directory):
    wav_path = decode_wav(source, directory)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise LyricsUnavailable("Cần FFmpeg trong PATH để nhận diện lời bài hát.")
    mp3_path = directory / "mono.mp3"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-nostdin", "-v", "error", "-i", str(wav_path),
             "-vn", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", str(mp3_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("Không nén được tệp âm thanh sang MP3 để gửi nhận diện.") from error
    if not mp3_path.is_file() or mp3_path.stat().st_size == 0:
        raise ValueError("Không tạo được tệp MP3 hợp lệ.")
    return mp3_path


def _validate_https_base_url(base_url: str, env_name: str):
    parts = urlsplit(base_url)
    if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment):
        raise ValueError(f"{env_name} phải là URL HTTPS không chứa thông tin đăng nhập.")
    return base_url.rstrip("/"), parts.hostname


def _dedupe_models(primary: str, fallbacks: tuple[str, ...]) -> tuple[str, ...]:
    ordered = []
    for item in (*[p.strip() for p in str(primary or "").split(",")], *fallbacks):
        if item and item not in ordered:
            ordered.append(item)
    return tuple(ordered)


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return ""
    return "".join(
        part.get("text", "") for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
    ).strip()


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


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
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/audio/transcriptions"
        self.host = parts.hostname
        self.model = model
        self._api_key = api_key
        self.work_dir = Path(work_dir)
        self.transport = transport

    def propose_sections(self, text, model):
        import httpx

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(22, connect=5), follow_redirects=False,
                              transport=self.transport) as client:
                response = client.post(
                    self.base_url + "/chat/completions", json=payload,
                    headers={"Authorization": "Bearer " + self._api_key},
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            return None

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
    def __init__(self, model, *, local_files_only, work_dir, device="auto", unload_after=False):
        self.model = model
        self.local_files_only = local_files_only
        self.work_dir = Path(work_dir)
        self.device = device
        self.unload_after = unload_after
        self._whisper = None
        self._loaded_device = None

    def _choose_device(self):
        if self.device == "cpu":
            return "cpu", "int8"
        try:
            import torch
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                if self.device == "cuda" or free >= 3.5 * 1024 ** 3:
                    return "cuda", "float16"
        except Exception:
            if self.device == "cuda":
                raise LyricsUnavailable("Không dùng được CUDA cho Whisper; đổi YUE2_LYRICS_DEVICE=auto hoặc cpu.") from None
        return "cpu", "int8"

    def close(self):
        loaded = self._loaded_device
        self._whisper = None
        self._loaded_device = None
        if loaded == "cuda":
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _load(self, *, force_cpu=False):
        if self._whisper is not None and not force_cpu:
            return self._whisper
        try:
            from faster_whisper import WhisperModel
        except (ImportError, OSError) as error:
            raise LyricsUnavailable(
                "Chưa cài nhận diện lời. Cài faster-whisper trong môi trường YuE2 rồi khởi động lại."
            ) from error
        device, compute_type = ("cpu", "int8") if force_cpu else self._choose_device()
        kwargs = {"local_files_only": self.local_files_only}
        if device == "cpu":
            kwargs["cpu_threads"] = min(4, os.cpu_count() or 1)
        try:
            self._whisper = WhisperModel(self.model, device=device, compute_type=compute_type, **kwargs)
            self._loaded_device = device
            return self._whisper
        except Exception as error:
            if device == "cuda" and self.device == "auto":
                log.warning("Whisper CUDA load failed (%s); falling back to CPU int8", error)
                return self._load(force_cpu=True)
            raise LyricsUnavailable(
                "Không nạp được Whisper. Kiểm tra model hoặc cấu hình YUE2_LYRICS_MODEL."
            ) from error

    def _run_transcribe(self, model, audio, language):
        segments, info = model.transcribe(
            audio, language=language, task="transcribe", beam_size=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.55,
            compression_ratio_threshold=2.2,
        )
        lines = [segment.text.strip() for segment in segments if segment.text.strip()]
        return lines, info.language

    def transcribe(self, source, language):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="lyrics-", dir=self.work_dir) as tmp:
            audio = decode_audio(source, Path(tmp))
            try:
                try:
                    lines, detected = self._run_transcribe(self._load(), audio, language)
                except Exception as error:
                    if self._loaded_device == "cuda" and self.device == "auto":
                        log.warning("Whisper CUDA transcribe failed (%s); retrying on CPU int8", error)
                        self.close()
                        lines, detected = self._run_transcribe(self._load(force_cpu=True), audio, language)
                    else:
                        raise
            finally:
                if self.unload_after and self._loaded_device == "cuda":
                    self.close()
        return lyric_result("\n".join(lines), detected)


class GeminiLyricsRecognizer:
    """Transcribe and structure song lyrics directly via Gemini multimodal audio."""

    def __init__(self, api_key, *, model="gemini-3-flash-preview",
                 base_url=DEFAULT_GEMINI_BASE_URL, work_dir, transport=None):
        self.base_url, self.host = _validate_https_base_url(base_url, "YUE2_GEMINI_API_BASE_URL")
        self._api_key = api_key
        self.models = _dedupe_models(model, GEMINI_FALLBACK_MODELS)
        self.work_dir = Path(work_dir)
        self.transport = transport

    def _call_model(self, client, model_name: str, parts: list[dict], *, temperature: float = 0.0):
        return client.post(
            f"{self.base_url}/models/{model_name}:generateContent",
            headers={"x-goog-api-key": self._api_key},
            json={
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": temperature},
            },
        )

    def propose_sections(self, text, model=None):
        import httpx

        models = _dedupe_models(model or "", self.models)
        parts = [{"text": f"{SYSTEM_PROMPT}\n\n{text}"}]
        with httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=False,
                          transport=self.transport) as client:
            for model_name in models:
                try:
                    response = self._call_model(client, model_name, parts)
                    if response.status_code == 200:
                        extracted = _strip_code_fences(_extract_gemini_text(response.json()))
                        if extracted:
                            return extracted
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    continue
        return None

    def transcribe(self, source, language):
        import httpx

        lang_hint = {
            "vi": "The vocals are in Vietnamese. Preserve accurate Vietnamese spelling and diacritics.",
            "en": "The vocals are in English. Transcribe accurately in English.",
        }.get(language, "Keep the exact original language(s) sung (e.g., Vietnamese with accurate diacritics if Vietnamese).")
        prompt = (
            "Transcribe ONLY the sung lyrics from this song audio accurately, word for word.\n"
            f"{lang_hint}\n"
            "Ignore spoken intro/outro banter, MC talk, applause, crowd noise, and purely instrumental passages.\n"
            "Format cleanly with numbered section tags such as [Verse 1], [Chorus 1], [Pre-Chorus 1], [Bridge 1], [Outro 1], "
            "short natural lyric lines, and blank lines between sections.\n"
            "Return ONLY the formatted lyrics without explanations or markdown code fences.\n"
            "If the audio has no sung vocals at all, reply with exactly: NO_VOCALS"
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="lyrics-gemini-", dir=self.work_dir) as tmp:
            mp3_path = encode_mp3(source, Path(tmp))
            audio_b64 = base64.b64encode(mp3_path.read_bytes()).decode("ascii")
        parts = [
            {"inline_data": {"mime_type": "audio/mp3", "data": audio_b64}},
            {"text": prompt},
        ]

        last_error = None
        with httpx.Client(timeout=httpx.Timeout(120, connect=10), follow_redirects=False,
                          transport=self.transport) as client:
            for model_name in self.models:
                try:
                    response = self._call_model(client, model_name, parts)
                except httpx.TimeoutException:
                    last_error = "Gemini API quá thời gian chờ."
                    continue
                except httpx.HTTPError:
                    last_error = "Không kết nối được với Gemini API."
                    continue

                if response.status_code in (401, 403):
                    raise LyricsUnavailable("Gemini API từ chối khóa hoặc quyền truy cập. Kiểm tra YUE2_GEMINI_API_KEY.")
                if response.status_code in (404, 429, 500, 502, 503, 504):
                    last_error = f"Gemini API trả HTTP {response.status_code}."
                    continue
                if not response.is_success:
                    raise LyricsUnavailable(f"Gemini API trả HTTP {response.status_code}.")
                try:
                    extracted = _strip_code_fences(_extract_gemini_text(response.json()))
                except (ValueError, KeyError, TypeError):
                    last_error = "Gemini API trả kết quả không đúng định dạng."
                    continue
                if not extracted:
                    last_error = "Gemini không trả về lời bài hát."
                    continue
                if extracted.upper() == "NO_VOCALS":
                    raise ValueError("Chưa nhận diện được lời. Bạn có thể tự dán lời bài hát.")
                return lyric_result(extracted, language or "auto")

        raise LyricsUnavailable(last_error or "Gemini API tạm thời không khả dụng.")


def _lyric_word_count(text: str) -> int:
    return sum(
        len(line.split()) for line in text.splitlines()
        if line.strip() and not HEADING.fullmatch(line.strip())
    )


def proofread_lyrics_with_gemini(
        raw_text: str,
        language: str | None,
        api_key: str,
        *,
        model: str = "gemini-3-flash-preview",
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        transport=None) -> str | None:
    """Repair Whisper ASR homophone/diacritic errors and add song section tags (fail-open)."""
    import httpx

    cleaned_in = raw_text.strip()
    if not cleaned_in or not api_key:
        return None
    try:
        resolved_base, _ = _validate_https_base_url(base_url, "YUE2_GEMINI_API_BASE_URL")
    except ValueError:
        return None
    models = _dedupe_models(model, GEMINI_FALLBACK_MODELS)
    lang_hint = {
        "vi": "The song is in Vietnamese; repair broken diacritics and phonetic mishearings into natural Vietnamese.",
        "en": "The song is in English; repair phonetic mishearings into natural English.",
    }.get(language, "Preserve the original sung language and repair phonetic mishearings/diacritics naturally.")
    prompt = (
        "You are a song lyrics proofreader repairing Whisper ASR transcription errors from singing over music.\n"
        f"{lang_hint}\n"
        "Strict rules:\n"
        "1. Fix ONLY obvious phonetic/homophone ASR mishearings and diacritic errors so each line makes natural sense.\n"
        "2. NEVER invent new verses and NEVER replace the song with a different song; keep the original structure, order, and chorus repeats.\n"
        "3. Remove obvious non-sung crowd chatter or MC banter if present.\n"
        "4. Format cleanly with numbered section tags ([Verse 1], [Chorus 1], [Pre-Chorus 1], [Bridge 1], [Outro 1]), short lines, and blank lines between sections.\n"
        "5. Output ONLY the cleaned lyrics with no explanations and no markdown code fences.\n\n"
        f"ASR transcript:\n{cleaned_in[:MAX_LYRIC_CHARS]}"
    )
    raw_words = _lyric_word_count(cleaned_in)
    with httpx.Client(timeout=httpx.Timeout(35, connect=5), follow_redirects=False,
                      transport=transport) as client:
        for model_name in models:
            try:
                response = client.post(
                    f"{resolved_base}/models/{model_name}:generateContent",
                    headers={"x-goog-api-key": api_key},
                    json={
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.1},
                    },
                )
                if response.status_code != 200:
                    continue
                candidate = _strip_code_fences(_extract_gemini_text(response.json()))
                if not candidate or len(candidate) > MAX_LYRIC_CHARS:
                    continue
                cand_words = _lyric_word_count(candidate)
                if raw_words >= 8 and not (0.5 * raw_words <= cand_words <= 1.5 * raw_words):
                    continue
                return candidate
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                continue
    return None


class HybridGeminiLyricsRecognizer:
    """Acoustic-anchored Whisper transcription followed by Gemini ASR proofreading."""

    def __init__(self, local_recognizer, api_key, *, model="gemini-3-flash-preview",
                 base_url=DEFAULT_GEMINI_BASE_URL, work_dir, transport=None):
        self.local_recognizer = local_recognizer
        self.base_url, self.host = _validate_https_base_url(base_url, "YUE2_GEMINI_API_BASE_URL")
        self._api_key = api_key
        self.model = model
        self.work_dir = Path(work_dir)
        self.transport = transport
        self._audio_fallback = GeminiLyricsRecognizer(
            api_key, model=model, base_url=base_url, work_dir=work_dir, transport=transport,
        )

    def propose_sections(self, text, model=None):
        return self._audio_fallback.propose_sections(text, model=model)

    def transcribe(self, source, language):
        try:
            raw = self.local_recognizer.transcribe(source, language)
        except LyricsUnavailable:
            source.seek(0)
            return self._audio_fallback.transcribe(source, language)
        detected = raw.get("language") or language
        polished = proofread_lyrics_with_gemini(
            raw["lyrics"], detected, self._api_key,
            model=self.model, base_url=self.base_url, transport=self.transport,
        )
        if polished:
            return lyric_result(polished, detected or "auto")
        return raw


def optimize_style_with_gemini(
        style: str,
        lyrics: str,
        api_key: str,
        *,
        model: str = "gemini-flash-lite-latest",
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        transport=None) -> str | None:
    """Translate/polish a user style prompt into concise English YuE2 musical tags (fail-open)."""
    import httpx

    cleaned = style.strip()
    if not cleaned or not api_key:
        return None
    try:
        resolved_base, _ = _validate_https_base_url(base_url, "YUE2_GEMINI_API_BASE_URL")
    except ValueError:
        return None
    models = _dedupe_models(model, GEMINI_STYLE_FALLBACK_MODELS)
    prompt = (
        "Convert the following song arrangement/style request into a single concise comma-separated line of "
        "English musical style tags optimized for the YuE2 music generation model (genre, mood, instruments, "
        "vocal timbre/gender, tempo/groove).\n"
        "Preserve the user's exact musical intent and instruments (for example, if they ask for 'only piano', "
        "keep solo piano and do not add drums). If the lyrics or style are Vietnamese, include 'Vietnamese' as the first tag.\n"
        "Output ONLY the comma-separated tag line (max 280 characters), no quotes, no bullet points, no commentary.\n"
        f"Style input: {cleaned[:600]}\n"
        f"Lyrics snippet: {lyrics[:240]}"
    )
    with httpx.Client(timeout=httpx.Timeout(10, connect=4), follow_redirects=False,
                      transport=transport) as client:
        for model_name in models:
            try:
                response = client.post(
                    f"{resolved_base}/models/{model_name}:generateContent",
                    headers={"x-goog-api-key": api_key},
                    json={
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.1},
                    },
                )
                if response.status_code != 200:
                    continue
                text = _strip_code_fences(_extract_gemini_text(response.json()))
                line = " ".join(part.strip() for part in text.splitlines() if part.strip()).strip('"\' ')
                if line and len(line) <= 600:
                    return line
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                continue
    return None
