<p align="center">
  <img src="assets/yue2-turbo-logo.png" alt="YuE2 Turbo" width="460" />
</p>

<h1 align="center">YuE2-Turbo: Suy luận nhanh và song song cho YuE2</h1>

<p align="center"><strong>Cùng mô hình YuE2. Nhanh hơn 1,68× mỗi bài hát. Tạo nhiều bài hơn 3,31× trên một GPU.</strong></p>

<p align="center">
  <a href="#môi-trường">🧰 Môi trường</a> ·
  <a href="#triển-khai-dịch-vụ">🚀 Triển khai</a> ·
  <a href="#tham-chiếu-cấu-hình">⚙️ Cấu hình</a> ·
  <a href="#về-yue2">🎵 Về YuE2</a> ·
  <a href="https://huggingface.co/m-a-p/YuE2-3B">🤗 Trọng số</a>
</p>

**YuE2-Turbo là tầng tăng tốc suy luận và phục vụ (serving) cho [YuE2](https://github.com/multimodal-art-projection/YuE).** Dự án giữ nguyên bộ trọng số `YuE2-3B` công khai cùng công thức sinh chuẩn (BF16, 32 bước flow-matching) và tối ưu cách thực thi: các bước tự hồi quy (AR) chạy trên **vLLM** (hoặc PyTorch `torch` trên CPU/Windows), toàn bộ trọng số giữ **thường trú**, tổng hợp âm thanh (NAR) chạy gộp batch, đi kèm **API hàng đợi HTTP + giao diện web Studio** (đã gỡ bỏ yêu cầu `YUE2_API_KEY` để dùng cục bộ trực tiếp).

## Môi trường

| Thành phần | Khuyến nghị / Hỗ trợ |
|---|---|
| Hệ điều hành | Linux x86_64 (cho `vllm` / `triton`); Windows hỗ trợ chạy qua `torch` (CPU/GPU) |
| GPU | Khuyến nghị NVIDIA hỗ trợ BF16 với 24–32 GB VRAM cho cấu hình tăng tốc; có thể chạy CPU qua `YUE2_BACKEND=torch` và `YUE2_DEVICE=cpu` |
| Python | 3.10, 3.11 hoặc 3.12 |
| PyTorch | 2.10.0 (`cu128` hoặc `cpu`) |
| vLLM / Triton | 0.19.0 / 3.6.0 (chỉ trên Linux qua extra `server`) |
| Ổ đĩa | ~8 GB cho `YuE2-3B` + `YuE2-Vae` |

## Triển khai dịch vụ

### 1. Chạy nhanh trên Windows (đã cấu hình sẵn)

Trong thư mục dự án đã cấu hình sẵn `.venv`, model tại `models/` và tệp `.env`:

- Chạy [start_studio.bat](start_studio.bat) (hoặc `.\start_studio.ps1`) để bật cả API (`http://127.0.0.1:8000`) và Web Studio (`http://127.0.0.1:8016`).
- Không cần nhập `YUE2_API_KEY`: mở thẳng `http://127.0.0.1:8016` để tạo bài hát, nghe thử và tải tệp `.flac` / `.abc`.

### 2. Cài đặt thủ công trên Linux (vLLM tăng tốc)

```bash
git clone <this repository> YuE2-Turbo
cd YuE2-Turbo
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install '.[server]'
```

### 3. Khởi chạy API và Web Studio

```bash
yue2-serve                                    # mặc định lắng nghe 127.0.0.1:8000
YUE2_UPSTREAM=http://127.0.0.1:8000 yue2-web  # giao diện Studio tại http://127.0.0.1:8016
```

Kiểm tra trạng thái sẵn sàng và xem tài liệu OpenAPI:

- Kiểm tra sẵn sàng: `http://127.0.0.1:8000/health/ready` (`503` khi đang nạp model, `200` khi sẵn sàng)
- OpenAPI docs: `http://127.0.0.1:8000/docs`

### 4. Gọi API trực tiếp

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: order-0001' \
  -d '{"style":"Vietnamese, acoustic piano pop, warm vocal","lyrics":"[Verse]\nÁnh nắng sớm bên khung cửa\n[Chorus]\nGửi câu hát theo làn gió","cot":"full","seed":42}'
```

| Endpoint | Mục đích |
|---|---|
| `GET /health/live`, `GET /health/ready` | Kiểm tra sống / sẵn sàng |
| `POST /v1/jobs` | Gửi tác vụ tạo bài hát (`n=2` để tạo 2 hạt giống cùng lúc) |
| `GET /v1/jobs/{id}` | Xem trạng thái: `queued → running → succeeded / truncated / failed / cancelled`, số token, thời gian |
| `POST /v1/jobs/{id}/cancel` | Hủy tác vụ đang chờ hoặc đang chạy |
| `GET /v1/jobs/{id}/audio`, `/score` | Tải tệp âm thanh FLAC và bản nhạc ABC |
| `POST /v1/covers` | Cover: trích xuất giai điệu từ tệp nhạc tải lên rồi tạo bài hát mới (`YUE2_SHEETSAGE_DEVICE` khác `off`) |

### 5. Cover một bản thu âm

Tính năng cover dùng [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) để trích xuất giai điệu thành bản nhạc ABC không hợp âm, sau đó `YuE2` sáng tác lại theo phong cách và lời bài hát mới (`cot=melody`). Cần có `ffmpeg` trên `PATH`. Từ dòng lệnh:

```bash
yue2 cover --audio song.mp3 --request cover-request.json --sheetsage-device auto --output outputs/cover
```

## Tham chiếu cấu hình

Tất cả biến môi trường đều có tiền tố `YUE2_` (xem thêm trong `.env`):

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `YUE2_BACKEND` | `vllm` | `vllm` (tăng tốc Linux), `torch` (PyTorch chuẩn), `torch-eager` |
| `YUE2_DEVICE` | `cuda` | Thiết bị chạy: `cuda`, `cpu`, `mps` |
| `YUE2_RESIDENT_MODELS` | `true` | Giữ MoT và VAE thường trú trong bộ nhớ giữa các bài |
| `YUE2_AR_CONCURRENCY` | `4` | Số tác vụ gửi vào vLLM đồng thời (≤ `YUE2_VLLM_MAX_NUM_SEQS`) |
| `YUE2_VLLM_MAX_NUM_SEQS` | `4` | Số chuỗi tối đa trong bộ lập lịch vLLM |
| `YUE2_VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | Số token tối đa mỗi bước lập lịch vLLM |
| `YUE2_VLLM_GPU_MEMORY_UTILIZATION` | `0.30` | Tỷ lệ VRAM dành cho engine vLLM |
| `YUE2_NAR_BATCH_SIZE` | `2` | Số bài tổng hợp âm thanh cùng lúc ở pha NAR (tối đa 2) |
| `YUE2_AR_NAR_OVERLAP` | `true` | Nạp trước pha AR kế tiếp khi pha hiện tại đang chạy NAR/VAE |
| `YUE2_AR_BATCH_WAIT_MS` | `50` | Thời gian chờ gom các yêu cầu gửi đồng thời |
| `YUE2_MEMORY_BUDGET_GIB` | `30` | Ngưỡng bộ nhớ cấp phát cho PyTorch |
| `YUE2_ODE_STEPS` | `32` | Số bước flow-matching |
| `YUE2_MAX_PENDING` | `16` | Độ sâu hàng đợi trước khi trả `429` |
| `YUE2_TASK_TIMEOUT_SECONDS` | `1200` | Thời gian chạy tối đa cho mỗi tác vụ (giây) |
| `YUE2_ARTIFACT_RETENTION_SECONDS` / `YUE2_ARTIFACT_MAX_GIB` | `86400` / `5` | Chính sách dọn tệp kết quả cũ |
| `YUE2_WARMUP` | `true` | Chạy nháp làm nóng model lúc khởi động (`false` để sẵn sàng ngay trên CPU) |
| `YUE2_MODEL` / `YUE2_VAE` / `YUE2_LOCAL_FILES_ONLY` | HF id / `false` | Nguồn tải trọng số mô hình |
| `YUE2_SHEETSAGE_DEVICE` | `auto` | Thiết bị trích xuất giai điệu cover: `auto`, `cpu`, `cuda`, `off` |
| `YUE2_SHEETSAGE_MIN_FREE_GIB` | `8` | VRAM trống tối thiểu (GiB) để đặt SheetSage2 lên GPU |

## Về YuE2

YuE2-Turbo phát triển dựa trên **YuE2** của cộng đồng M-A-P:

**[github.com/multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)** · [🤗 YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) · [🤗 YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) · [Trang demo](https://map-yue2.github.io/)

## Giấy phép

Mã nguồn trong repository dùng giấy phép **[Apache 2.0](LICENSE)**. Trọng số mô hình YuE2 dùng giấy phép riêng **[CC BY-NC 4.0](MODEL_LICENSE)**; các thành phần bên thứ ba giữ [giấy phép gốc](THIRD_PARTY_NOTICES.md).


## Nhận diện lời trong Studio (Windows)

Tại thư mục dự án trên ổ E, cài phần nhận diện vào môi trường Python đang chạy Studio:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir "faster-whisper==1.2.1"
```

Cần có FFmpeg trong `PATH`. Trong `.env`, đặt `YUE2_LYRICS_MODEL` thành đường dẫn thư mục Whisper CTranslate2 có sẵn để chạy offline. Nếu chưa có model, đặt `YUE2_LYRICS_LOCAL_FILES_ONLY=false` để tải Whisper đa ngôn ngữ `small` từ Hugging Face khi dùng lần đầu. Script Windows đặt `HF_HOME` bên trong repo để cache nằm trên ổ E. Chỉ model được tải về; âm thanh được xử lý trên CPU tại máy bạn.

Khởi động lại Studio. Mở **Cover**, chọn bài, chọn tiếng Việt, tiếng Anh hoặc tự phát hiện rồi bấm **Nhận diện lời**. Xem lại kết quả trước khi tạo cover vì giọng hát lẫn nhạc nền có thể làm nhận sai. Để giữ phần lời đang sửa, kết quả nhận diện sẽ hiện riêng và bạn có thể bấm **Dùng lời nhận diện** khi muốn thay thế. Giới hạn nhận diện là 12 phút và tệp dưới 40 MiB.

Cover dùng SheetSage2 để chép bản nhạc. **Phối tự do** lấy giai điệu, **Bám bản gốc** lấy bản nhạc có giai điệu và hợp âm. Tùy chọn giọng nam/nữ được đưa vào mô tả của YuE2. Chất lượng và mức độ bám phụ thuộc vào kết quả chép nhạc và mô hình tạo bài.

### Nhận diện qua API

Trong tệp `.env` của bản cài trên ổ E, thêm:

```dotenv
YUE2_LYRICS_API_BASE_URL=https://codex.vorte.me/v1
YUE2_LYRICS_API_MODEL=gpt-4o-transcribe
YUE2_LYRICS_API_KEY=<khoa-proxy-moi>
```

Khởi động lại bằng `start_studio.bat`, mở **Cover**, chọn một tệp và chọn **API** trước khi bấm **Nhận diện lời**. Khi đó, Studio gửi bản âm thanh đã chuẩn hóa tới máy chủ API. Chọn **Trên máy** để giữ âm thanh cục bộ. Cả hai chế độ cần FFmpeg trong `PATH`; riêng chế độ tại máy cần Faster-Whisper. Giữ Studio trên localhost khi cấu hình khóa và lưu khóa trong `.env` cục bộ.
