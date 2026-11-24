# Các ví dụ mẫu

`song.json` chứa lời bài hát và mô tả phong cách mẫu **City Lights**. `melody.abc` là bản nhạc giai điệu 8 ô nhịp tương ứng. `score.abc` bổ sung phần hòa âm; `score-jazz.abc` thay đổi các ký hiệu hợp âm.

Từ thư mục gốc của repository:

```bash
# Tạo bài hát và bản nhạc từ lời và phong cách.
python examples/generate.py --output outputs/create

# Hiện thực hóa giai điệu có sẵn với phần đệm tự do.
python examples/generate.py --abc-file examples/melody.abc \
  --cot melody --output outputs/melody

# So sánh hai cách hòa âm với cùng giai điệu, lời bài hát, phong cách và hạt giống.
python examples/generate.py --abc-file examples/score.abc \
  --cot full --output outputs/harmony-original
python examples/generate.py --abc-file examples/score-jazz.abc \
  --cot full --output outputs/harmony-edited
```

Script lưu toàn bộ kết quả gốc, từ chối ghi đè thư mục đã tồn tại và báo lỗi nếu quá trình sinh bị cắt ngắn. Dùng `--revision` và `--vae-revision` để cố định phiên bản model.

Kiểm tra can thiệp bản nhạc ABC không cần GPU:

```bash
python skills/yue2-music/scripts/abc_tools.py inspect examples/score.abc
python skills/yue2-music/scripts/abc_tools.py compare \
  examples/score.abc examples/score-jazz.abc
```
