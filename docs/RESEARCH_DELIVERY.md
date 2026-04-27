# Tổng hợp bàn giao nghiên cứu (In-Car Multi-Speaker ASR)

Bản cập nhật: 2026-04-25 (lần chạy đánh giá thực tế trên eval1, CPU).

## Kết quả đo gần nhất (AISHELL-5 eval1, 18 phiên, `data_note: real_aishell5`)

- **Cấu hình:** `whisper-tiny` + tách nguồn `channel_select` (CPU, không SepFormer).
- **Baseline (1 kênh từ mixture):** WER (CER) trung bình ≈ **158%**, cpWER ≈ **142%** — xem `outputs/metrics/summary_baseline_eval1.json`.
- **Full pipeline:** WER theo từng bản ghi nói n ≈ **36** (2 speaker × 18 file), trung bình CER **≈184%**; **cpWER ≈181%** (n=18) — `summary_pipeline_eval1.json`. Latency: trung bình **~77s**/session (P95 **~106s**), do session dài (vài trăm giây) và full-file / chunked ASR trên CPU.
- **Lưu ý:** CER có thể **> 100%** (chuẩn tiếng Trung) khi giả thuyết dài, tham chiếu toàn bộ câu, hoặc lệch speaker; đây là **kết quả thực đo** trên tập materialize từ OpenSLR, không mock.
- Với tách kênh đơn giản, pipeline **không** cải thiện so vaseline baseline dưới cùng thước đo này — kỳ vọng hợp lý tới khi bật GPU + SepFormer / tinh chỉnh.

## Dữ liệu (trừ train)

- Tải: `bash scripts/download_data.sh --all-except-train` (Dev + Eval1 + Eval2 + noise, không `train.tar.gz`).
- Cây thư mục gốc OpenSLR: session `*/DX01-04C01.wav` + `DX01C01.TextGrid` — cần **chuẩn hóa** sang `wav/*.wav` 4 kênh + `text/*.txt` bằng:
  - `python scripts/materialize_aishell5_flat.py --source data/Eval1 --dest data/eval1 --label eval1` (tương tự `Dev` → `dev`, `Eval2` → `eval2`).
- Thư mục nguồn raw có thể giữ tên `data/dev_openslr_raw` sau khi tách.

## Trạng thái task

Toàn bộ mục trong `docs/tasks.csv` đang ở trạng thái **done**, kèm ghi chú ngắn ở cột **Note** (gồm cả sửa OpenSLR `Dev/Eval1/Eval2` và chính sách dữ liệu thật so với fixture).

## Cấu hình thực nghiệm thực tế (máy dev hiện tại)

- **PyTorch / CUDA:** `CUDA: False` khi chạy `run_eval.sh` trên CPU.
- **ASR:** `configs/default.yaml` dùng `openai/whisper-tiny` (CPU-friendly).
- **Tách nguồn:** trên CPU, pipeline dùng `cpu_fallback_method: channel_select` (không tải SepFormer đầy đủ trên GPU).
- **Bảng số trong paper** được sinh từ `outputs/metrics/summary_*.json` qua `scripts/generate_tables.py`; mỗi file summary có trường **`data_note`**. Chỉ khi `data_note` phản ánh **AISHELL-5 thật** thì mới dùng các chỉ số đó như kết quả benchmark cuối.

## Dữ liệu OpenSLR #159 (AISHELL-5)

- Tải: `bash scripts/download_data.sh` (ánh xạ `Dev.tar.gz` / `Eval1.tar.gz` / `Eval2.tar.gz` — tên chữ hoa trên server).
- **Sau tải:** script kiểm tra **`gzip -t`** trước khi `tar -xzf` để tránh giải nén file chưa tải xong.
- Các gói lớn (`train.tar.gz`, `noise.tar.gz`) tùy nhu cầu huấn luyện/augment; cho đánh giá **eval1** thường chỉ cần **Dev + Eval1** (và tùy chọn Eval2).

## Chuỗi lệnh tái lập (repro)

```bash
# 1) Cài phụ thuộc
pip install -r requirements.txt

# 2) Tải và giải nén dữ liệu (cần đủ dung lượng + thời gian; eval2+noise vài chục GB)
bash scripts/download_data.sh --all-except-train
# 2b) Chuẩn hóa cây session OpenSLR → data/{split}/wav + text/ (4 kênh + tham chiếu)
#     python scripts/materialize_aishell5_flat.py --source data/Dev --dest data/dev --label dev
#     (tương tự Eval1 → eval1, Eval2 → eval2)

# 3) Đánh giá (n đầy đủ) — yêu cầu data/eval1 thật
bash scripts/run_eval.sh
# Nhanh, vẫn cần eval1/ có .wav
bash scripts/run_eval.sh --quick
# Chỉ môi trường CI / không có bộ eval: (KHÔNG dùng cho số liệu báo cáo)
# bash scripts/run_eval.sh --synthetic-ok

# 4) Bảng LaTeX (run_eval cũng gọi bước này; có thể chạy lại)
python scripts/generate_tables.py
cp -f outputs/tables/*.tex paper/tables/

# 5) PDF paper
bash scripts/build_paper.sh
```

## Ghi chú cài đặt gần đây

- `scripts/download_data.sh`: cờ `--all-except-train`, `gzip -t` trước `tar`, sửa lỗi `set -e` khi thư mục split chưa tồn tại, bỏ qua tải nếu đã có `*/wav/*.wav` hoặc cây raw có `.wav`.
- `scripts/materialize_aishell5_flat.py`: gộp DX01–04C01 → 4 kênh; TextGrid → `text/*.txt` (SPK*).
- Log đầy đủ: `outputs/run_eval_full_20260425_070050.log` (lần chạy eval1 thật).

## Việc có thể làm thêm (sau khi Eval2 + noise tải xong)

1. `python scripts/materialize_aishell5_flat.py --source data/Eval2 --dest data/eval2 --label eval2`
2. `bash scripts/run_eval.sh` lại để bật bước Eval2 (có thể tốn nhiều giờ trên CPU).
3. Tập `noise/`: dùng cho mô phỏng / augment (bộ đánh giá mặc định **không** cần noise để tính WER trên eval1).
