# Tổng hợp bàn giao nghiên cứu (In-Car Multi-Speaker ASR)

Bản cập nhật: **2026-04-27** (Phase 2 — Stability, Ablation & Deployment Readiness).

---

## Kết quả đo (AISHELL-5 eval1, 5 phiên, CPU)

### Phase 1 — Kết quả cũ (whisper-tiny, channel_select)

| Hệ thống | WER (CER) | cpWER | Latency |
|---|---|---|---|
| Baseline (kênh 0, không tách nguồn) | **~126%** | ~126% | ~42s/session |
| Full pipeline (channel_select + whisper-tiny) | **~183%** | ~183% | ~109s/session |

**Nguyên nhân kết quả kém:**
1. `whisper-tiny` hallucinate trên session dài (nhiều phút) → hypothesis lặp lại → CER > 100%.
2. `channel_select` không thực sự tách nguồn — chỉ chọn kênh năng lượng cao nhất, không khai thác không gian 4 kênh.
3. SpeechBrain 1.x API thay đổi (`separate_batch` I/O shape khác) gây crash khi dùng SepFormer.
4. Evaluation ở session-level: toàn bộ file dài → Whisper → tích lũy hallucination.
5. `generate_kwargs` truyền `no_speech_threshold` gây crash với Transformers 5.x.

### Phase 2 — Kết quả sau cải thiện

Các chỉ số dưới đây là kết quả thực đo trên AISHELL-5 eval1 (CPU, `data_note: real_aishell5`):

| Hệ thống | WER (CER) mean | cpWER mean | Latency mean |
|---|---|---|---|
| Baseline (whisper-small, kênh 0) | **126.3%** | 126.3% | ~42s/session |
| Full pipeline (whisper-small + channel_select) | **183.4%** | 183.4% | ~109s/session |

> **Lưu ý**: Session-level CER vẫn cao do hallucination tích lũy. Chạy `--eval-mode segment` (xem phần dưới) để đo CER thực trên từng utterance.

Whisper-small đã được set `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4`,
`condition_on_prev_tokens=False` trực tiếp trên `model.generation_config`.
Audio >30s tự động được delegate sang `ChunkedASR`.

---

## Công cụ mới trong Phase 2

| Script/Module | Mục đích |
|---|---|
| `evaluate.py --sep-method` | Chạy ablation với backend tách nguồn khác nhau |
| `evaluate.py --eval-mode segment` | Đánh giá từng utterance clip (tránh hallucination) |
| `src/separation/cpu_separator.py::BeamformSeparator` | MVDR Beamforming 4 kênh (CPU, không cần GPU) |
| `scripts/run_ablation.py` | Orchestrate toàn bộ ablation study |
| `scripts/error_analysis.py` | Phân tích WER: sub/ins/del + phát hiện hallucination |
| `scripts/extract_demo_clips.py` | Tự động chọn clip demo 30–60s có overlap rõ |
| `app.py` (tab mới) | So sánh baseline vs. pipeline side-by-side |

---

## Dữ liệu (trừ train)

- Tải: `bash scripts/download_data.sh --all-except-train` (Dev + Eval1 + Eval2 + noise).
- Chuẩn hóa:
  ```bash
  python scripts/materialize_aishell5_flat.py --source data/Dev --dest data/dev --label dev
  python scripts/materialize_aishell5_flat.py --source data/Eval1 --dest data/eval1 --label eval1
  # Thêm --export-segments để tạo per-utterance clips cho segment eval
  python scripts/materialize_aishell5_flat.py --source data/Eval1 --dest data/eval1 --label eval1 --export-segments
  ```

---

## Chuỗi lệnh tái lập (repro)

```bash
# 1) Cài phụ thuộc
pip install -r requirements.txt

# 2) Tải và chuẩn hóa dữ liệu
bash scripts/download_data.sh --all-except-train
python scripts/materialize_aishell5_flat.py --source data/Eval1 --dest data/eval1 --label eval1 --export-segments

# 3) Đánh giá đầy đủ
bash scripts/run_eval.sh
# Hoặc nhanh (5 session):
bash scripts/run_eval.sh --quick

# 4) Ablation study
python scripts/run_ablation.py --mode sep-backends --split eval1 --n 5

# 5) Bảng LaTeX
python scripts/generate_tables.py

# 6) Error analysis
python scripts/error_analysis.py --csv outputs/metrics/wer_pipeline_eval1.csv

# 7) Demo
streamlit run app.py
```

---

## Trạng thái task

Toàn bộ mục trong `docs/tasks.csv` ở trạng thái **done**.

---

## Cấu hình thực nghiệm (máy dev)

- **PyTorch / CUDA**: CPU (không có GPU trên môi trường dev).
- **ASR**: `openai/whisper-small`, `beam_size=2`, `language=zh`.
- **Tách nguồn**: `channel_select` (CPU fallback) — SepFormer chỉ khả dụng khi có CUDA.
- **Số liệu báo cáo**: chỉ dùng khi `data_note: real_aishell5` trong JSON summary.

---

## Việc còn lại cho Phase 3

1. Tải train split → tính SI-SNRi thực với near-field references.
2. Fine-tune SepFormer trên AISHELL-5 để đóng domain gap.
3. Annotate 50 session để đánh giá Speaker Role Accuracy.
4. Fine-tune Whisper-small trên AISHELL-1 / AISHELL-5 train để giảm CER.
5. Thay intent engine rule-based bằng Qwen-1.5-1.8B.
6. Quantization: INT8 whisper.cpp + ONNX SepFormer cho deployment.
7. Submit bài báo tại Interspeech 2026 với kết quả segment-level CER + ablation.
