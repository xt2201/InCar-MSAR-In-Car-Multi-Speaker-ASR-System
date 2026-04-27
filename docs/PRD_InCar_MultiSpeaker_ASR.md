# PRD – In-Car Multi-Speaker ASR System
**Dự án:** Hệ thống nhận dạng và hiểu giọng nói đa người trong ô tô  
**Dataset:** AISHELL-5 (OpenSLR #159)

---

## 1. Tổng quan sản phẩm (Product Overview)

### 1.1 Tầm nhìn (Vision)
Xây dựng hệ thống AI end-to-end có khả năng **nghe, tách, nhận dạng và hiểu lệnh thoại từ nhiều người cùng nói trong cabin ô tô** – điều mà các voice assistant thương mại (Siri, Alexa, Google Assistant) hiện chưa làm được.

### 1.2 Tuyên ngôn sản phẩm
> *"Xe của bạn hiểu được khi nhiều người nói cùng lúc."*

### 1.3 Mục tiêu kinh doanh
| Mục tiêu | Chỉ tiêu đo lường |
|---|---|
| Nghiên cứu khoa học | Công bố benchmark đầu tiên trên AISHELL-5, paper tại Interspeech/ICASSP |
| Demo sản phẩm | Prototype chạy được, trình bày được cho nhà đầu tư/OEM |
| Startup foundation | SDK/API sẵn sàng tích hợp vào hệ thống infotainment |

---

## 2. Phạm vi dự án (Scope)

### 2.1 Trong phạm vi (In-Scope)
- Pipeline xử lý âm thanh đa kênh (4-channel, 16kHz) từ AISHELL-5
- Module Speech Separation (Asteroid SepFormer/ConvTasNet)
- Module ASR (OpenAI Whisper, tiếng Trung Mandarin)
- Module Speaker Role Classification (ECAPA-TDNN + rule-based)
- Module Intent Engine (rule-based keyword mapping, ~15 intent)
- Demo UI bằng Streamlit (phát audio mẫu, hiển thị transcript real-time)
- Bộ đánh giá: WER, cpWER, Speaker Accuracy, Latency
- Báo cáo nghiên cứu dạng paper + bộ code có thể tái sản xuất

### 2.2 Ngoài phạm vi (Out-of-Scope)
- Thu thập dữ liệu mới (chỉ dùng AISHELL-5 công khai)
- Fine-tune model từ đầu (chỉ dùng pretrained, fine-tune nhẹ nếu cần)
- Tích hợp phần cứng thực (hardware ECU, CAN bus)
- Hỗ trợ ngôn ngữ ngoài tiếng Trung Mandarin ở giai đoạn 1

---

## 3. Người dùng và các bên liên quan (Stakeholders)

| Vai trò | Mô tả | Nhu cầu chính |
|---|---|---|
| **AI Researcher** | Người xây dựng & đánh giá | Pipeline hoạt động đúng, metrics rõ ràng |
| **Product Manager** | Định hướng sản phẩm | Demo thuyết phục, roadmap rõ |
| **OEM / Hãng xe** | Khách hàng tiềm năng (VinFast, Toyota, BYD) | SDK/API tích hợp được, độ trễ thấp |
| **Nhà đầu tư / Startup** | Cần pitch demo | Wow-factor, số liệu cải thiện rõ |
| **Tài xế (end-user)** | Người dùng thực | Hệ thống hiểu đúng lệnh của mình, không nhầm với hành khách |

---

## 4. Yêu cầu chức năng (Functional Requirements)

### FR-01: Xử lý đầu vào âm thanh đa kênh
- **Mô tả:** Hệ thống nhận file `.wav` 4 kênh (shape `[4, T]`, 16kHz) từ AISHELL-5
- **Hành vi:** Tải đúng định dạng bằng `torchaudio`, hỗ trợ streaming theo chunk 1–2 giây
- **Ưu tiên:** Must-have

### FR-02: Tách nguồn giọng nói (Speech Separation)
- **Mô tả:** Tách tín hiệu đa người thành N track đơn kênh (N = số speaker, 2–4)
- **Công nghệ:** Asteroid SepFormer (`speechbrain/sepformer-wsj02mix`) hoặc ConvTasNet
- **Hành vi:** Output N file `.wav` đơn kênh, mỗi file chứa giọng một người
- **Chỉ tiêu:** SI-SNRi cải thiện so với mixture input
- **Ưu tiên:** Must-have

### FR-03: Nhận dạng giọng nói (ASR)
- **Mô tả:** Chuyển mỗi audio track thành văn bản tiếng Trung Mandarin
- **Công nghệ:** OpenAI Whisper (HuggingFace Transformers, `language="zh"`)
- **Hành vi:** Trả về transcript dạng string, hỗ trợ chunked inference
- **Chỉ tiêu:** WER < 35% trên Eval1 (single-channel baseline), cải thiện khi kết hợp separation
- **Ưu tiên:** Must-have

### FR-04: Phân loại vai trò người nói (Speaker Role Classification)
- **Mô tả:** Xác định speaker nào là Driver, Passenger 1, Passenger 2, ...
- **Công nghệ MVP:** Rule-based theo năng lượng kênh (kênh micro gần ghế lái)
- **Công nghệ nâng cao:** ECAPA-TDNN embedding (SpeechBrain pretrained)
- **Hành vi:** Gán nhãn `Driver` / `Passenger_N` cho mỗi track
- **Chỉ tiêu:** Speaker Attribution Accuracy > 80%
- **Ưu tiên:** Must-have (rule-based MVP), Should-have (ECAPA)

### FR-05: Hiểu ý định lệnh (Intent Engine)
- **Mô tả:** Mapping transcript → intent category và thực thể (entity)
- **Scope intent:** Climate control, Media control, Navigation, Window/Sunroof, Phone call, General query
- **Hành vi:** Trả về JSON `{"intent": "climate_control", "action": "decrease", "value": "24°C", "speaker": "Driver"}`
- **Ưu tiên:** Should-have (rule-based), Could-have (LLM-based)

### FR-06: Giao diện demo (Streamlit UI)
- **Mô tả:** Web app demo trực quan cho pitch và nghiên cứu
- **Thành phần:**
  - Dropdown chọn audio sample từ AISHELL-5 dev/eval
  - Player phát audio gốc (4-channel mixed)
  - Bảng real-time: `Speaker | Transcript | Intent | Confidence`
  - Waveform visualization (optional)
  - Status bar: chunk đang xử lý
  - Nút Reset / Start
- **Ưu tiên:** Must-have

### FR-07: Đánh giá và metrics
- **Mô tả:** Script tự động tính toàn bộ metrics trên dev/eval
- **Metrics:**
  - **WER** (Word Error Rate per speaker)
  - **cpWER** (Concatenated Permutation WER – multi-speaker)
  - **Speaker Attribution Accuracy**
  - **End-to-end Latency** (ms/chunk)
- **Output:** CSV + bảng LaTeX cho paper
- **Ưu tiên:** Must-have

### FR-08: Reproducibility Package
- **Mô tả:** Đảm bảo bất kỳ ai cũng có thể chạy lại toàn bộ hệ thống
- **Bao gồm:** `requirements.txt`, `Dockerfile`, `README.md`, script `run_demo.sh`, `evaluate.py`
- **Ưu tiên:** Must-have

---

## 5. Yêu cầu phi chức năng (Non-Functional Requirements)

| ID | Yêu cầu | Chỉ tiêu |
|---|---|---|
| NFR-01 | **Latency** | Pipeline xử lý 1 chunk 2s trong < 3s (trên GPU T4) |
| NFR-02 | **Accuracy** | cpWER < 35% trên Eval1 với pipeline đề xuất |
| NFR-03 | **Scalability** | Hỗ trợ 2–4 speakers không cần thay đổi code |
| NFR-04 | **Portability** | Chạy được trên Docker, GPU NVIDIA (CUDA 11+) |
| NFR-05 | **Reproducibility** | Chạy đúng kết quả với lệnh `bash run_eval.sh` |
| NFR-06 | **License compliance** | AISHELL-5 CC BY-SA 4.0 – ghi rõ trong doc |

---

## 6. Kiến trúc hệ thống (System Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                    IN-CAR MULTI-SPEAKER ASR                     │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  INPUT LAYER │    │  SEPARATION LAYER│    │   ASR LAYER   │  │
│  │              │    │                  │    │               │  │
│  │  4-ch WAV    │───▶│  Asteroid        │───▶│  Whisper      │  │
│  │  [4, T]      │    │  SepFormer /     │    │  (zh-CN)      │  │
│  │  16kHz       │    │  ConvTasNet      │    │               │  │
│  │  Chunk: 2s   │    │  Output: N×[1,T] │    │  Output: text │  │
│  └──────────────┘    └──────────────────┘    └───────────────┘  │
│                                                      │           │
│  ┌──────────────────────┐    ┌────────────────────────────────┐  │
│  │  INTENT ENGINE       │    │  SPEAKER ROLE CLASSIFIER       │  │
│  │                      │    │                                │  │
│  │  Rule-based mapping  │◀───│  ECAPA-TDNN / Energy-based     │  │
│  │  JSON intent output  │    │  Driver / Passenger_N label    │  │
│  └──────────────────────┘    └────────────────────────────────┘  │
│                │                                                  │
│  ┌─────────────▼──────────────────────────────────────────────┐  │
│  │                    STREAMLIT DEMO UI                       │  │
│  │   Audio Player │ Transcript Table │ Intent Display         │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Stack công nghệ

| Layer | Công nghệ | Version |
|---|---|---|
| Speech Separation | Asteroid / SpeechBrain SepFormer | ≥ 0.4 |
| ASR | OpenAI Whisper (HuggingFace) | `large-v3` / `small` |
| Speaker Embedding | SpeechBrain ECAPA-TDNN | pretrained |
| Audio Processing | torchaudio, librosa | ≥ 2.0 / ≥ 0.10 |
| Backend | Python 3.10, FastAPI (optional) | 3.10+ |
| UI | Streamlit | ≥ 1.30 |
| Evaluation | jiwer (WER), custom cpWER | latest |
| Infrastructure | Docker, NVIDIA CUDA 11.8+ | — |
| Data | AISHELL-5 (OpenSLR #159) | CC BY-SA 4.0 |

---

## 7. Dataset chi tiết (AISHELL-5)

| Split | Kích thước | Nội dung | Sử dụng trong dự án |
|---|---|---|---|
| `train` | 54 GB | Far-field + near-field (headset) | Optional: fine-tune nhẹ |
| `dev` | 1.9 GB | Far-field 4-ch, đa speaker | **Development & tuning** |
| `eval1` | 1.7 GB | Evaluation set 1 | **Benchmark chính** |
| `eval2` | 1.8 GB | Evaluation set 2 | Benchmark phụ |
| `noise` | 13 GB | Ambient car noise | Data augmentation |

**Download commands:**
```bash
wget https://openslr.magicdatatech.com/resources/159/dev.tar.gz
wget https://openslr.magicdatatech.com/resources/159/eval1.tar.gz
```

---

## 8. Metrics & Tiêu chí thành công (Success Criteria)

| Metric | Baseline (Single-ch) | Target (Pipeline) | Upper-bound (Near-field) |
|---|---|---|---|
| WER (%) | ~45 | < 30 | ~18 |
| cpWER (%) | ~48 | < 33 | ~19 |
| Speaker Acc. (%) | ~62 | > 80 | ~92 |
| Latency (s/chunk) | 0.3 | < 1.0 | — |

---

## 9. Rủi ro và giải pháp (Risk Register)

| Rủi ro | Xác suất | Mức độ | Giải pháp |
|---|---|---|---|
| Separation kém (nhiều nguồn chồng lấp) | Cao | Cao | Bắt đầu 2-speaker, fallback direct ASR nếu SI-SNRi < 3dB |
| Whisper WER cao với tiếng Trung có noise | Trung bình | Cao | Thêm tiền xử lý denoising; fine-tune trên AISHELL-1 nếu cần |
| Latency vượt ngưỡng demo | Trung bình | Trung bình | Dùng Whisper `small`, chunked streaming, giảm beam size |
| GPU không khả dụng | Thấp | Cao | Chuẩn bị cả CPU fallback (Whisper `tiny`); dùng Colab |
| AISHELL-5 download chậm | Thấp | Thấp | Chỉ tải dev/eval (~5GB), dùng mirror |

---

## 10. Lộ trình 10 tuần (Sprint Roadmap)

| Sprint | Tuần | Tên Sprint | Mục tiêu chính |
|---|---|---|---|
| S1 | 1 | Environment & Data | Setup môi trường, tải và khám phá data |
| S2 | 2 | Separation Baseline | Chạy speech separation đầu tiên |
| S3 | 3 | ASR Integration | Tích hợp Whisper, đo WER baseline |
| S4 | 4 | End-to-End Pipeline | Ghép toàn bộ pipeline, test end-to-end |
| S5 | 5 | Speaker Classification | Phân loại Driver/Passenger |
| S6 | 6 | Intent Engine & Demo UI | Rule-based intent + Streamlit UI |
| S7 | 7 | Evaluation & Metrics | Đo đầy đủ metrics trên eval set |
| S8 | 8 | Ablation & Optimization | Ablation study, tối ưu latency |
| S9 | 9 | Report Writing | Viết báo cáo khoa học |
| S10 | 10 | Polish & Delivery | Hoàn thiện, đóng gói, demo sẵn sàng |

---

## 11. Định nghĩa hoàn thành (Definition of Done – Global)

Một task được coi là **Done** khi:
1. ✅ Code được review và merge vào nhánh chính
2. ✅ Có unit test hoặc notebook minh họa hoạt động đúng
3. ✅ Kết quả được log/ghi vào file output (CSV, JSON, hoặc notebook)
4. ✅ Không có lỗi runtime khi chạy trên môi trường Docker chuẩn
5. ✅ Có tài liệu mô tả ngắn (docstring hoặc README section)

---

## 12. Phụ lục: Intent Categories (MVP)

| Intent | Ví dụ câu lệnh | Output JSON |
|---|---|---|
| `climate_increase` | "tăng nhiệt độ lên" | `{"intent":"climate","action":"increase"}` |
| `climate_decrease` | "giảm điều hòa xuống 22 độ" | `{"intent":"climate","action":"decrease","value":22}` |
| `media_play` | "mở nhạc jazz" | `{"intent":"media","action":"play","genre":"jazz"}` |
| `media_pause` | "tắt nhạc đi" | `{"intent":"media","action":"pause"}` |
| `media_volume_up` | "tăng âm lượng" | `{"intent":"media","action":"vol_up"}` |
| `media_volume_down` | "nhỏ âm lượng xuống" | `{"intent":"media","action":"vol_down"}` |
| `navigation_go` | "đi về nhà" | `{"intent":"nav","action":"start","dest":"home"}` |
| `navigation_cancel` | "hủy dẫn đường" | `{"intent":"nav","action":"cancel"}` |
| `window_open` | "mở cửa sổ" | `{"intent":"window","action":"open"}` |
| `window_close` | "đóng cửa" | `{"intent":"window","action":"close"}` |
| `phone_call` | "gọi cho mẹ" | `{"intent":"phone","action":"call","contact":"mẹ"}` |
| `unknown` | (không khớp) | `{"intent":"unknown"}` |
