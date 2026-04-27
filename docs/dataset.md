# AISHELL-5: Dataset và Benchmark cho In-Car Multi-Speaker ASR

**Technical Report**

## 1. Giới thiệu

Bài báo *“AISHELL-5: The First Open-Source In-Car Multi-Channel Multi-Speaker Speech Dataset for Automatic Speech Diarization and Recognition”* giới thiệu một benchmark mới cho Automatic Speech Recognition (ASR) trong môi trường xe hơi thực tế — một setting đặc biệt khó do:

* Nhiều người nói đồng thời (multi-speaker overlap)
* Microphone xa (far-field)
* Nhiễu môi trường phức tạp (road noise, engine, wind, etc.)
* Đa kênh (multi-channel spatial audio)

AISHELL-5 là dataset đầu tiên công khai giải quyết **đồng thời 3 bài toán**:

* Speech separation
* Speaker diarization
* Automatic speech recognition (ASR)

---

## 2. Dataset Design

### 2.1 Tổng quan cấu trúc

Dataset gồm **2 thành phần chính**:

#### (1) Multi-channel speech corpus

* **>100 giờ** dữ liệu thoại thực tế ([Hugging Face][1])
* Thu trong **xe điện thật** (real driving environment) ([Hugging Face][1])
* **>60 kịch bản lái xe** khác nhau ([Hugging Face][1])

#### (2) Environmental noise corpus

* **~40 giờ** noise recording ([Hugging Face][1])
* Dùng cho:

  * Data simulation
  * Augmentation
  * Robustness evaluation

---

### 2.2 Thiết lập microphone (key contribution)

AISHELL-5 cung cấp cấu hình recording hiếm thấy:

| Loại tín hiệu | Mô tả                        |
| ------------- | ---------------------------- |
| Far-field     | 4 microphone đặt trên cửa xe |
| Near-field    | Micro headset từng speaker   |

➡️ Ý nghĩa:

* Cho phép **supervised source separation** (clean ground truth)
* Benchmark realistic hơn so với dataset chỉ có far-field

---

### 2.3 Multi-speaker setting

* Nhiều người trong xe nói cùng lúc
* Có:

  * Overlap speech
  * Turn-taking tự nhiên
* Đây là **real conversational data**, không phải scripted speech

➡️ Khác biệt lớn so với:

* AISHELL-1/2 (single speaker, clean speech)
* Librispeech (read speech, không nhiễu)

---

### 2.4 Độ đa dạng dữ liệu

Dataset bao phủ:

* Driving conditions:

  * City / highway
  * Idle / moving
* Noise types:

  * Engine
  * Road friction
  * Wind
  * Passenger movement

➡️ Điều này tạo ra **distribution shift mạnh**, thách thức mô hình ASR.

---

### 2.5 Annotation & ground truth

Dù paper không chi tiết đầy đủ trong abstract, nhưng có thể suy ra:

* Transcript cho từng speaker
* Alignment giữa:

  * multi-channel signal
  * speaker identity
* Có thể hỗ trợ:

  * Diarization labels
  * Source separation supervision

---

## 3. Benchmark Tasks

AISHELL-5 không chỉ là dataset, mà còn là **benchmark đa nhiệm**.

### 3.1 Task decomposition

#### Task 1: Speech Separation

* Input: multi-channel far-field audio
* Output: clean signal per speaker

#### Task 2: Speaker Diarization

* Xác định:

  * “ai nói khi nào”

#### Task 3: ASR (Speech Recognition)

* Transcribe nội dung từng speaker

---

### 3.2 Pipeline evaluation setting

Benchmark giả định pipeline:

```
Multi-channel audio
        ↓
Speech separation
        ↓
Speaker-wise signals
        ↓
ASR
        ↓
Transcripts
```

➡️ Quan trọng:

* Error propagation giữa các stage
* Không thể optimize từng task độc lập

---

### 3.3 Evaluation challenges

AISHELL-5 nhắm đến các khó khăn thực tế:

1. **Cocktail party problem**
2. **Far-field degradation**
3. **Noise robustness**
4. **Multi-channel spatial modeling**
5. **Real-time constraints (implicit)**

---

## 4. Baseline System

### 4.1 Kiến trúc tổng thể

Baseline được thiết kế theo pipeline 2-stage:

#### Stage 1: Speech Frontend

* Source separation model
* Input: multi-channel signals
* Output: estimated clean speech per speaker

#### Stage 2: ASR Backend

* Nhận từng stream riêng biệt
* Transcribe nội dung

([Hugging Face][1])

---

### 4.2 Speech Separation Module

Vai trò:

* Tách tín hiệu từng người nói từ mixture

Đặc điểm:

* Khai thác:

  * Spatial cues (multi-channel)
  * Temporal cues

➡️ Đây là phần **critical bottleneck** của hệ thống.

---

### 4.3 ASR Module

* Nhận input đã được “clean”
* Có thể dùng:

  * Conventional ASR
  * End-to-end models (Transformer / Conformer)

Tuy nhiên:

* Performance phụ thuộc mạnh vào quality của separation

---

### 4.4 End-to-end behavior

Pipeline này thể hiện:

* Không phải joint optimization
* Dễ bị:

  * Error compounding
  * Latency tăng

➡️ Đây là baseline "classical pipeline", không phải SOTA end-to-end multi-task model.

---

## 5. Experimental Findings (Key Insights)

### 5.1 Main observation

> Các mô hình ASR mainstream gặp khó khăn đáng kể trên AISHELL-5 ([Hugging Face][1])

---

### 5.2 Nguyên nhân chính

#### (1) Domain mismatch

* Models trained on:

  * clean speech
* Dataset:

  * noisy, overlapping, far-field

#### (2) Separation errors

* Imperfect disentanglement → ASR degradation

#### (3) Multi-speaker interference

* Overlap speech → WER tăng mạnh

---

### 5.3 Implicit benchmark difficulty

AISHELL-5 tạo ra benchmark với:

* High noise entropy
* High speaker overlap
* Realistic acoustic condition

➡️ Đây là benchmark **harder than traditional ASR benchmarks**.

---

## 6. So sánh với benchmark trước đó

| Benchmark   | Đặc điểm                               | Hạn chế          |
| ----------- | -------------------------------------- | ---------------- |
| LibriSpeech | Clean, read speech                     | Không realistic  |
| AISHELL-1   | Mandarin clean speech                  | Single speaker   |
| CHiME       | Noisy speech                           | Ít multi-speaker |
| AISHELL-5   | Multi-channel + multi-speaker + in-car | —                |

➡️ AISHELL-5 lấp gap giữa:

* Lab datasets
* Real-world deployment

---

## 7. Ý nghĩa nghiên cứu

### 7.1 Đóng góp chính

* Dataset đầu tiên:

  * In-car
  * Multi-channel
  * Multi-speaker
* Benchmark unified:

  * Separation + diarization + ASR
* Baseline reproducible system

---

### 7.2 Impact

AISHELL-5 mở ra hướng nghiên cứu:

* Joint modeling:

  * separation + ASR
* Robust ASR:

  * under domain shift
* Multi-modal spatial audio modeling

---

## 8. Hạn chế và hướng mở

### 8.1 Hạn chế

* Pipeline baseline chưa end-to-end
* Chưa tối ưu:

  * latency
  * real-time deployment
* Dataset domain:

  * chỉ in-car (không general toàn bộ acoustic space)

---

### 8.2 Future directions

* End-to-end multi-task learning
* Neural beamforming + ASR integration
* Self-supervised pretraining với multi-channel audio
* Domain adaptation / simulation

---

## 9. Kết luận

AISHELL-5 thiết lập một **benchmark thực tế và khó** cho ASR trong môi trường xe hơi, với các đặc điểm nổi bật:

* Multi-channel spatial audio
* Multi-speaker overlap
* Real-world noise

Baseline pipeline cho thấy:

* Performance còn xa mức deployable
* Cần phương pháp mới vượt qua:

  * separation bottleneck
  * domain mismatch

➡️ Dataset này đóng vai trò như:

> “stress test” cho thế hệ ASR models tiếp theo

---

## 9. Cấu trúc OpenSLR #159 (AISHELL-5) & cách dùng trong repo

Tài liệu này tóm tắt bối cảnh nghiên cứu; phần dưới mô tả **chính xác** cách sử dụng các split **dev, eval1, eval2, noise** trong codebase (`scripts/download_data.sh`, `scripts/materialize_aishell5_flat.py`, `evaluate.py`).

### 9.1 Các bộ con (public, OpenSLR)

| Split  | Mục đích chính | WER/cpWER trong `evaluate.py`? |
|--------|----------------|---------------------------------|
| **dev**  | phát triển / hợp lệ hóa pipeline | Có, `--split dev` (cần `data/dev/wav/`, `data/dev/text/`) |
| **eval1** | benchmark chính (track 1)      | Có, `--split eval1` (mặc định)  |
| **eval2** | benchmark / track 2            | Có, `--split eval2` (sau khi tải + materialize) |
| **noise** | ồn xe ~40h, **không** lời nói   | **Không** dùng trực tiếp cho WER — mô phỏng / tăng cường dữ liệu / thử nghiệm ASR+noise. Không thêm `noise` vào `evaluate.py` mặc định. |

**train** (~51GB): dùng huấn luyện / near-field, không cần cho benchmark eval trong repo mặc định.

### 9.2 Cây thư mục từ server (sau khi `tar -xzf`)

Bài release chuẩn đặt **mỗi kênh far-field thành một file mono** theo tên, ví dụ:

- Một session (thư mục số) chứa `DX01C01.wav` … `DX04C01.wav` = bốn micro cửa; `DA*.wav` = near-field headset (có ở train/dev, không bắt buộc cho far-field ASR thuần).  
- Nhãn câu/loa trong **Praat TextGrid** (theo từng kênh far-field, ví dụ `DX01C01.TextGrid`).

Pipeline và `AISHELL5Loader` **không** đọc trực tiếp cây này; cần bước chuẩn hóa (một 4 kênh `.wav` + `text/SPK*.txt` mỗi session).

### 9.3 Định dạng chuẩn cho loader / đánh giá (trong repo)

Sau chạy `python scripts/materialize_aishell5_flat.py --source <Dev|Eval1|Eval2> --dest data/<dev|eval1|eval2> --label ...`:

- `data/<split>/wav/*.wav` — **4 kênh, một file mỗi session** (stack từ DX01–04C01)  
- `data/<split>/text/<id>.txt` — mỗi dòng dạng `SPK1: ...`, `SPK2: ...` (gộp toàn bộ lời theo từng loa từ TextGrid)

Khi đó `AISHELL5Loader` và `python evaluate.py --split ...` mới tính CER / cpWER đúng nghĩa.

### 9.4 Đánh giá đa nói: cpWER + Hungarian

Các bản ghi tách nguồn ra **thứ tự track 0,1,…** không trùng với thứ tự `SPK1, SPK2` trong tham chiếu. Báo cáo chính dùng **cpWER** (gán tối ưu theo thuật toán Hungary). Ở `evaluate.py`, WER từng dòng cũng dùng **cùng một gán** tối ưu (`assign_references_to_hypotheses`) để số trên CSV và cpWER thống nhất.

### 9.5 Noise (split riêng)

- Chứa tín hiệu **môi trường xe, không lời**, dùng để mix, simulation, stress-test robustness.  
- Không tạo `references` ASR: **không** dùng làm tập WER tĩnh trong `evaluate.py`. Khi cần, viết thí nghiệm riêng (mix với hỗn hợp 4 kênh rồi gọi pipeline).
