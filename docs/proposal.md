# Tóm tắt (Executive Summary)

- **Mục tiêu:** Xây dựng một hệ thống AI hoàn chỉnh cho **nhận dạng và hiểu giọng nói đa người trong ô tô**, sử dụng dataset AISHELL-5. Hệ thống sẽ tách nguồn (separation) và nhận dạng (ASR) giọng nói từ micro nhiều kênh trong xe, xác định vai trò người nói (tài xế/hành khách), và hiểu lệnh điều khiển (intent).
- **Động lực:** Hệ thống trợ lý giọng nói trong ô tô hiện tại *không tốt khi nhiều người cùng nói lẫn nhau* và *chưa tối ưu với môi trường ồn và nhiều micro*. AISHELL-5 mở ra **benchmark mới** cho ASR thực tế trong xe hơi. Khi làm việc với dataset này, chúng ta đóng góp vào **hướng nghiên cứu mới** (ASR trong môi trường thực, multi-speaker, multi-kênh) và có thể phát triển một prototype (demo) hấp dẫn cho startup.
- **Phương án đề xuất:** Dự án triển khai **giai đoạn 8–12 tuần**. Xây dựng pipeline đa kênh-end-to-end: **(1) Tách giọng nói** (multi-channel separation); **(2) Nhận dạng ASR** mỗi kênh; **(3) Xác định vai trò người nói** (driver/passenger); **(4) Hiểu ngữ nghĩa lệnh** và trả về phản hồi. Tất cả modules dùng **mô hình pretrained công khai** (Asteroid cho separation, Whisper cho ASR, SpeechBrain-ECAPA cho nhận diện người nói) để **không cần thu thập thêm dữ liệu**. 
- **Demo thực tế:** Phát lại audio đa người của AISHELL-5 (từ tập dev/eval), hệ thống tách được từng người và transcribe đúng lệnh. Ví dụ: input 2 người nói chồng lệnh “mở nhạc” và “giảm nhiệt độ”; output hiển thị rõ “[Tài xế]: giảm nhiệt độ xuống 24 độ” và “[Hành khách]: mở nhạc rock” (và ưu tiên driver). Demo này “mang tính khéo mắt” (mời nhà đầu tư/chuyên gia xem hệ thống hiểu đúng trong tình huống thực tế).
- **Đóng góp khoa học:** Báo cáo sẽ nêu benchmark mới (WER, cpCER, speaker accuracy) trên AISHELL-5, so sánh baseline (single-channel ASR) với pipeline multi-channel+separation của đề xuất. Khuyến khích kết quả: cải thiện đáng kể WER trong tiếng ồn/đa người. System hoàn chỉnh cũng có thể công bố (paper describing pipeline, results, demo).

# 1. Mục tiêu và Động lực  
- **Nhu cầu thực tế:** Thiết bị âm thanh trong xe (loại Android Auto, Apple CarPlay, hệ thống infotainment) ngày càng phổ biến. Tuy nhiên, **ASR trong ô tô đang yếu** khi gặp nhiễu (tiếng máy, đường xá) và nhiều người nói cùng lúc. Trợ lý giọng nói truyền thống như Siri/Alexa/Echo không được thiết kế cho tình huống “2–4 người nói trò chuyện trong xe”.  
- **Ý tưởng dự án:** Xây dựng **hệ thống nhận dạng giọng nói đa người trong xe (In-Car Multi-Speaker ASR)**. Hệ thống sẽ *lắng nghe* qua nhiều micro (4 cửa xe), tách tiếng của từng người, chuyển thành văn bản và hiểu lệnh (intent) của mỗi người. Quan trọng là **không cần data mới**: chỉ dùng dataset AISHELL-5 công khai, tận dụng mô hình pretrained.
- **Giá trị nghiên cứu:** Mặc dù AISHELL-5 vừa ra mắt, **chưa có nhiều công trình follow-up** (xem kết quả research) nên bạn “đi đầu” trong hướng này. Hướng nghiên cứu chính: ASR đa người trong môi trường thật, multi-kênh, kết hợp separation+ASR. Các vấn đề chính: *speech separation*, *multi-channel ASR*, *speaker labeling*, *tương tác ngôn ngữ/ngữ cảnh*. Đóng góp của ta là: 1) pipeline hoàn chỉnh và benchmark trên AISHELL-5, 2) các thí nghiệm minh họa hiệu quả multi-channel vs single-channel, 3) đưa ra metric đặc biệt (cpCER…) cho multi-speaker.
- **Giá trị sản phẩm (startup):** Là SDK/giải pháp phục vụ các hãng xe (OEM) và hệ thống thoại thông minh ô tô. Ví dụ: VINFAST, TOYOTA, Ford cần trợ lý hiểu nhiều người trong cabin. Mô hình kinh doanh: cung cấp API/SDK speech-intelligence cho xe (trên cloud hoặc on-device). Khẩu hiệu demo: **“Xe của bạn hiểu khi nhiều người nói cùng lúc”**.

# 2. State-of-the-Art ngắn liên quan AISHELL-5  
- **Dataset AISHELL-5 (In-Car, Multi-Kênh, Multi-Speaker):** AISHELL-5 là dataset Mandarin đa-kênh ghi trong ô tô (4 micro cửa xe, 165 người, 100+ giờ speech). Mỗi bản ghi có 2–4 người trò chuyện tự nhiên. Ngoài ra có 40 giờ **dữ liệu noise ô tô thực tế** (xe chạy, máy lạnh) để giả lập noise. Đây là *dataset công khai đầu tiên* cho ASR đa người trong xe.  
- **Baseline AISHELL-5:** Bài báo gốc (Interspeech 2025) cung cấp hệ thống baseline: **tách nguồn** (speech separation) rồi **ASR** riêng cho mỗi người. Thí nghiệm cho thấy các ASR thông thường khi áp dụng lên AISHELL-5 cho WER rất tệ, đặc biệt với tín hiệu far-field và nhiều người.  
- **Công nghệ liên quan:** Một số công nghệ tiên phong có thể áp dụng:    
  - *Speech Separation:* Mô hình Conv-TasNet/SepFormer cho tách nguồn on short (kênh đơn) hay SpatialNet/Beamforming cho multi-kênh. Toolkit như Asteroid cung cấp sẵn các mô hình tách âm.  
  - *ASR hiện đại:* Whisper (OpenAI) là mô hình pretrained đa ngôn ngữ (99 languages) đạt state-of-art cho ASR. Dù Whisper chủ yếu train tiếng Anh, nó hỗ trợ tiếng Trung tốt do pretrain đa ngữ (5M+ hours). Ngoài ra, Wav2Vec2, Wenet, Kaldi đều có thể thử nghiệm.  
  - *Speaker Embedding:* SpeechBrain có pretrained ECAPA-TDNN cho speaker recognition, có thể dùng để xác định ai đang nói dựa trên embedding giọng nói.  
  - *Voice Assistant in Car:* Các sản phẩm công nghiệp (Nghe Google Duplex, Amazon Alexa Auto) đang nhắm tới lệnh thoại trong ô tô, nhưng hầu như chỉ xử lý single-user. Hướng mới là nhận dạng đa người, xác định vai trò (driver/hành khách) để ưu tiên.
- **Khoảng trống hiện tại:** AISHELL-5 mới (2025), chưa nhiều bài nghiên cứu sử dụng. Nhiều khả năng đây sẽ trở thành benchmark chuẩn cho ASR đa người thực tế. Hiện chưa có công trình công bố pipeline toàn diện hay ứng dụng cụ thể trên dataset này, nên bạn có cơ hội mở đường.
  
# 3. Dataset AISHELL-5 (tải & mẫu)  
- **Link download chính thức:** AISHELL-5 được host trên OpenSLR: [openslr.org/159](https://www.openslr.org/159/). Tại đây có link download:  
  - Train (54GB), Dev (1.9GB), Eval1 (1.7GB), Eval2 (1.8GB), Noise (13GB).  
  - Ví dụ lệnh tải (UNIX):  
    ```bash
    wget https://www.openslr.org/resources/159/Dev.tar.gz
    wget https://www.openslr.org/resources/159/Eval1.tar.gz
    ```  
  - Giải nén: `tar -xzf dev.tar.gz`. Sau khi giải nén, cấu trúc:  
    - `dev/wav/`: chứa file .wav đa kênh (4-channel) cho mỗi recording.  
    - `dev/text/`: file transcript (multi-speaker) tương ứng.  
    - Tương tự cho `eval1/`, `eval2/`. (Train có thêm kênh near-field headset).  
- **Cấu trúc dữ liệu:** Theo mô tả chính thức, AISHELL-5 gồm:  
  - **Speech (far-field, 4 kênh):** Micro trên bốn cửa xe, ghi âm các cuộc hội thoại tự nhiên (2–4 người). Ghi âm trong >60 kịch bản lái thực (đường cao tốc, nội thành, đỗ xe…).  
  - **Speech (near-field):** Mỗi người nói đeo tai nghe thu âm sạch (chỉ trong train set). Dùng để đối chiếu và huấn luyện.  
  - **Noise:** ~40 giờ tiếng ồn ô tô thực tế (không người nói). Dùng để mix noise hoặc data augmentation.  
  - **Transcripts:** File chuẩn từng speaker (SPK1, SPK2…). Định dạng example:  
    ```
    SPK1: 我们现在去哪里？  
    SPK2: 去公司吧  
    SPK1: 好的
    ```
    (Mỗi dòng có tag speaker). Các tập dev/eval đã có transcripts.  
- **Sample cụ thể:** Ví dụ lấy từ `dev` (giả lập):  
  - Lệnh để trích xuất:  
    ```bash
    wget -O sample_dev.wav https://www.openslr.org/resources/159/dev/wav/dev_0001.wav
    wget -O sample_dev.txt https://www.openslr.org/resources/159/dev/text/dev_0001.txt
    ```  
  - Trong đó `sample_dev.wav` là audio 4 kênh (far-field) của một đoạn hội thoại. `sample_dev.txt` là transcript ví dụ cho từng speaker. Kết hợp hai kênh bất kỳ để tạo dạng overlap (nhiều speaker).  
- **Lưu ý thực tế:** Vì dữ liệu rất lớn, bạn chỉ cần dùng **dev/eval** (~5GB) để phát triển và đánh giá demo. Dev/Eval có đủ tình huống liên quan để minh hoạ. Không cần train model từ đầu; chỉ cần chạy inference trên dev/eval để thử nghiệm, demo.

# 4. Thiết kế hệ thống (Architecture & Pipeline)

 *Hình 1: Kiến trúc pipeline đề xuất (hệ thống xử lý giọng nói đa người trong xe).*  

Hệ thống sẽ có các module chính (từ đầu vào audio đa kênh đến output lệnh). Các bước chi tiết:  

- **(A) Input Audio (Multi-channel):**  Audio 4 kênh (16kHz) từ micro các cửa xe. Ví dụ code: `torchaudio.load("eval1/spk1.wav")` trả về tensor shape `[4, T]`. Hệ thống hoạt động dạng streaming: chia audio thành khung 1–2s để xử lý gần thời gian thực.  
- **(B) Speech Separation (Frontend):** Dùng mô hình **multi-channel source separation** để tách các nguồn giọng nói. Ví dụ: dùng *Asteroid* với mô hình pretrained (Conv-TasNet hoặc SepFormer). Có thể áp dụng *neural beamforming* (ví dụ Acoustic Spatial Filtering). Mục tiêu: đầu ra thành N *audio track* đơn kênh, mỗi track chứa giọng nói của một speaker riêng biệt. (Số N bằng số speaker, 2–4). Asteroid cung cấp toolkit mạnh cho separation. (Đề xuất: dùng model *SepFormer* fine-tuned cho speech, public trên HuggingFace như `speechbrain/sepformer-wsj02mix` hoặc các model noise-robust multi-mic.)  
- **(C) ASR Module:** Với mỗi track tách được, chạy Automatic Speech Recognition. Dùng mô hình **Whisper** (OpenAI) pretrained: hỗ trợ ASR đa ngôn ngữ. Cài đặt đơn giản với transformers (AutoModelForSpeechSeq2Seq) như ví dụ trên HuggingFace. Whisper strong generalization nhờ >5M giờ data huấn luyện, mặc dù dataset AISHELL-5 là tiếng Trung, Whisper có bộ giải mã tiếng Trung built-in. (Các model ASR khác có thể dùng: Wav2Vec2, Espnet, Wenet, nhưng Whisper tiện vì có pipeline sẵn và đã fine-tune mạnh.)  
- **(D) Speaker-Role Classifier:** Sau khi có transcript mỗi track (loại “Speaker1: text”), xác định vai trò (Tài xế hay Hành khách). Triển khai đơn giản: dựa vào **vị trí vật lý micro** hoặc embedding người nói. Ví dụ: driver thường nói vào micro cửa driver (kênh nào năng lượng cao nhất). Hoặc dùng mô hình *Speaker Recognition* (SpeechBrain ECAPA-TDNN) để nhận diện nếu biết trước đặc điểm driver. (MVP: rule-based ưu tiên người ở “lái”). Phân biệt này quan trọng vì lệnh của tài xế được ưu tiên xử lý ngay.  
- **(E) Intent Engine:** Chuyển transcript thành lệnh thực thi. Ban đầu dùng **quy tắc đơn giản**: mapping keyword (thời tiết, âm nhạc, điều hoà…) → chức năng. Ví dụ “mở nhạc/chương trình radio” → điều khiển media, “giảm/ tăng nhiệt độ” → climate control. Nâng cao: dùng LLM nhỏ (GPT, Llama) đã fine-tune gói in-car dialogues để phân tích intent. Tuy nhiên MVP chỉ cần rule-base.  
- **(F) Command Layer:** Tích hợp với hệ thống giả lập xe (hoặc API demo). Ví dụ chỉ cần print ra log: “[Driver] bật điều hòa 24°C”. Hoặc điều khiển nhạc giả định. Phần này phụ, nhằm minh họa end-to-end.  
- **(G) Streaming Design:** Hệ thống hoạt động theo thời gian thực: input audio stream (chunk ~2s), liên tục chạy pipeline, update transcript. Đảm bảo độ trễ (latency) thấp đủ cho demo. Có thể dùng `streamlit-webrtc` để phát trực tiếp audio simulaiton và hiển thị kết quả.  

Các module và công nghệ cụ thể sẽ được tóm tắt trong **Bảng dưới**. Mọi thành phần dùng model pretrained/off-the-shelf: Asteroid, Whisper, SpeechBrain ECAPA, đồ họa Python (FastAPI/Flask + Streamlit). 

| **Module**                | **Công cụ/Tecnology**                                         |
|---------------------------|-------------------------------------------------------------|
| Speech Separation         | Asteroid SepFormer/ConvTasNet (PyTorch)          |
| ASR                       | OpenAI Whisper (HuggingFace Transformers)       |
| Speaker Embedding/Label   | SpeechBrain ECAPA-TDNN (pretrained)              |
| Intent Understanding      | Rule-based (với từ khoá domain cụ thể) hoặc LLM (OpenAI/GPT)   |
| Backend                   | Python (FastAPI/Flask)                                      |
| UI                        | Streamlit + streamlit-webrtc (audio streaming interface)    |
| Data Pipeline             | Librosa/Torchaudio for wav loading, JSON/CSV cho transcripts  |
| Inference & Runtime       | GPU for model inference (prefer GPU cho Whisper)            |
| Containerization          | Docker (pytorch, python 3.10)                                |

Với thiết kế này, trọng tâm nghiên cứu là step (B)+(C): “multi-channel separation + ASR”. Các bước còn lại là engineering tích hợp.  

# 5. User-flow & UI Mockup  
- **Tình huống demo:** Giả sử 2 người trong xe (Tài xế và Hành khách). Cả hai nói lệnh gần cùng lúc. Ví dụ: Tài xế nói “Giảm nhiệt độ xuống 24 độ” và Hành khách nói “Mở nhạc jazz”. Hệ thống sẽ:  
  1. Tách giọng Tài xế và Hành khách ra hai file audio riêng.  
  2. Dịch giọng của mỗi người sang text: “[Driver]: Giảm nhiệt độ xuống 24 độ”, “[Passenger]: Mở nhạc jazz”.  
  3. Hiểu intent: driver ra lệnh điều hoà, passenger ra lệnh mở nhạc.  
  4. Thực thi (tiếp tục chạy máy lạnh, phát nhạc) hoặc đơn giản log output.  
- **Luồng giao diện (UI):** Ứng dụng Streamlit đơn giản có các thành phần:  
  - **Input:** Phát audio mẫu (pre-recorded) từ AISHELL-5 (có nút play để simulate). Có thể chọn track 4-channel.  
  - **Waveform Real-time:** Live plot sóng âm (nếu muốn fancy).  
  - **Transcript/Output:** Bảng 2 cột hiển thị kết quả: cột “Speaker” (Driver/Passenger), cột “Transcript”. Ví dụ:  
    | Speaker    | Transcript                    |
    |------------|-------------------------------|
    | Driver     | giảm nhiệt độ xuống 24 độ     |
    | Passenger  | mở nhạc jazz                 |
  - **Status:** Hiển thị status (e.g. “Đang xử lý chunk #3…”).  
  - **Button Demo:** Bắt đầu/Reset pipeline.  
  
Không cần UI phức tạp, chỉ hiển thị rõ transcript theo từng speaker để thuyết phục. Mẫu Streamlit code có thể như: `st.text_area(label, value=transcript)` cho mỗi người.  

# 6. Kế hoạch triển khai & Lộ trình 6 tuần  

**Lộ trình chi tiết (ví dụ 6 tuần):**

| **Tuần** | **Mục tiêu/Nhiệm vụ**                                                    | **Kết quả (Deliverables)**                               |
|---------|------------------------------------------------------------------------|--------------------------------------------------------|
| 1       | - Tổ chức môi trường phát triển (Docker, Python, GPU) <br> - Download AISHELL-5 dev/eval, đọc thử file audio/transcript (xác nhận data loading) <br> - Thiết lập notebooks/Python để đọc multi-channel audio (torchaudio) | Code load audio mẫu (4-ch) và transcript, notebook demo nhỏ. |
| 2       | - Cài đặt Asteroid & chạy thử mô hình speech separation pretrained (Ví dụ ConvTasNet) trên sample AISHELL-5. <br> - Đánh giá trực quan: microphone 4 kênh -> 2 nguồn output. <br> - Nghiên cứu model Whisper (transformers), test dự đoán transcript trên audio sạch. | Script phân tách nguồn (4ch→2ch) trên 1-2 sample, output wav. Script demo Whisper trên file audio (near-field có sẵn hoặc far-field sạch). |
| 3       | - Tích hợp pipeline: separation + ASR. Cho đầu vào (4-ch) → tách → từng wav → dùng Whisper trả về văn bản. <br> - Xử lý multi-speaker transcripts (đánh dấu SPK, gộp text) <br> - Đánh giá initial WER trên dev/eval (ở mức ý tưởng). | Module pipeline end-to-end cơ bản (không UI). Tính WER ban đầu (baselines). |
| 4       | - Xây dựng speaker-role classifier: ví dụ đơn giản (dựa vào năng lượng kênh gần tài xế). Gắn label “Driver/Passenger”. <br> - Viết rule intent mapping (ví dụ 10 intent cơ bản của demo: nhiệt độ, nhạc, đèn, nhiệt độ). <br> - Xây dựng giao diện Streamlit sơ bộ hiển thị transcript và label. | Prototype hệ thống demo: chọn 1 sample audio, chạy pipeline, Streamlit show kết quả. |
| 5       | - Đánh giá chi tiết: chạy trên toàn tập dev/eval AISHELL-5. Tính **WER**, **cpWER** (multi-speaker WER), **speaker attribution accuracy**. <br> - Viết code tính metric (căn chỉnh output đúng speaker). <br> - Chuẩn bị sơ đồ pipeline (mermaid) và timeline. <br> - Bắt đầu viết báo cáo/proposal: mục tiêu, motivation, state-of-art, design. | Bảng kết quả WER baseline vs pipeline, log số liệu. (Ví dụ: Single-ch WER 35%, multi-ch 25%, v.v.) Report draft sắp xếp ý. |
| 6       | - Hoàn thiện báo cáo (viết dạng bài nghiên cứu/study): bao gồm mục đích, phương pháp, kết quả, thảo luận. <br> - Viết phần kinh doanh & commercialization: khách hàng, mô hình kinh doanh. <br> - Chuẩn bị form trình bày/delivery (slide, demo summary). <br> - Đóng gói code vào repo (README, Dockerfile). | Bản cuối proposal/đề xuất hoàn chỉnh (viết bằng tiếng Việt). Code sạch, có hướng dẫn reproducibility, file Docker if có. Demo streamlit sẵn sàng. |
  
(※ Lưu ý: Có thể chia nhỏ hoặc gộp/nối tuần tuỳ tiến độ, nhưng đảm bảo **mỗi tuần có deliverable** rõ ràng.)

# 7. Kế hoạch đánh giá và viết báo cáo nghiên cứu  

- **Metrics chính:** Tính *Word Error Rate (WER)* của hệ thống ASR (theo tập dev/eval) cho mỗi speaker. Vì nhiều speaker, cần metric *cpWER/cpCER* (concatenated permutation WER) đánh giá tổng thể multi-speaker. Cũng đo **speaker attribution accuracy** (tỉ lệ câu đúng gán đúng speaker). Đo thêm *latency* pipeline (ms per chunk).  
- **Thiết lập thí nghiệm:**  
  - *Baselines:* (1) ASR đơn kênh (chọn kênh 1 làm đại diện) không tách nguồn; (2) Pipeline multi-channel không tách (ghép 4ch rồi ASR); (3) Pipeline tách + ASR (đề xuất). Có thể thêm (4) Thêm bước lọc noise.  
  - *Ablations:* bỏ ý tưởng speaker-label (chỉ ASR) hoặc bỏ dùng multi-channel (dùng single-ch cho separation) để xem hiệu quả.  
  - *Data splits:* Dùng sẵn dev/eval (AISHELL-5) làm validation/test. Có thể chia nhỏ dev thành val/train-dev nếu cần học ngưỡng CTC, nhưng chủ yếu inference.  
- **Kịch bản so sánh:** Ví dụ: WER trên Eval1 khi: (i) dùng file 1mic (driver) → ASR, (ii) dùng multi-ch tách rồi ASR, (iii) giọng gốc near-field (lý tưởng) → ASR (upper bound). So sánh cpWER của mọi trường hợp.  
- **Kết quả kỳ vọng:** Hy vọng **WER giảm rõ** khi dùng pipeline multi-channel+separation so với baseline one-channel. Cố gắng cpWER < 20–30% (tùy độ khó), speaker accuracy > 80%. Tiêu chí thành công: có cải thiện WER đáng kể và hệ thống xử lý tình huống demo ổn định.  
- **Bảng thí nghiệm (ví dụ):**

| **Cấu hình thí nghiệm**         | **WER (%)** | **cpWER (%)** | **Speaker Acc. (%)** | **Latency (s)** | Ghi chú          |
|---------------------------------|-------------|---------------|----------------------|----------------|------------------|
| Single-channel ASR             | 45.2        | 47.5          | 62.3                 | 0.3 (chunk)    | Kênh 1 (driver)  |
| Multi-channel + tách nguồn (ours) | 28.8        | 30.5          | 85.7                 | 0.5 (chunk)    | Conv-TasNet + Whisper |
| Near-field clean (upper-bound) | 18.0        | 19.0          | 92.0                 | 0.2           | Tai nghe (train) |

*Ví dụ giả lập: Kết quả WER chỉ mang tính minh hoạ. Thực tế sẽ đo đầy đủ trên Eval1/Eval2.*

- **Ablation studies:** Thử dùng model khác: thay Whisper bằng Wav2Vec2; hoặc bỏ bước separation (bỏ ConvTasNet). Thể hiện trong bảng để chứng minh mọi phần đem lại giá trị.  
- **Scripts/Commands:** Ghi chú lệnh dùng HuggingFace pipeline (pip install, from_pretrained), Asteroid inference. Bảo đảm mọi thí nghiệm có thể tái sản xuất qua script (ví dụ `run_eval.sh`).  
- **Đảm bảo tái sản xuất:** Repos sẽ có Dockerfile/builder, ghi rõ cấu hình (Python 3.10, PyTorch, Transformers, SpeechBrain, Asteroid). Ghi README hướng dẫn chi tiết (environment, cách chạy). Nếu được, code công khai (GitHub/Gitlab).

# 8. Triển khai tái sản xuất (Reproducibility)

- **Mã nguồn:** Tất cả code (data loading, pipeline, eval) cần công bố kèm báo cáo để người khác chạy lại. Tốt nhất là repo GitHub công khai. Bao gồm `requirements.txt` (hoặc `environment.yml`).  
- **Môi trường:** Docker image (nếu có) với GPU-enabled PyTorch, CUDA. Hoặc cung cấp `conda` env file. Ghi rõ card GPU đề xuất (nVIDIA T4/RTX3000 trở lên).  
- **Dữ liệu:** Link OpenSLR và script tải mẫu (dev/eval). Lưu ý license CC BY-SA 4.0, cần ghi trong doc.  
- **Chạy lại:** Hướng dẫn chạy pipeline demo `python app.py`, chạy đánh giá `python evaluate.py`. Mọi hyperparameters (threshold separation, beam size ASR) cần cố định để dễ so sánh.  
- **Tính mở rộng:** Mô hình training riêng (nếu có) nên tối thiểu, vì tiêu chí “không thu data mới”. Chủ yếu dùng pretrained.  

# 9. Rủi ro & Giải pháp

- **Khó khăn separation:** Mô hình đa người, đa kênh là bài toán khó; separation có thể không hoàn hảo. Giải pháp: Bắt đầu với model đơn giản (chỉ 2 speaker) trên sample clean trước. Thử các model pretrained, đánh đổi performance vs latency. Tích hợp feedback: nếu separation thất bại (WER cao), thử fallback direct ASR.  
- **Performance ASR:** Whisper có thể cần điều chỉnh để hiểu tiếng Trung giọng Trung Quốc. Có thể cài đặt `language="Chinese"` trong pipeline. Nếu performance kém, cân nhắc fine-tune Whisper nhỏ trên AISHELL-1 (clean Mandarin) trước. Nhưng đây là option sau.  
- **Latency:** Pipeline (separation + Whisper) nặng, có thể trễ. Phân khúc thời gian thực (chunk 2s) và model light (Whisper small) để demo nhanh. Nâng cao: streaming inference (Whisper chunked) hoặc giảm frame rate.   
- **UI/Streamlit:** Streamlit có thể không hỗ trợ mermaid trực tiếp, nhưng UI đơn giản là text và bảng. Không phụ thuộc vào hình vẽ mermaid thực.  
- **Tài nguyên:** Cần GPU để inference Whisper (especially large). Giải pháp: dùng model base/small, chỉ demo vài file. Đảm bảo môi trường test có GPU (như Google Colab, AWS).

# 10. Go-to-Market & Kinh doanh  

- **Khách hàng tiềm năng:**  
  - **Hãng xe/Ô tô (OEM):** Những hãng đang phát triển hệ thống infotainment (VinFast, VinAI Auto, Toyota, Ford, BYD…). Họ cần giải pháp voice assistant nâng cao cho xe hơi, đặc biệt với cabin có nhiều người (đặc biệt taxi, ride-share).  
  - **Công ty công nghệ ô tô:** Ví dụ Parrot, Alpine (audio car systems), Bosch, Continental… có thể tích hợp assistant đa người.  
  - **Dịch vụ call center / dịch vụ karaoke ô tô:** Có thể sử dụng tách giọng tài xế/hành khách để phục vụ khác (ít nhất là demo công nghệ).  
- **Mô hình kinh doanh:**  
  - **Bán SDK/Phần mềm:** Bán license gói phần mềm (SDK hoặc API) để tích hợp vào hệ thống giọng nói của xe. Cung cấp kèm dịch vụ cloud processing hoặc on-device solutions.  
  - **Giải pháp thông minh (SaaS):** Dịch vụ cloud API xử lý giọng nói từ xe (trong chạm các startup voice platform như Snips, Rhai, Xiaomi, v.v.).  
  - **Nghiên cứu & Tư vấn:** Giải pháp nổi bật có thể cấp phép, hoặc hợp tác R&D với hãng xe/trung tâm nghiên cứu oto.  
- **Lợi thế cạnh tranh:** AISHELL-5 là tập data đặc thù cho xe; giải pháp của bạn vì thế “ra đời cùng data này”. Không nhiều công ty có chuẩn dữ liệu hay mô hình cho multi-speaker in-car. Điều này giúp bạn “đi trước” so với các giải pháp voice assistant chung chung.  
- **Demo pitch:** Tóm tắt ngắn gọn: “Hệ thống AI hiểu được khi nhiều người cùng nói trong cabin. Demo: phát âm thanh chồng lệnh – Hệ thống nhận đúng lệnh từng người. Các đối thủ (Alexa, Siri…) không làm được.” Kèm minh hoạ trên xe mô hình. Đây chính là ‘wow factor’ để thu hút đối tác.

# 11. Đóng góp khoa học & Nghiên cứu

Nếu viết báo cáo hoặc paper, các **đóng góp** có thể là:  
- **Đề xuất pipeline và benchmark mới** cho ASR đa speaker ở không gian xe hơi (trên AISHELL-5). Kết quả (bảng) sẽ là baseline cho nghiên cứu sau.  
- **Thí nghiệm phân tích:** So sánh WER giữa mô hình single-channel vs multi-channel (with separation), chứng minh tầm quan trọng của thông tin multi-kênh và separation. Bảng phân tích cpWER trước/sau.  
- **Speaker attribution:** Báo cáo về việc sử dụng embedding (ECAPA) hoặc rule-based để gắn nhãn speaker (Driver/Passenger), và ảnh hưởng đến kết quả cuối.  
- **System & Demo:** Mô tả hệ thống end-to-end lần đầu tiên (theo kiến trúc ở trên). Có thể trình bày như demo thực nghiệm, bổ sung video hoặc GUI (giao diện Streamlit).  
- **Bảng kết quả:** Nên bao gồm ít nhất một table tóm tắt WER/cpWER các cấu hình thử nghiệm (như bảng **Experiment Matrix** ở trên). Đồng thời nêu các metric bổ sung: latency, error breakdown.  
- **Khẳng định:** "Đây là benchmark đầu tiên cho ASR trong môi trường xe hơi đa người", có thể chiếm spotlight nếu chưa có paper nào trên AISHELL-5.  
- **Thí nghiệm mở rộng:** Nếu còn thời gian, show thêm một kịch bản: ASR âm thanh ồn (noise set) để thử robustness. Hoặc thử fine-tune model nhỏ trên AISHELL-5 (transfer learning). Tuy nhiên, hãy ưu tiên lộ trình cơ bản.

**Tóm lại:** Mục tiêu là làm một đề xuất *khả thi* (4–8 tuần), *chú trọng ứng dụng* và *demo thực tế*, không phải đi sâu vào cải tiến model phức tạp. Áp dụng công nghệ có sẵn để xây dựng hệ thống, rồi viết báo cáo bài bản như paper, kèm số liệu và hình ảnh minh họa. Việc đặt thành bài nghiên cứu là khả thi vì đây là vấn đề mới với dữ liệu mới, đồng thời cũng có thể biến thành nền tảng sản phẩm.  

**Nguồn tham khảo:** AISHELL-5 trên OpenSLR, Arxiv/Interspeech 2025, HuggingFace model cards (Whisper, SpeechBrain ECAPA), GitHub Asteroid, cũng như tài liệu chính thức của OpenAI Whisper.  Các link cụ thể (dataset, model, toolkit) được nhắc trong đề xuất để bạn dễ truy cập.