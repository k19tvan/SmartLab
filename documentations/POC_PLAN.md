# SmartLab — Kế hoạch chi tiết triển khai Giai đoạn POC (Proof of Concept)
*(Đã cập nhật: Loại bỏ ROI — Mô hình phòng kín Full-Frame)*

Tài liệu này chi tiết hóa toàn bộ các yêu cầu của giai đoạn **POC (Proof of Concept)** của dự án **SmartLab** thành hệ thống các **Issues/Tickets** kỹ thuật độc lập. Vì phòng câu lạc bộ là **không gian phòng kín (Enclosed Room)** với một camera cố định bao quát toàn cảnh, hệ thống áp dụng cơ chế **Full-Frame Detection (đếm trên toàn bộ khung hình)**, loại bỏ hoàn toàn sự phức tạp không cần thiết của đa giác ROI và quy tắc neo chân (Foot-Anchor).

---

## 1. Tổng quan kế hoạch Sprint POC

### 1.1. Mục tiêu tối thượng của POC
1. **Kiểm chứng khả thi thị giác máy tính toàn khung hình (Full-Frame Vision Feasibility)**: Chứng minh mô hình phát hiện đối tượng (YOLOv8n) phát hiện và đếm chính xác số người hiện diện trong phòng kín với sai số chấp nhận được (Count MAE <= 1.8 người) trên video mẫu thực tế.
2. **Kiểm chứng khả thi tổng hợp & chuỗi thời gian (Aggregation & Baseline Feasibility)**: Chuẩn hóa quy trình chuyển đổi số đếm thô -> làm mượt trung vị (Median Filter) -> tổng hợp occupancy theo giờ -> áp dụng mô hình dự báo Baseline (Same-Hour Last Week, Rolling Mean).
3. **Sẵn sàng chuyển đổi lên MVP (MVP Readiness Gate)**: Nhận diện các điểm mù (blind spots), tình trạng che khuất khi ngồi sau bàn ghế, hiện tượng ngược sáng từ cửa sổ/đèn trần và chốt bộ tham số tối ưu cho giai đoạn MVP trên AWS.

### 1.2. Sơ đồ phụ thuộc giữa các Issue (Issue Dependency Graph)

```text
+-----------------------------------------------------------------------------------+
| SƠ ĐỒ PHỤ THUỘC GIỮA CÁC ISSUES TRONG SPRINT POC                                 |
+-----------------------------------------------------------------------------------+
[POC-01] Thu thập video mẫu & Khảo sát phòng kín
    |-- [POC-02] Xây dựng tập kiểm thử Ground Truth 50 frames
    |        └── [POC-05] Đánh giá định lượng mô hình đếm ────────┐
    └── [POC-03] Ingestion & Nhận diện người Full-Frame (YOLOv8n) │
             |-- [POC-04] Bộ lọc trung vị & Hourly Occupancy      │
             |        |-- [POC-05] ───────────────────────────────┼──> [POC-08] Nghiệm thu
             |        └── [POC-06] Mô hình dự báo Baseline ───────┼──>         và chuyển giao
             └── [POC-07] Dashboard tương tác Streamlit ──────────┘            lên MVP
+-----------------------------------------------------------------------------------+
```

---

## 2. Chi tiết từng Issue/Ticket kỹ thuật

---

### [POC-01] Thu thập dữ liệu video mẫu & Khảo sát góc nhìn phòng kín

* **Issue ID**: `POC-01`
* **Epic**: Ingestion & Data Preparation
* **Nguồn Requirement**: `idea.md` Mục 3.3, 4.1; `README.md` Mục 1.1, 2.2.
* **Thời lượng ước tính**: 2 ngày

#### Mô tả (Description)
Thu thập các đoạn video clip ghi hình thực tế tại phòng kín CLB từ góc camera cố định. Khảo sát phạm vi bao quát của camera đối với toàn bộ không gian phòng kín, xác định các điểm mù (góc khuất cột, sau tủ kệ) và điều kiện ánh sáng (bật đèn, tắt đèn, ánh sáng tự nhiên).

#### Phạm vi công việc (Scope)
1. Thu thập tối thiểu 2 đoạn video clip (`.mp4`, 1080p hoặc 720p, 25/30 FPS), mỗi đoạn 15–30 phút:
   * *Clip A*: Khung giờ vắng / ít người (0–5 người).
   * *Clip B*: Khung giờ đông / sinh hoạt CLB (10–25 người).
2. Tạo file cấu hình camera `config/camera_poc.json` ghi nhận thông số kỹ thuật (độ phân giải, FPS, góc quan sát bao quát phòng).
3. Đánh giá sơ bộ độ rõ nét của hình ảnh và độ bao quát: Xác nhận camera có nhìn thấy phần lớn không gian sinh hoạt của phòng kín hay không.

#### Ngoài phạm vi (Out of Scope)
* Không cần vẽ đa giác ROI hay lọc bỏ hành lang vì phòng là không gian kín độc lập.
* Chưa kết nối luồng RTSP camera thật (dành cho MVP).

#### Hướng dẫn triển khai kỹ thuật
* Cấu trúc file `config/camera_poc.json`:
  ```json
  {
    "camera_id": "cam_club_01",
    "room_name": "Club Main Room (Enclosed)",
    "frame_width": 1920,
    "frame_height": 1080,
    "fps": 30,
    "detection_mode": "full_frame",
    "notes": "Fixed camera covering entire enclosed room area"
  }
  ```

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Có ít nhất 2 video clips mẫu được lưu tại thư mục `data/raw_videos/` (hoặc DVC tracked).
- [ ] File `config/camera_poc.json` được tạo lập đầy đủ thông số kỹ thuật.
- [ ] Ảnh snapshot đầu tiên của phòng kín `docs/images/room_overview.png` thể hiện góc bao quát rõ ràng toàn bộ phòng.
- [ ] Xác nhận không có hiện tượng camera bị che khuất nghiêm trọng (>20% diện tích phòng).

---

### [POC-02] Xây dựng tập kiểm thử Ground Truth 50 frames gán nhãn thủ công

* **Issue ID**: `POC-02`
* **Epic**: Evaluation & Quality Assurance
* **Nguồn Requirement**: `idea.md` Mục 4.1, 10.1; `README.md` Mục 2.2, 6.1.
* **Thời lượng ước tính**: 2 ngày

#### Mô tả (Description)
Để đánh giá mô hình đếm khách quan, cần có tập dữ liệu "Ground Truth" do con người đếm thủ công toàn bộ số người có mặt trong phòng trên 50 khung hình đại diện.

#### Phạm vi công việc (Scope)
1. Trích xuất ngẫu nhiên **50 khung hình** từ các video clip thu thập ở `POC-01`:
   * 15 frames: Cảnh vắng (0–4 người).
   * 20 frames: Cảnh trung bình (5–12 người).
   * 15 frames: Cảnh đông/sinh hoạt tập trung (13–25+ người).
   * *Quy tắc độc lập mẫu*: Các frames trích xuất phải cách nhau tối thiểu 10–30 giây để tránh phụ thuộc chuỗi thời gian.
2. Gán nhãn thủ công số người nhìn thấy trong toàn bộ khung hình (`actual_count`), bao gồm cả người đứng, người ngồi sau bàn, người ở góc phòng.
3. Lưu nhãn kiểm thử vào file `data/ground_truth/poc_ground_truth.json`.
4. Ghi chú các trường hợp khó (người bị che khuất một phần sau màn hình máy tính, lưng ghế).

#### Ngoài phạm vi (Out of Scope)
* Không cần gán nhãn bounding box chi tiết hay loại trừ theo vùng ROI.

#### Hướng dẫn triển khai kỹ thuật
* Cấu trúc file `data/ground_truth/poc_ground_truth.json`:
  ```json
  [
    {
      "frame_id": "frame_001.jpg",
      "timestamp_offset_sec": 45.0,
      "crowd_level": "medium",
      "ground_truth_count": 8,
      "annotator": "member_cv_01",
      "notes": "2 people sitting behind conference table"
    }
  ]
  ```

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Thư mục `data/ground_truth/frames/` chứa đủ 50 ảnh `.jpg` độ phân giải gốc.
- [ ] File `data/ground_truth/poc_ground_truth.json` chứa đủ 50 bản ghi tương ứng.
- [ ] Đủ 3 mức độ mật độ người (vắng, vừa, đông).
- [ ] 2 thành viên trong nhóm đối chiếu chéo (Cross-check) 10 frames ngẫu nhiên để xác nhận tính nhất quán của số đếm thủ công.

---

### [POC-03] Module Ingestion & Nhận diện người Full-Frame (YOLOv8n)

* **Issue ID**: `POC-03`
* **Epic**: Computer Vision (Pipeline A)
* **Nguồn Requirement**: `idea.md` Mục 6.1, 6.3; `README.md` Mục 2.2, 4.1.
* **Thời lượng ước tính**: 2 ngày *(Rút ngắn do không cần tính toán hình học ROI)*

#### Mô tả (Description)
Xây dựng module đọc video, lấy mẫu khung hình theo tần suất 1 FPS, chạy suy luận phát hiện đối tượng với mô hình `yolov8n.pt` trên toàn bộ khung hình (Full-Frame), lọc class `person` và xuất số đếm thô (`raw_count`).

#### Phạm vi công việc (Scope)
1. Xây dựng module `src/vision/video_reader.py`:
   * Đọc video file qua OpenCV `cv2.VideoCapture`.
   * Lấy mẫu khung hình với tần suất chính xác 1 FPS.
2. Xây dựng module `src/vision/detector.py`:
   * Tải mô hình `yolov8n.pt` (PyTorch hoặc ONNX).
   * Lọc chỉ giữ lại class `person` (Class ID 0 trong tập COCO).
   * Ngưỡng độ tin cậy mặc định: conf >= 0.35.
   * Đếm trực tiếp toàn bộ các bounding boxes người được phát hiện trong khung hình:
     ```text
raw_count = sum(boxes with class == 0 and conf >= 0.35)
```
3. Xuất kết quả từng frame: `frame_idx`, `timestamp`, `raw_count`, danh sách bounding boxes.

#### Ngoài phạm vi (Out of Scope)
* Không cần kiểm tra đa giác ROI hay tính toán điểm neo chân (foot-anchor).

#### Hướng dẫn triển khai kỹ thuật
```python
# src/vision/detector.py (Đơn giản, gọn nhẹ và tối ưu tốc độ)
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, conf_thresh=0.35):
        # Dự đoán trực tiếp class person (0) trên toàn bộ khung hình
        results = self.model.predict(frame, classes=[0], conf=conf_thresh, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            boxes.append(box.xyxy[0].tolist())
        return len(boxes), boxes
```

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Chạy lệnh `python -m src.vision.detector --video data/raw_videos/clip_sample.mp4` xử lý trơn tru không lỗi.
- [ ] Tốc độ suy luận đạt tối thiểu >= 15 FPS trên CPU máy tính dev thông thường (nhanh hơn phiên bản có ROI).
- [ ] Kết quả xuất ra file CSV tạm `output/poc_raw_counts.csv` gồm: `frame_idx, timestamp_sec, raw_count`.
- [ ] Unit test `tests/test_vision.py` kiểm tra phát hiện person trên ảnh mẫu hoạt động chính xác.

---

### [POC-04] Bộ lọc làm mượt trung vị & Tổng hợp Hourly Occupancy

* **Issue ID**: `POC-04`
* **Epic**: Data Aggregation (Pipeline A)
* **Nguồn Requirement**: `idea.md` Mục 2.3, 6.4, 6.5, 7.1, 7.2; `README.md` Mục 1.1, 2.2.
* **Thời lượng ước tính**: 2 ngày

#### Mô tả (Description)
Số đếm thô (`raw_count`) từng giây thường bị dao động do người di chuyển che khuất thoáng qua hoặc mô hình bỏ sót trong 1–2 khung hình. Cần áp dụng bộ lọc trung vị cửa sổ trượt (Moving Median Filter) để làm mượt số đếm, sau đó tổng hợp thành số người trung bình theo giờ (`mean_count`) và tính toán tỷ lệ độ phủ dữ liệu hợp lệ (`valid_coverage`).

#### Phạm vi công việc (Scope)
1. Xây dựng module `src/vision/smoother.py`:
   * Cài đặt bộ lọc trượt trung vị với kích thước cửa sổ W = 5 giây (chỉ dùng các quan sát quá khứ t-4 đến t).
   * Xuất `smoothed_count = round(median(raw_counts[t-4:t]), 1)`.
2. Xây dựng module `src/aggregator/poc_aggregator.py`:
   * Đọc chuỗi thời gian `smoothed_count`.
   * Gom nhóm thành các bucket 1 giờ (và bucket 15 phút).
   * Tính mean_count = (1/K) * sum(smoothed_count), max_count = max(smoothed_count).
   * Tính `valid_coverage = valid_samples / expected_samples`.
   * Quy tắc chuẩn 90%: Nếu `valid_coverage >= 0.90` thì `is_eligible = True`, ngược lại `False`.
3. Xuất kết quả ra file `data/poc_hourly_occupancy.csv`.

#### Ngoài phạm vi (Out of Scope)
* Chưa ghi vào cơ sở dữ liệu PostgreSQL (dùng file CSV/DuckDB nội bộ cho POC).

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Dữ liệu sau làm mượt triệt tiêu được các dao động tức thời (ví dụ chuỗi 10, 10, 0, 10, 10 -> 10, 10, 10, 10, 10).
- [ ] File `data/poc_hourly_occupancy.csv` được sinh ra tự động và đúng schema.
- [ ] Unit test `tests/test_aggregator.py` kiểm tra:
  - Khung giờ thiếu > 10% dữ liệu được gán đúng `is_eligible = False`.
  - Tuyệt đối không tự ý điền số 0 vào các khoảng thiếu dữ liệu.

---

### [POC-05] Đánh giá định lượng mô hình đếm (Count MAE & Error Analysis)

* **Issue ID**: `POC-05`
* **Epic**: Evaluation & Quality Assurance
* **Nguồn Requirement**: `idea.md` Mục 10.1, 16.2; `README.md` Mục 2.2, 6.1.
* **Thời lượng ước tính**: 2 ngày

#### Mô tả (Description)
Thực hiện đánh giá định lượng độ chính xác của mô hình đếm người toàn khung hình trên tập 50 frames gán nhãn thủ công (từ `POC-02`), tính toán Count MAE tổng thể, Count MAE cho cảnh đông và phân tích các trường hợp sai lệch do che khuất bàn ghế trong phòng.

#### Phạm vi công việc (Scope)
1. Xây dựng script `src/evaluation/eval_detector.py`:
   * Đọc 50 frames kiểm thử và nhãn ground truth từ `data/ground_truth/poc_ground_truth.json`.
   * Chạy suy luận Full-Frame trên 50 frames để thu được y_pred_i.
   * Tính toán các chỉ số:
     * Overall Count MAE = (1/N) * sum(|y_pred_i - y_true_i|).
     * Crowded Scene MAE (trên các frames có y_true >= 13).
     * Mean Latency (thời gian suy luận trung bình tính bằng ms/frame).
2. Xây dựng báo cáo phân tích lỗi (Error Analysis Report):
   * Tìm top 5 frames có độ lệch lớn nhất |y_pred - y_true|.
   * Lưu ảnh trực quan hóa vào thư mục `docs/reports/poc_errors/`.
   * Đánh giá nguyên nhân: Ngồi khuất sau ghế cao/màn hình máy tính, góc chiếu đèn trần chói sáng.
3. Xuất file tóm tắt `docs/reports/poc_vision_report.md`.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Chạy `python -m src.evaluation.eval_detector` tự động sinh bảng báo cáo số liệu.
- [ ] **Chỉ số đạt yêu cầu Definition of Done của POC**:
  - Overall Count MAE <= 1.8 người.
  - Crowded Scene MAE <= 2.5 người.
- [ ] File `docs/reports/poc_vision_report.md` phân tích rõ các góc bàn ghế bị khuất và khuyến nghị vị trí gắn camera tối ưu cho MVP.

---

### [POC-06] Mô hình dự báo Baseline & Lịch phát hành cutoff

* **Issue ID**: `POC-06`
* **Epic**: Forecasting (Pipeline B)
* **Nguồn Requirement**: `idea.md` Mục 2.4, 2.5, 8.1, 8.4; `README.md` Mục 2.2, 6.2.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Xây dựng và đánh giá 2 mô hình dự báo Baseline trên chuỗi thời gian occupancy theo giờ. Thiết lập cơ chế phát hành dự báo với giới hạn thời gian dữ liệu nghiêm ngặt (`data_cutoff = 23:00 Chủ Nhật`) để kiểm chứng logic chọn Ngày đông nhất và Giờ đông nhất mà không bị rò rỉ dữ liệu tương lai.

#### Phạm vi công việc (Scope)
1. Chuẩn bị tập dữ liệu chuỗi thời gian occupancy:
   * Kết hợp dữ liệu thực tế từ video mẫu với dữ liệu mô phỏng 4 tuần theo lịch sinh hoạt thực tế của phòng CLB (08:00 – 21:00 hàng ngày, có chu kỳ tuần và ngày nghỉ).
2. Xây dựng module `src/forecasting/poc_baselines.py`:
   * **Baseline 1 — Same-Hour Last Week**: Dự báo khung giờ T tuần này bằng đúng giá trị thực tế của khung giờ T-168h (cùng giờ thứ tương ứng tuần trước).
   * **Baseline 2 — Rolling Mean**: Dự báo bằng trung bình cộng của cùng khung giờ trong 3 tuần gần nhất.
3. Áp dụng quy tắc xếp hạng:
   * Chỉ xếp hạng trong các giờ mở cửa (`is_open = True`).
   * Chọn Giờ đông nhất từng ngày (max(predicted_mean)).
   * Tính trung bình ngày và chọn Ngày đông nhất trong tuần.
4. Đánh giá sai số trên 1 tuần kiểm thử:
   * Tính Hourly MAE của Baseline 1 vs Baseline 2.
   * Tính Peak Regret (sai lệch giữa đỉnh thực tế và đỉnh do baseline lựa chọn).

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Chạy lệnh `python -m src.forecasting.poc_baselines --cutoff "2026-11-15 23:00:00"` xuất kết quả dự báo 84 giờ tiếp theo ra file `output/poc_forecast_results.csv`.
- [ ] Báo cáo được so sánh giữa Baseline 1 và Baseline 2 với các chỉ số: `Hourly MAE`, `Peak Hour Accuracy`, `Peak Regret`.
- [ ] Không có hiện tượng rò rỉ dữ liệu (không truy cập bất kỳ dòng nào sau mốc cutoff).
- [ ] Xử lý đúng trường hợp ngày đóng cửa: hiển thị "Đóng cửa — không xếp hạng" thay vì chọn bừa một khung giờ.

---

### [POC-07] Dashboard tương tác POC (Streamlit Local UI)

* **Issue ID**: `POC-07`
* **Epic**: Frontend & Visualization
* **Nguồn Requirement**: `idea.md` Mục 4.3, 12.1, 12.2, 12.3; `README.md` Mục 2.2, 4.1.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Xây dựng một ứng dụng web nội bộ bằng **Streamlit** chạy cục bộ để trực quan hóa toàn bộ chuỗi xử lý của POC, giúp toàn bộ thành viên nhóm và stakeholders nghiệm thu trực quan kết quả.

#### Phạm vi công việc (Scope)
Xây dựng ứng dụng `app_poc.py` gồm 3 màn hình (Tabs):
1. **Tab 1: Computer Vision Inspector**:
   * Cho phép chọn video clip mẫu phòng kín.
   * Hiển thị video/ảnh kèm bounding box của tất cả người được phát hiện trong phòng.
   * Hiển thị trực tiếp: Số người thô (`raw_count`), số người sau làm mượt (`smoothed_count`), và thời gian suy luận (ms).
2. **Tab 2: Historical Occupancy Trends**:
   * Biểu đồ đường (Line Chart) biểu diễn số người trung bình theo từng giờ trong phòng.
   * Đánh dấu rõ các mốc cao điểm trong ngày.
   * Cảnh báo các khung giờ bị thiếu dữ liệu (valid_coverage < 90%).
3. **Tab 3: Next-Week Forecast & Baseline Benchmark**:
   * Bảng Heatmap hiển thị số người dự báo cho 7 ngày x các giờ mở cửa.
   * Nổi bật **Ngày đông nhất tuần** và **Giờ đông nhất mỗi ngày**.
   * Bảng đối chiếu so sánh kết quả dự báo giữa Baseline 1, Baseline 2 và Thực tế.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Chạy lệnh `streamlit run app_poc.py` khởi động ứng dụng mượt mà không lỗi.
- [ ] Giao diện trực quan, hiển thị video kèm bounding box rõ nét trên toàn bộ không gian phòng kín.
- [ ] Render biểu đồ lịch sử và heatmap dự báo đúng chuẩn múi giờ `Asia/Ho_Chi_Minh`.

---

### [POC-08] Báo cáo nghiệm thu POC & Tiêu chuẩn chuyển giao lên MVP

* **Issue ID**: `POC-08`
* **Epic**: MLOps & Project Governance
* **Nguồn Requirement**: `idea.md` Mục 14, 15, 16; `README.md` Mục 2.2, 6.3.
* **Thời lượng ước tính**: 2 ngày

#### Mô tả (Description)
Tổng hợp toàn bộ kết quả thử nghiệm thực tế của giai đoạn POC, đối chiếu với Definition of Done (DoD) của POC, đánh giá các điểm mù trong phòng kín và phê duyệt chuyển giao sang xây dựng hệ thống tự động hóa MVP trên AWS.

#### Phạm vi công việc (Scope)
1. Soạn thảo tài liệu báo cáo `docs/POC_FINAL_REPORT.md`:
   * Bảng tổng hợp các metric đã đo đạc (Count MAE, Latency, Baseline MAE, Peak Regret).
   * Đánh giá tính khả thi của góc đặt camera hiện tại trong phòng kín (các góc khuất, phản chiếu ánh sáng đèn/cửa sổ).
   * Xác nhận tính ưu việt của Full-Frame Detection so với cấu hình ROI phức tạp trong môi trường phòng kín.
2. Xây dựng bản khuyến nghị cho MVP:
   * Cấu hình phần cứng tối thiểu cho Camera Worker (CPU vs GPU).
   * Lựa chọn kiến trúc Ingestion: Hybrid (Edge Worker) vs Cloud Ingestion.
   * Ngưỡng làm mượt và kích thước cửa sổ tối ưu cho Camera thực tế.
3. Tổ chức buổi nghiệm thu nội bộ (Internal Demo & Review) đạt toàn bộ tiêu chí DoD của POC.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Báo cáo `docs/POC_FINAL_REPORT.md` hoàn thiện kèm biểu đồ minh họa trích xuất từ `POC-05` và `POC-06`.
- [ ] Toàn bộ checklist **Definition of Done của POC** được tích xanh:
  - [x] Code chạy end-to-end từ video -> số đếm -> occupancy -> baseline forecast -> Streamlit UI.
  - [x] Đạt chỉ số Count MAE <= 1.8 trên tập 50 frames gán nhãn thủ công.
  - [x] Baseline forecast không bị rò rỉ dữ liệu tương lai.
- [ ] Được sự phê duyệt và thống nhất của toàn bộ nhóm để bắt đầu Sprint MVP.

---

## 3. Bảng phân công trách nhiệm & Kế hoạch tiến độ (WBS & RACI Matrix)

| Issue ID | Tên Issue / Hạng mục | Người phụ trách chính (R) | Người phối hợp (C) | Thời gian dự kiến | Phụ thuộc |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`POC-01`** | Thu thập video mẫu & Khảo sát phòng kín | **CV Engineer** | QA / Domain Expert | Ngày 1 – Ngày 2 | Không |
| **`POC-02`** | Gán nhãn thủ công 50 frames Ground Truth | **QA / Data Annotator** | CV Engineer | Ngày 2 – Ngày 3 | `POC-01` |
| **`POC-03`** | Ingestion & Nhận diện người Full-Frame | **CV Engineer** | Backend Engineer | Ngày 3 – Ngày 4 | `POC-01` |
| **`POC-04`** | Bộ lọc trung vị & Tổng hợp Hourly Occupancy | **Data Engineer** | CV Engineer | Ngày 4 – Ngày 5 | `POC-03` |
| **`POC-05`** | Đánh giá định lượng mô hình đếm | **CV Engineer** | QA / Tester | Ngày 5 – Ngày 6 | `POC-02`, `POC-04` |
| **`POC-06`** | Mô hình dự báo Baseline & Lịch phát hành | **Forecasting Engineer** | Data Engineer | Ngày 6 – Ngày 8 | `POC-04` |
| **`POC-07`** | Dashboard trực quan hóa Streamlit | **Frontend / Fullstack** | CV & Forecast Eng | Ngày 7 – Ngày 9 | `POC-03`, `POC-04`, `POC-06` |
| **`POC-08`** | Báo cáo nghiệm thu & Chuyển giao MVP | **MLOps Lead / Project Lead**| Cả nhóm | Ngày 9 – Ngày 10| `POC-05`, `POC-06`, `POC-07` |

*(Tổng thời gian thực hiện ước tính cho giai đoạn POC: **2 tuần làm việc**)*

---

## 4. Ma trận kiểm thử & Tiêu chuẩn nghiệm thu (Acceptance Test Matrix)

| Test Case ID | Tình huống kiểm thử | Các bước thực hiện | Kết quả mong đợi |
| :--- | :--- | :--- | :--- |
| **`TC-POC-01`** | Kiểm tra nhận diện Full-Frame các vị trí | Đưa vào frame có người ở các vị trí khác nhau trong phòng (gần camera, xa camera, ngồi sau bàn). | Tất cả người nhìn thấy đều được nhận diện bounding box; `raw_count` bằng đúng số người thực tế. |
| **`TC-POC-02`** | Kiểm tra độ trễ suy luận | Chạy suy luận trên 100 frames liên tiếp trên CPU máy dev. | Tốc độ trung bình >= 15 FPS (<= 66 ms/frame). |
| **`TC-POC-03`** | Kiểm tra khả năng khử nhiễu của Median Filter | Tạo chuỗi số đếm thô có giọt nhiễu đột ngột (8, 8, 0, 8, 8 hoặc 8, 8, 20, 8, 8). | Chuỗi sau lọc làm mượt thành 8, 8, 8, 8, 8. |
| **`TC-POC-04`** | Kiểm tra tính chính xác của mô hình đếm | Chạy `eval_detector.py` trên 50 frames gán nhãn thủ công. | Count MAE <= 1.8; Crowded MAE <= 2.5. |
| **`TC-POC-05`** | Kiểm tra ngăn ngừa rò rỉ dữ liệu dự báo | Chạy script baseline với cutoff Chủ Nhật 23:00; kiểm tra dữ liệu đầu vào. | Tuyệt đối không có bản ghi nào sau mốc 23:00 được đưa vào feature. |
| **`TC-POC-06`** | Khởi chạy giao diện Streamlit | Chạy `streamlit run app_poc.py`. | Mở trình duyệt hiển thị đầy đủ 3 tabs, video play được và biểu đồ render đúng. |
