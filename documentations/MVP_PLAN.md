# SmartLab — Kế hoạch chi tiết triển khai Giai đoạn MVP (Minimum Viable Product)
*(Production-Ready MLOps System trên AWS & GitHub Actions)*

Tài liệu này chi tiết hóa toàn bộ các yêu cầu của giai đoạn **MVP (Minimum Viable Product)** của dự án **SmartLab** thành hệ thống các **Issues/Tickets** kỹ thuật độc lập (chuẩn format GitHub Issues / Jira). Hệ thống được thiết kế hướng tới môi trường Production thực tế trên nền tảng **Amazon Web Services (AWS)**, quy trình tự động hóa **GitHub Actions CI/CD/MLOps**, áp dụng cơ chế **Full-Frame Person Detection (không dùng ROI do phòng kín)**, quản lý 2 pipeline máy học riêng biệt và vượt qua đầy đủ 6 kịch bản kiểm thử bắt buộc.

---

## 1. Tổng quan kế hoạch Sprint MVP

### 1.1. Mục tiêu cốt lõi của MVP
1. **Tự động hóa luồng Camera thật 24/7 (Pipeline A)**: Thu nhận luồng RTSP từ camera phòng kín, xử lý đếm người thời gian thực bằng YOLOv8n ONNX, lọc nhiễu median filter, tổng hợp occupancy theo giờ kèm cơ chế kiểm soát chất lượng dữ liệu (ngưỡng 90% coverage).
2. **Dự báo tuần tới tự động & chống rò rỉ dữ liệu (Pipeline B)**: Tự động chạy batch job lúc 23:00 Chủ Nhật hàng tuần (`Asia/Ho_Chi_Minh`), áp dụng `data_cutoff` nghiêm ngặt, huấn luyện mô hình GBDT (LightGBM) và so sánh với Baseline qua Rolling Backtest để dự báo 84 giờ hoạt động của tuần tới.
3. **MLOps Governance & Vận hành tin cậy**: Quản lý phiên bản dữ liệu bằng DVC + AWS S3, quản lý vòng đời mô hình bằng MLflow Model Registry, cổng kiểm duyệt tự động (Automated Gatekeeper) và cơ chế Rollback an toàn chỉ mất dưới 30 giây.
4. **Hạ tầng Cloud & CI/CD hoàn chỉnh**: Vận hành trên AWS (S3, RDS PostgreSQL, ECS Fargate, CloudWatch, SNS) với quy trình CI/CD/CML tự động từ GitHub Actions qua kết nối bảo mật GitHub OIDC.
5. **Dashboard 4 phân hệ**: Cung cấp giao diện trực quan cho cả người quản lý phòng CLB và kỹ sư vận hành MLOps.

---

### 1.2. Sơ đồ phụ thuộc giữa các Issue MVP (Issue Dependency Graph)

```text
========================================================================================
                  SƠ ĐỒ PHỤ THUỘC GIỮA 13 ISSUES TRONG SPRINT MVP
========================================================================================

[HẠ TẦNG & DỮ LIỆU NỀN TẢNG]
  [MVP-01] Thiết lập Hạ tầng AWS (Terraform, S3, RDS, ECS, OIDC)
      |-- [MVP-02] Thiết kế Database Schema & Migration (PostgreSQL) ──> [PIPELINE A]
      |-- [MVP-06] Data Management & Versioning với DVC & AWS S3 ───────> [PIPELINE B]
      └── [MVP-11] CI/CD & MLOps Workflows trên GitHub Actions ─────────> [NGHIỆM THU]

[PIPELINE A: VISION & INGESTION]
  [MVP-03] RTSP Stream Ingestion & Resilient Camera Worker
      └──> [MVP-04] Full-Frame Detection Engine (ONNX) & Median Filter
               └──> [MVP-05] Data Aggregator: Tổng hợp Hourly Occupancy (90%) ──┐
                                                                                |
[PIPELINE B: FORECASTING & EVALUATION]                                          |
  [MVP-07] Feature Engineering chống rò rỉ dữ liệu (Cutoff 23:00 CN)            |
      └──> [MVP-08] Huấn luyện LightGBM & Framework Rolling Backtest            |
               └──> [MVP-09] MLflow Model Registry & Automated Gatekeeper       |
                        └──> [MVP-10] Synchronized Safe Rollback Engine         |
                                                                                |
[PHỤC VỤ & NGHIỆM THU]                                                          |
  [MVP-12] Dashboard 4 phân hệ & API Serving (FastAPI + Streamlit) <────────────┤
  [MVP-13] Nghiệm thu 6 Kịch bản Bắt buộc & Bàn giao Hệ thống <─────────────────┘
========================================================================================
```

---

## 2. Chi tiết từng Issue/Ticket kỹ thuật MVP

---

### [MVP-01] Thiết lập Hạ tầng AWS (Terraform, S3, RDS, ECS, OIDC)

* **Issue ID**: `MVP-01`
* **Epic**: Cloud Infrastructure & DevOps
* **Nguồn Requirement**: `idea.md` Mục 13; `README.md` Mục 4.1, 4.3.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Xây dựng toàn bộ hạ tầng đám mây trên **AWS (Region: `ap-southeast-1`)** sử dụng **Terraform (Infrastructure as Code)**. Cấu hình bảo mật IAM Role hỗ trợ **GitHub OIDC** để GitHub Actions có thể deploy và truy cập tài nguyên AWS mà không cần lưu trữ AWS Access Key / Secret Key tĩnh.

#### Phạm vi công việc (Scope)
1. Viết mã Terraform khởi tạo các tài nguyên AWS:
   * **AWS S3**: Bucket `smartlab-mlops-artifacts-<account_id>` với các prefix `dvc-storage/`, `mlflow-artifacts/`, `error-frames/`.
   * **AWS RDS PostgreSQL 16**: Instance `db.t4g.micro` (hoặc `db.t4g.small`), kích hoạt Automated Backup và Storage Encryption.
   * **AWS ECR**: 3 repositories chứa container images: `smartlab-camera-worker`, `smartlab-forecaster`, `smartlab-dashboard`.
   * **AWS ECS Fargate**: Cluster `smartlab-cluster` phục vụ scheduled task chạy dự báo và web service.
   * **AWS CloudWatch & SNS**: Log Group `/ecs/smartlab`, CloudWatch Metric Filter phát hiện mất heartbeat camera, SNS Topic `smartlab-alerts` gửi thông báo Email/Telegram.
2. Cấu hình **AWS IAM OIDC Identity Provider** trỏ tới `token.actions.githubusercontent.com`.
3. Tạo IAM Role `GitHubActionsSmartLabRole` với chính sách Least-Privilege cho phép GitHub Actions đẩy image lên ECR và đọc/ghi S3 DVC bucket.

#### Ngoài phạm vi (Out of Scope)
* Chưa triển khai cụm Kubernetes đa vùng phức tạp (Fargate serverless đáp ứng đủ bài toán với chi phí tối ưu).

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Chạy lệnh `terraform plan` và `terraform apply` khởi tạo thành công 100% tài nguyên mà không có lỗi.
- [ ] GitHub Actions có thể xác thực thành công với AWS qua OIDC Role và thực hiện lệnh `aws s3 ls`.
- [ ] RDS PostgreSQL có thể kết nối an toàn từ VPC nội bộ qua kết nối được mã hóa TLS.
- [ ] SNS Topic gửi thành công 1 email kiểm thử xác nhận đăng ký nhận cảnh báo.

---

### [MVP-02] Thiết kế Database Schema & Migration (PostgreSQL & Alembic)

* **Issue ID**: `MVP-02`
* **Epic**: Database & Backend Architecture
* **Nguồn Requirement**: `idea.md` Mục 7; `README.md` Mục 5.
* **Thời lượng ước tính**: 2 ngày

#### Mô tả (Description)
Xây dựng mô hình dữ liệu quan hệ (Relational Data Models) sử dụng **SQLAlchemy 2.0** và thiết lập công cụ quản lý lược đồ **Alembic**. CSDL lưu trữ trơn tru cả 2 pipeline: quan sát từng giây, tổng hợp theo giờ và lịch sử các đợt phát hành dự báo.

#### Phạm vi công việc (Scope)
1. Định nghĩa các bảng CSDL trong `src/database/models.py`:
   * `cameras`: `camera_id` (PK), `room_id`, `stream_url`, `status`, `last_heartbeat_at`.
   * `person_observations`: `id` (PK, BigSerial), `camera_id` (FK), `observed_at` (Indexed), `raw_count`, `smoothed_count`, `quality_status`, `count_model_version`, `camera_config_version`.
   * `hourly_occupancy`: `id` (PK), `room_id`, `hour_start` (Indexed, Unique per room), `mean_count`, `max_count`, `valid_coverage`, `is_open`, `is_eligible`, `aggregation_version`.
   * `forecast_runs`: `forecast_run_id` (PK), `issued_at`, `data_cutoff`, `forecast_model_version`, `feature_version`, `status`, `test_mae`.
   * `hourly_forecasts`: `id` (PK), `forecast_run_id` (FK), `room_id`, `target_hour` (Indexed), `predicted_mean_count`, `is_peak_hour_of_day`.
2. Tạo file migration ban đầu bằng Alembic: `alembic revision --autogenerate -m "Initial schema"`.
3. Viết script kiểm thử CRUD và tính toàn vẹn khóa ngoại trong `tests/test_database.py`.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Chạy lệnh `alembic upgrade head` áp dụng thành công schema lên AWS RDS PostgreSQL.
- [ ] Bảng `hourly_occupancy` có ràng buộc `UNIQUE(room_id, hour_start)` chống trùng lặp dữ liệu giờ.
- [ ] Bản ghi bị lỗi (`quality_status != 'VALID'`) cho phép lưu `raw_count` và `smoothed_count` là `NULL` (không ép về 0).
- [ ] Toàn bộ unit tests database trong `pytest tests/test_database.py` đạt pass 100%.

---

### [MVP-03] RTSP Stream Ingestion & Resilient Camera Worker

* **Issue ID**: `MVP-03`
* **Epic**: Computer Vision (Pipeline A)
* **Nguồn Requirement**: `idea.md` Mục 6.1, 6.2, 11.1, 11.2; `README.md` Mục 2.3, 3.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Xây dựng dịch vụ Camera Worker chạy nền liên tục (chạy trên EC2 hoặc Edge Server phòng CLB), kết nối luồng RTSP camera thật, có khả năng tự động phục hồi kết nối (Auto-reconnect with Exponential Backoff), kiểm tra tính hợp lệ của khung hình và phát hiện camera offline.

#### Phạm vi công việc (Scope)
1. Xây dựng module `src/vision/stream_reader.py`:
   * Đọc luồng RTSP bằng OpenCV / PyAV với buffer size tối thiểu để tránh tích tụ độ trễ (latency accumulation).
   * Lấy mẫu khung hình chính xác tần suất 1 FPS.
2. Cài đặt các cơ chế kiểm tra lỗi khung hình (Frame Integrity Validation):
   * **Decode Check**: Bắt lỗi khung hình bị rách, mất gói tin (`cv2.error`, null frames).
   * **Stale Timestamp Check**: Kiểm tra timestamp khung hình có bị đứng hình/treo hay không.
3. Cơ chế phục hồi kết nối (Resilient Connection):
   * Khi luồng RTSP bị ngắt: Tự động kết nối lại sau 2s, 4s, 8s... tối đa 30s.
   * Ghi log trạng thái `quality_status = 'CAMERA_OFFLINE'` hoặc `'FRAME_DECODE_ERROR'`.
   * **Tuyệt đối không lưu giá trị 0 vào CSDL khi camera mất kết nối**.
4. Heartbeat telemetry: Gửi ping metric `CameraHeartbeat = 1` lên AWS CloudWatch mỗi 60 giây.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Camera Worker chạy liên tục 24/7 không bị crash hay tràn bộ nhớ (Memory Leak).
- [ ] Khi ngắt kết nối mạng camera giả lập, worker tự động ghi nhận trạng thái `CAMERA_OFFLINE` và tự kết nối lại thành công ngay khi có mạng trở lại.
- [ ] CloudWatch Alarm phát tín hiệu cảnh báo qua SNS nếu worker ngừng gửi heartbeat quá 3 phút liên tiếp.

---

### [MVP-04] Full-Frame Detection Engine (YOLOv8n ONNX) & Median Filter

* **Issue ID**: `MVP-04`
* **Epic**: Computer Vision (Pipeline A)
* **Nguồn Requirement**: `idea.md` Mục 6.3, 6.4, 7.1; `README.md` Mục 2.3, 4.1.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Tối ưu hóa mô hình phát hiện đối tượng với định dạng ONNX Runtime chạy trên CPU/GPU, thực hiện đếm người trên toàn bộ khung hình phòng kín (Full-Frame Detection) và áp dụng bộ lọc trung vị thời gian thực (Real-time Moving Median Filter), lưu kết quả vào CSDL PostgreSQL.

#### Phạm vi công việc (Scope)
1. Xuất mô hình `yolov8n.pt` sang định dạng `yolov8n.onnx` để tối ưu hóa tốc độ và giảm phụ thuộc PyTorch nặng nề.
2. Xây dựng module `src/vision/detector.py`:
   * Chạy suy luận ONNX Runtime đa luồng (multi-threaded).
   * Lọc bounding box thuộc class `person` với ngưỡng conf >= 0.35.
   * Đếm toàn bộ số người trong phòng: raw_count = sum(boxes).
3. Xây dựng module `src/vision/smoother.py`:
   * Hàng đợi vòng tròn (Ring Buffer) trượt W = 5 quan sát gần nhất.
   * Tính `smoothed_count = round(median([r_0, ..., r_4]), 1)`.
4. Ghi bản ghi vào bảng `person_observations` trong PostgreSQL theo batch (mỗi 5 hoặc 10 giây ghi một lần để tối ưu I/O database).

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Tốc độ suy luận đạt >= 15 FPS trên CPU AWS EC2 `c6i.large` hoặc máy trạm tại chỗ.
- [ ] Khử được các xung đột biến ngắn hạn do người bị che khuất thoáng qua trong 1–2 giây.
- [ ] Bản ghi trong CSDL có đầy đủ metadata: `observed_at`, `raw_count`, `smoothed_count`, `quality_status`, `count_model_version`.

---

### [MVP-05] Data Aggregator: Tổng hợp Hourly Occupancy & Ngưỡng 90%

* **Issue ID**: `MVP-05`
* **Epic**: Data Engineering & Aggregation
* **Nguồn Requirement**: `idea.md` Mục 2.3, 6.5, 7.2; `README.md` Mục 1.1, 2.3.
* **Thời lượng ước tính**: 2 ngày

#### Mô tả (Description)
Xây dựng tác vụ định kỳ tự động chạy mỗi đầu giờ (ví dụ lúc 09:00:05 để tổng hợp giờ 08:00–09:00), tổng hợp 3.600 ô 1 giây của khung giờ vừa qua thành số người trung bình (`mean_count`), số người cao nhất (`max_count`) và tỷ lệ dữ liệu hợp lệ (`valid_coverage`), áp dụng quy tắc chuẩn 90% để gán cờ huấn luyện `is_eligible`.

#### Phạm vi công việc (Scope)
1. Xây dựng module `src/aggregator/hourly_aggregator.py`:
   * Truy vấn toàn bộ các bản ghi trong bảng `person_observations` thuộc khung giờ [T, T + 1h).
   * Tính valid_coverage = (Số lượng bản ghi có quality_status = 'VALID') / 3600.
   * Tính mean_count = mean(smoothed_count của các bản ghi VALID).
   * Tính max_count = max(smoothed_count của các bản ghi VALID).
2. Kiểm tra lịch hoạt động phòng:
   * Đọc cấu hình `config/schedule.json` để xác định `is_open` (`True`/`False`).
3. Áp dụng quy tắc nghiệm thu 90%:
   * `is_eligible = True` khi và chỉ khi: valid_coverage >= 0.90 và is_open = True.
   * Nếu valid_coverage < 0.90: Gán `is_eligible = False`. **Không tự ý điền số 0 vào giờ thiếu dữ liệu**.
4. Ghi bản ghi vào bảng `hourly_occupancy` (hỗ trợ `ON CONFLICT DO UPDATE` để có thể tính toán lại nếu cần).

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Tác vụ chạy ổn định mỗi 60 phút qua Background Scheduler / Cron.
- [ ] Khung giờ mất mạng camera 10 phút (coverage khoảng 83% < 90%) được đánh dấu chính xác `is_eligible = False`.
- [ ] Khung giờ phòng đóng cửa (ban đêm) được ghi nhận đúng `is_open = False` và `is_eligible = False`.
- [ ] Unit test `tests/test_aggregator.py` kiểm tra mọi tình huống biên (đầy đủ dữ liệu, thiếu một phần, mất toàn bộ khung hình) đều pass.

---

### [MVP-06] Data Management & Versioning với DVC & AWS S3

* **Issue ID**: `MVP-06`
* **Epic**: MLOps & Data Governance
* **Nguồn Requirement**: `idea.md` Mục 4.1, 9.1, 9.2; `README.md` Mục 4.1, 4.3.
* **Thời lượng ước tính**: 2 ngày

#### Mô tả (Description)
Thiết lập công cụ quản lý phiên bản dữ liệu **DVC (Data Version Control)** kết nối với remote storage **AWS S3**. Đảm bảo mọi đợt huấn luyện mô hình dự báo đều có thể truy xuất nguồn gốc chính xác đến từng snapshot dữ liệu chuỗi thời gian occupancy và tập ảnh kiểm thử gán nhãn thủ công.

#### Phạm vi công việc (Scope)
1. Cấu hình DVC trong repository:
   * Khởi tạo `.dvc/` trỏ tới remote storage `s3://smartlab-mlops-artifacts-<account_id>/dvc-storage/`.
   * Cấu hình AWS Region `ap-southeast-1` và xác thực qua IAM Role.
2. Xây dựng script snapshot dữ liệu định kỳ `src/mlops/data_snapshot.py`:
   * Kết xuất dữ liệu từ bảng `hourly_occupancy` ra file CSV chuẩn `data/snapshots/occupancy_snapshot_<YYYYMMDD>.csv`.
   * Tạo bản ghi DVC: `dvc add data/snapshots/occupancy_snapshot_<YYYYMMDD>.csv`.
   * Đẩy dữ liệu lên S3: `dvc push`.
   * Tạo commit Git lưu file `.dvc` tương ứng để gắn liền mã nguồn với dữ liệu.
3. Quản lý tập ảnh kiểm thử gán nhãn thủ công 100 frames kiểm thử độc lập trên S3 thông qua DVC.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Lệnh `dvc pull` và `dvc push` thực thi trơn tru giữa máy dev/CI runner và AWS S3 bucket.
- [ ] File dữ liệu lớn không bị commit trực tiếp vào Git repository.
- [ ] Mọi lần chạy huấn luyện mô hình trong MLflow đều ghi nhận mã hash `data_snapshot_id` do DVC cung cấp.

---

### [MVP-07] Feature Engineering chống rò rỉ dữ liệu (Cutoff 23:00 CN)

* **Issue ID**: `MVP-07`
* **Epic**: Time-Series Forecasting (Pipeline B)
* **Nguồn Requirement**: `idea.md` Mục 7.4, 8.2, 8.3; `README.md` Mục 1.1, 2.3.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Xây dựng pipeline tạo đặc trưng (Feature Pipeline) cho bài toán dự báo chuỗi thời gian đa bước (Multi-step Ahead Forecasting), tuân thủ nghiêm ngặt nguyên tắc **ngăn ngừa rò rỉ dữ liệu tương lai (Data Leakage Prevention)** với mốc giới hạn `data_cutoff = 23:00 Chủ Nhật`.

#### Phạm vi công việc (Scope)
1. Xây dựng module `src/forecasting/feature_builder.py`:
   * **Quy tắc Cutoff**: Chỉ truy vấn các dòng trong `hourly_occupancy` có `hour_start <= data_cutoff` và `is_eligible = True`.
   * **Temporal Features** (cho khung giờ đích cần dự báo T_target):
     * `hour_of_day` (0 - 23), `day_of_week` (0 - 6), `is_weekend` (0 hoặc 1).
     * Biến đổi chu kỳ hình sin/cosin: sin(2 * pi * hour / 24), cos(2 * pi * hour / 24).
   * **Calendar & Event Features**:
     * `is_club_open` (theo lịch phòng).
     * `event_type` (từ lịch sự kiện đã công bố trước mốc cutoff).
   * **Lag Features & Rolling Statistics** (tính toán hoàn toàn từ quá khứ trước cutoff):
     * Giá trị cùng giờ tuần trước (T - 168h).
     * Giá trị cùng giờ 2 tuần trước (T - 336h).
     * Trung bình động cùng giờ các tuần trước (Moving Average 3–4 tuần).
     * Tỷ lệ thiếu dữ liệu trong 7 ngày gần nhất.
2. Kiểm tra tính toàn vẹn của tập đặc trưng: Không chứa giá trị vô lý (`NaN`, `Inf`), mã hóa One-Hot hoặc Target Encoding đúng quy chuẩn.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Unit test `tests/test_forecasting.py` kiểm tra kiểm soát rò rỉ dữ liệu: Nếu cố tình đưa vào bản ghi sau mốc 23:00 Chủ Nhật, hàm sẽ văng lỗi hoặc tự động lọc bỏ.
- [ ] Tập đặc trưng sinh ra cho 84 giờ hoạt động tuần tới có đầy đủ kích thước và không bị thiếu cột.
- [ ] Version của module tạo đặc trưng (`feature_version = 'v1.0'`) được lưu cố định cho mỗi lần chạy.

---

### [MVP-08] Huấn luyện LightGBM & Framework Rolling Backtest

* **Issue ID**: `MVP-08`
* **Epic**: Time-Series Forecasting (Pipeline B)
* **Nguồn Requirement**: `idea.md` Mục 8.4, 10.2, 10.3; `README.md` Mục 2.3, 6.2.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Xây dựng mô hình máy học chính sử dụng thuật toán **LightGBM Regressor** để dự báo số người trung bình theo giờ. Xây dựng framework **Rolling Backtest (Walk-Forward Validation)** trên 4 tuần quá khứ độc lập để so sánh công bằng và khoa học giữa LightGBM với 2 Baseline models.

#### Phạm vi công việc (Scope)
1. Xây dựng module `src/forecasting/trainer.py`:
   * Khởi tạo mô hình `LGBMRegressor(objective='mae', n_estimators=200, learning_rate=0.05)`.
   * Huấn luyện mô hình với Early Stopping trên tập Validation.
2. Xây dựng framework `src/forecasting/backtester.py`:
   * Thực hiện Rolling Backtest qua 4 chu kỳ tuần độc lập (Tuần 1, Tuần 2, Tuần 3, Tuần 4).
   * Tại mỗi bước: Giả lập mốc cutoff lúc 23:00 Chủ Nhật của tuần đó -> huấn luyện -> dự báo 84 giờ tuần tiếp theo -> đối chiếu với thực tế đã xảy ra.
3. Tính toán bộ metric đánh giá toàn diện:
   * **Hourly MAE**: Sai số tuyệt đối trung bình trên các giờ mở cửa có dữ liệu hợp lệ.
   * **Peak Hour Accuracy**: Tỷ lệ ngày chọn trúng chính xác giờ đông nhất thực tế.
   * **Peak Hour +/- 1h Accuracy**: Tỷ lệ ngày chọn giờ đông nhất cách đỉnh thực tế tối đa 1 giờ.
   * **Peak Regret**: Sai lệch số người thực tế tại đỉnh thật so với đỉnh do mô hình chọn:
     ```text
Peak Regret = Actual_Occupancy(true_peak) - Actual_Occupancy(predicted_peak)
```
4. So sánh kết quả của LightGBM với:
   * *Baseline 1*: Cùng giờ tuần trước (Same-Hour Last Week).
   * *Baseline 2*: Trung bình 3 tuần gần nhất (Rolling 3-Week Mean).

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Framework Rolling Backtest chạy tự động và xuất bảng so sánh chi tiết giữa Baseline 1, Baseline 2 và LightGBM.
- [ ] Mô hình LightGBM chứng minh cải thiện chỉ số rõ rệt: Hourly MAE < Baseline 1 và Peak Regret <= 1.5 người.
- [ ] Đầu ra dự báo luôn không âm (y_pred >= 0).
- [ ] Xử lý chính xác logic chọn Ngày đông nhất tuần và Giờ đông nhất từng ngày theo đúng đặc tả của bài toán.

---

### [MVP-09] MLflow Model Registry & Automated Gatekeeper

* **Issue ID**: `MVP-09`
* **Epic**: MLOps Lifecycle & Governance
* **Nguồn Requirement**: `idea.md` Mục 9.2, 9.3, 9.4, 9.5; `README.md` Mục 2.3, 4.1.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Triển khai **MLflow Tracking Server** kết nối backend RDS PostgreSQL và artifact store AWS S3. Xây dựng cổng kiểm duyệt tự động (**Automated Gatekeeper**) để kiểm soát việc chuyển đổi trạng thái mô hình trong **Model Registry**, đảm bảo mô hình kém chất lượng không bao giờ được đưa vào phục vụ sản xuất.

#### Phạm vi công việc (Scope)
1. Cấu hình MLflow Server trên AWS ECS / App Runner:
   * Backend Store URI: `postgresql://user:pass@rds-endpoint/mlflow_db`.
   * Artifact Store: `s3://smartlab-mlops-artifacts-<account_id>/mlflow-artifacts/`.
2. Quản lý 2 Registered Models riêng biệt:
   * `smartlab-person-detector`: Quản lý mô hình đếm người (Vision).
   * `smartlab-occupancy-forecaster`: Quản lý mô hình dự báo chuỗi thời gian.
3. Chuẩn hóa 4 trạng thái mô hình: `Candidate` -> `Validated` -> `Production` -> `Retired`.
4. Xây dựng module kiểm duyệt `src/mlops/gatekeeper.py`:
   * Khi một mô hình mới được huấn luyện (trạng thái `Candidate`):
   * Chạy đánh giá Rolling Backtest đối đầu với phiên bản đang giữ tag `Production`.
   * **Tiêu chí thăng hạng (Promotion Criteria)**:
     ```text
MAE(Candidate) < MAE(Production)  và  Peak Hour Accuracy(Candidate) >= Baseline
```
   * Nếu ĐẠT: Tự động chuyển Candidate thành `Validated` và gán alias/tag `Production` (phiên bản cũ chuyển thành `Retired`).
   * Nếu KHÔNG ĐẠT: Giữ nguyên mô hình Production hiện tại, ghi nhận log giải trình lý do từ chối và gửi cảnh báo qua SNS.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] MLflow UI hiển thị đầy đủ các runs, metrics, parameters và artifacts lưu trữ trên AWS S3.
- [ ] Unit test `tests/test_mlops.py` mô phỏng:
  - Một mô hình kém hơn bị Gatekeeper từ chối và không đổi tag Production.
  - Một mô hình tốt hơn được Gatekeeper tự động phê duyệt thăng hạng.
- [ ] Hai mô hình Vision và Forecasting được version hóa độc lập hoàn toàn.

---

### [MVP-10] Synchronized Safe Rollback Engine

* **Issue ID**: `MVP-10`
* **Epic**: MLOps Lifecycle & Reliability
* **Nguồn Requirement**: `idea.md` Mục 9.6, 11.2; `README.md` Mục 2.3, 8.3.
* **Thời lượng ước tính**: 2 ngày

#### Mô tả (Description)
Xây dựng cơ chế phục hồi phiên bản an toàn (**Safe Rollback Engine**) cho phép chuyển đổi ngược lại bộ phiên bản ổn định trước đó trong vòng chưa đầy 30 giây khi phát hiện mô hình mới phát sinh lỗi runtime hoặc suy giảm độ chính xác trên môi trường thực tế.

#### Phạm vi công việc (Scope)
1. Xây dựng module `src/mlops/rollback.py`:
   * Nhận tham số đầu vào: `--model-type [vision|forecast]` và `--target-version <version_number>`.
   * Kiểm tra tính toàn vẹn của artifacts phiên bản mục tiêu trên AWS S3 và MLflow Registry.
   * **Phục hồi đồng bộ 3 thành phần**:
     1. File trọng số mô hình (`model.onnx` hoặc `model.pkl`).
     2. Cấu hình trích xuất đặc trưng (`feature_version`).
     3. Tham số cấu hình tiền xử lý/làm mượt tương ứng.
   * Cập nhật tag `Production` trong MLflow Model Registry về phiên bản đích.
   * Không xóa lịch sử dự báo cũ hay bản ghi lỗi (để giữ nguyên audit trail phục vụ điều tra).
2. Xây dựng API endpoint nội bộ `/api/v1/mlops/rollback` có bảo mật bằng API Key.
3. Tích hợp lệnh rollback với GitHub Actions Workflow Dispatch để người vận hành có thể thao tác bằng 1 cú nhấp chuột.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Chạy lệnh `python -m src.mlops.rollback --model-type forecast --target-version 1` hoàn tất trong vòng <= 30 giây.
- [ ] Service API nhận diện ngay lập tức mô hình đã rollback mà không cần khởi động lại toàn bộ database.
- [ ] Gửi thông báo thành công qua Amazon SNS thông báo phiên bản vừa được khôi phục.

---

### [MVP-11] CI/CD & MLOps Workflows trên GitHub Actions (CML & OIDC)

* **Issue ID**: `MVP-11`
* **Epic**: CI/CD & Automation
* **Nguồn Requirement**: `idea.md` Mục 9.1, 9.3; `README.md` Mục 4.2.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Xây dựng 3 GitHub Actions Workflows hoàn chỉnh để tự động hóa toàn bộ vòng lặp kỹ thuật phần mềm và máy học: (1) CI/CD kiểm thử mã nguồn và triển khai container lên AWS; (2) MLOps CML tự động đánh giá mô hình trên Pull Request; (3) Workflow Dispatch kích hoạt Rollback khẩn cấp.

#### Phạm vi công việc (Scope)
1. **Workflow 1: CI/CD Pipeline (`.github/workflows/ci-cd.yml`)**:
   * Kích hoạt khi có commit/PR vào nhánh `main`.
   * Chạy `ruff check`, `black --check`, `mypy src`.
   * Chạy toàn bộ test suites `pytest --cov=src` (yêu cầu coverage >= 80%).
   * Build Docker images đa tầng tối ưu hóa dung lượng: `smartlab-worker`, `smartlab-forecast`, `smartlab-dashboard`.
   * Xác thực với AWS qua **GitHub OIDC Role** và đẩy images lên **AWS ECR**.
   * Cập nhật Task Definition và kích hoạt deploy lên AWS ECS Fargate.
2. **Workflow 2: MLOps Model Evaluation & CML (`.github/workflows/model-eval.yml`)**:
   * Kích hoạt khi PR có thay đổi trong `src/forecasting/`, `src/vision/` hoặc file `.dvc`.
   * Tải snapshot dữ liệu từ S3 bằng `dvc pull`.
   * Chạy script đánh giá so sánh Candidate vs Production.
   * Sử dụng công cụ **CML (Continuous Machine Learning)** để sinh biểu đồ trực quan hóa sai số (Residual Plots, MAE per Hour).
   * **Bình luận tự động báo cáo Markdown kèm ảnh trực tiếp vào Pull Request**.
3. **Workflow 3: Emergency Rollback (`.github/workflows/rollback.yml`)**:
   * Kích hoạt thủ công qua `workflow_dispatch` với input chọn service và target version.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Cả 3 workflows chạy thành công trên GitHub Actions Runner mà không gặp lỗi phân quyền AWS.
- [ ] Pull Request thay đổi mô hình tự động nhận được bình luận báo cáo CML kèm biểu đồ so sánh.
- [ ] Docker container build sạch sẽ, quét bảo mật bằng Trivy không có lỗ hổng nghiêm trọng (Critical Vulnerabilities).

---

### [MVP-12] Dashboard 4 phân hệ & API Serving (FastAPI + Streamlit)

* **Issue ID**: `MVP-12`
* **Epic**: Serving & User Interface
* **Nguồn Requirement**: `idea.md` Mục 12.1, 12.2, 12.3, 12.4; `README.md` Mục 2.3, 4.1.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Xây dựng dịch vụ backend API bằng **FastAPI** và giao diện trực quan tương tác bằng **Streamlit** (hoặc Web Dashboard) bao gồm đầy đủ 4 phân hệ theo chuẩn đặc tả của SmartLab, phục vụ cả người quản lý phòng CLB và đội ngũ kỹ thuật vận hành.

#### Phạm vi công việc (Scope)
1. Xây dựng dịch vụ RESTful API bằng **FastAPI** (`src/web/app.py`):
   * `GET /api/v1/live`: Lấy số người hiện diện gần nhất, thời điểm cập nhật cuối, trạng thái camera và cờ độ mới dữ liệu.
   * `GET /api/v1/history`: Lấy chuỗi thời gian occupancy theo giờ với các bộ lọc ngày/tuần, kèm cờ đánh dấu dữ liệu hợp lệ.
   * `GET /api/v1/forecast/latest`: Lấy dự báo 84 giờ của tuần tới, ngày đông nhất, giờ đông nhất từng ngày và mã lần chạy.
   * `GET /api/v1/mlops/telemetry`: Lấy phiên bản mô hình đang phục vụ, kết quả đánh giá và trạng thái pipeline.
2. Xây dựng giao diện tương tác **Streamlit** (`src/web/dashboard.py`) gồm 4 màn hình:
   * **Màn hình 1 — Trạng thái hiện tại**: Thẻ số liệu lớn (Live Occupancy), trạng thái kết nối Camera (Online / Stale / Offline), thời gian cập nhật.
   * **Màn hình 2 — Lịch sử sử dụng phòng**: Biểu đồ đường chuỗi thời gian occupancy; phân biệt trực quan giữa giờ đóng cửa và giờ bị thiếu dữ liệu camera.
   * **Màn hình 3 — Dự báo tuần tới**: Bảng Heatmap 7 ngày x các giờ mở cửa; làm nổi bật **Ngày đông nhất tuần** và **Giờ cao điểm từng ngày**; cảnh báo nếu đang dùng mô hình fallback baseline.
   * **Màn hình 4 — Vận hành kỹ thuật (MLOps Operations)**: Hiển thị Model Version hiện tại, bảng so sánh MAE các lần chạy, logs hệ thống và nút kích hoạt Manual Re-run hoặc Rollback.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Dashboard kết nối mượt mà với FastAPI backend và CSDL PostgreSQL.
- [ ] Giao diện hiển thị chuẩn xác múi giờ `Asia/Ho_Chi_Minh` (UTC+7).
- [ ] Khi camera mất kết nối, màn hình Live Status chuyển ngay sang màu đỏ hiển thị `CAMERA OFFLINE` và thời điểm cập nhật cuối (không hiển thị số 0).
- [ ] Render Heatmap dự báo rõ ràng, trực quan, dễ hiểu đối với người quản lý phòng CLB không chuyên kỹ thuật.

---

### [MVP-13] Nghiệm thu 6 Kịch bản Bắt buộc & Bàn giao Hệ thống

* **Issue ID**: `MVP-13`
* **Epic**: Quality Assurance & Project Delivery
* **Nguồn Requirement**: `idea.md` Mục 16.4; `README.md` Mục 2.3, 6.3.
* **Thời lượng ước tính**: 3 ngày

#### Mô tả (Description)
Thực hiện chạy thử nghiệm toàn diện hệ thống trên môi trường AWS Production, ghi lại video/logs minh chứng và nghiệm thu vượt qua 100% cả **6 kịch bản thực nghiệm bắt buộc** được quy định trong tài liệu thiết kế.

#### Phạm vi công việc (Scope)
Thực hiện và tài liệu hóa 6 kịch bản kiểm thử:
1. **Kịch bản 1 (Normal Operation)**: Camera hoạt động bình thường, hệ thống đếm người trong phòng kín, làm mượt và cập nhật số người hiện diện lên dashboard mỗi 5 giây; sau 60 phút tổng hợp đầy đủ 1 bản ghi vào `hourly_occupancy`.
2. **Kịch bản 2 (Camera Disconnect Handling)**: Ngắt luồng camera mô phỏng. Dashboard lập tức hiển thị trạng thái `CAMERA OFFLINE` kèm thời điểm cập nhật cuối; **tuyệt đối không hiển thị 0 người** và không ghi số 0 vào database.
3. **Kịch bản 3 (Missing Data Handling)**: Mô phỏng mất tín hiệu camera 15 phút trong khung giờ (độ phủ chỉ đạt 75% < 90%). Giờ đó tự động bị gán `is_eligible = FALSE` và bị loại khỏi tập huấn luyện mô hình dự báo.
4. **Kịch bản 4 (Scheduled Forecast Execution)**: Chạy job dự báo với mốc cutoff 23:00 Chủ Nhật. Kết quả dự báo 84 giờ tuần kế tiếp được lưu vào CSDL; Dashboard hiển thị bảng Heatmap, chỉ ra chính xác Ngày đông nhất và Giờ cao điểm từng ngày.
5. **Kịch bản 5 (Model Deployment Gatekeeping)**: Huấn luyện một mô hình thử nghiệm bị underfitting hoặc overfitting. Pipeline kiểm thử tự động so sánh metric với Baseline và Production hiện tại, **từ chối chuyển trạng thái sang Production** và ghi lại báo cáo giải trình.
6. **Kịch bản 6 (Safe Rollback)**: Kích hoạt lệnh Rollback thông qua GitHub Actions / CLI. Hệ thống tự động phục hồi đồng bộ: artifact mô hình cũ, cấu hình tạo đặc trưng và version tiền xử lý mà không làm gián đoạn API serving hay xóa lịch sử dự báo cũ.
7. Soạn thảo tài liệu bàn giao `docs/HANDOVER_GUIDE.md` và hướng dẫn vận hành production.

#### Tiêu chí nghiệm thu (Acceptance Criteria)
- [ ] Cả 6 kịch bản đều có video quay màn hình hoặc nhật ký log chi tiết chứng minh hệ thống phản ứng chính xác theo yêu cầu.
- [ ] Hệ thống vận hành liên tục tối thiểu 7 ngày trên AWS không phát sinh lỗi nghiêm trọng (Zero Critical Downtime).
- [ ] Toàn bộ checklist **Definition of Done của MVP** được nghiệm thu và ký duyệt bàn giao.

---

## 3. Bảng phân công trách nhiệm & Kế hoạch tiến độ MVP (WBS & RACI Matrix)

| Issue ID | Tên Issue / Hạng mục | Người phụ trách chính (R) | Người phối hợp (C) | Thời gian dự kiến | Phụ thuộc |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`MVP-01`** | Thiết lập Hạ tầng AWS (Terraform, S3, RDS, ECS, OIDC) | **DevOps / Cloud Eng** | MLOps Lead | Tuần 1 (Ngày 1–3) | Không |
| **`MVP-02`** | Thiết kế Database Schema & Migration (PostgreSQL) | **Backend Engineer** | Data Engineer | Tuần 1 (Ngày 3–4) | `MVP-01` |
| **`MVP-03`** | RTSP Stream Ingestion & Resilient Camera Worker | **CV Engineer** | Backend Engineer | Tuần 1 (Ngày 4–5) | `MVP-02` |
| **`MVP-04`** | Full-Frame Detection Engine (ONNX) & Median Filter | **CV Engineer** | Data Engineer | Tuần 2 (Ngày 6–8) | `MVP-03` |
| **`MVP-05`** | Data Aggregator: Tổng hợp Hourly Occupancy (90%) | **Data Engineer** | Backend Engineer | Tuần 2 (Ngày 8–9) | `MVP-04` |
| **`MVP-06`** | Data Management & Versioning với DVC & AWS S3 | **MLOps Engineer** | Data Engineer | Tuần 2 (Ngày 9–10)| `MVP-01` |
| **`MVP-07`** | Feature Engineering chống rò rỉ dữ liệu (Cutoff 23:00) | **Forecasting Eng** | Data Engineer | Tuần 3 (Ngày 11–13)| `MVP-05`, `MVP-06` |
| **`MVP-08`** | Huấn luyện LightGBM & Framework Rolling Backtest | **Forecasting Eng** | MLOps Engineer | Tuần 3 (Ngày 13–15)| `MVP-07` |
| **`MVP-09`** | MLflow Model Registry & Automated Gatekeeper | **MLOps Engineer** | Forecasting Eng | Tuần 4 (Ngày 16–18)| `MVP-08` |
| **`MVP-10`** | Synchronized Safe Rollback Engine | **MLOps Engineer** | Backend Engineer | Tuần 4 (Ngày 18–19)| `MVP-09` |
| **`MVP-11`** | CI/CD & MLOps Workflows trên GitHub Actions (CML) | **DevOps / MLOps Eng** | Cả nhóm | Tuần 4 (Ngày 19–20)| `MVP-01`, `MVP-09` |
| **`MVP-12`** | Dashboard 4 phân hệ & API Serving (FastAPI/Streamlit) | **Frontend / Backend** | QA / Tester | Tuần 5 (Ngày 21–23)| `MVP-05`, `MVP-08`, `MVP-10` |
| **`MVP-13`** | Nghiệm thu 6 Kịch bản Bắt buộc & Bàn giao Hệ thống | **Cả nhóm (Lead: QA)** | Stakeholders | Tuần 5 (Ngày 24–25)| Toàn bộ các Issues |

*(Tổng thời gian thực hiện ước tính cho giai đoạn MVP: **5 tuần làm việc**)*

---

## 4. Ma trận kiểm thử nghiệm thu MVP (Acceptance Test Matrix)

| Test Case ID | Kịch bản kiểm thử | Mô tả hành vi & Dữ liệu đưa vào | Kết quả mong đợi chuẩn xác |
| :--- | :--- | :--- | :--- |
| **`TC-MVP-01`** | Xác thực GitHub Actions qua AWS OIDC | Kích hoạt GitHub Actions CI chạy job deploy. | Xác thực thành công bằng IAM Role OIDC; không dùng access key tĩnh; push image lên ECR thành công. |
| **`TC-MVP-02`** | Xử lý ngắt luồng Camera thật | Ngắt cáp mạng camera IP trong 5 phút. | Worker không crash; CSDL ghi nhận `CAMERA_OFFLINE`; dashboard đổi cờ đỏ; **không ghi số 0 vào database**. |
| **`TC-MVP-03`** | Tự động phục hồi kết nối Camera | Cắm lại cáp mạng camera sau khi ngắt. | Worker tự động kết nối lại sau vài giây bằng Exponential Backoff; tiếp tục đếm và ghi số liệu bình thường. |
| **`TC-MVP-04`** | Xử lý khung giờ thiếu dữ liệu | Giả lập khung hình lỗi liên tục trong 15 phút (valid < 90%). | Tác vụ tổng hợp gán `is_eligible = FALSE`; cờ này ngăn không cho giờ đó vào tập huấn luyện của Pipeline B. |
| **`TC-MVP-05`** | Ngăn ngừa rò rỉ dữ liệu dự báo | Kích hoạt job dự báo tuần với `data_cutoff = 23:00 Chủ Nhật`. | Tuyệt đối không có bất kỳ đặc trưng nào sử dụng dữ liệu sau 23:00 Chủ Nhật; xuất dự báo đúng 84 giờ tuần tới. |
| **`TC-MVP-06`** | Thử nghiệm Cổng kiểm duyệt Gatekeeper | Đưa vào mô hình Candidate có MAE = 3.5 trong khi Production đang là MAE = 2.1. | Gatekeeper từ chối thăng hạng; giữ nguyên mô hình Production; gửi cảnh báo qua Amazon SNS. |
| **`TC-MVP-07`** | Thử nghiệm Rollback an toàn | Kích hoạt lệnh rollback khẩn cấp về version trước qua GitHub Actions. | Hệ thống khôi phục đồng bộ model + feature config trong < 30 giây; API tiếp tục phục vụ bình thường. |
| **`TC-MVP-08`** | Vận hành liên tục 7 ngày (Endurance Test) | Chạy toàn bộ hệ thống liên tục trên AWS trong 7 ngày thực tế. | Zero crash; dự báo tuần tự động sinh vào 23:00 Chủ Nhật; dashboard hoạt động mượt mà không quá tải RAM/CPU. |
