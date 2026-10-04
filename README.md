# Hệ thống phát hiện khói và lửa từ video (Xử lý ảnh truyền thống)

Dự án Bài tập lớn môn Xử lý ảnh (PTIT): **Phát hiện khói hoặc lửa từ video**.

Mục tiêu chính: Thể hiện năng lực xử lý ảnh truyền thống (Computer Vision cổ điển), module hóa chuẩn mực để sẵn sàng mở rộng sang Học sâu (Deep Learning).

---

## 📌 Các kỹ thuật xử lý ảnh cốt lõi

1. **Phân tích màu sắc (Color Modeling)**:
   - **Lửa**: Kết hợp mô hình màu RGB (Chen et al., 2004: $R > R_T$, $R \ge G > B$, bão hòa theo độ sáng) và YCbCr (Çelik & Demirel, 2009: phân tách độ sáng Y và sắc độ Cr, Cb).
   - **Khói**: Khói xám có độ bão hòa thấp trong HSV ($S \le S_{max}$), sai khác giữa các kênh RGB nhỏ ($|R-G|, |G-B|, |R-B| \le \alpha$) và độ sáng nằm trong dải xám.
2. **Phát hiện chuyển động (Motion Detection)**:
   - Tách nền bằng Gaussian Mixture Models (MOG2 / KNN) kết hợp chạy hiệu 3 frame liên tiếp.
   - Kết hợp mặt nạ màu và mặt nạ chuyển động ($\text{Mask}_{color} \cap \text{Mask}_{motion}$) để loại bỏ vật thể đỏ tĩnh (biển báo, tường đỏ) hoặc khói tĩnh.
   - Ước lượng vector chuyển động (Optical Flow Farnebäck) đo hướng di chuyển (khói có xu hướng bốc lên).
3. **Phân tích biến đổi hình dạng & thời gian (Temporal & Shape Analysis)**:
   - **Flicker**: Đếm chu kỳ đổi trạng thái pixel ngẫu nhiên và phân tích năng lượng phổ FFT / Biến đổi Wavelet 1D ở tần số 3–13 Hz.
   - **Biến dạng**: Hệ số biến thiên diện tích $\sigma/\mu$, độ bất thường biên (Circularity, Roughness, Hu Moments).
   - **Đặc trưng khói**: Đo độ mờ biên (suy giảm năng lượng wavelet tần số cao so với nền) và độ mịn bề mặt bằng Local Binary Patterns (LBP).
4. **Lọc phát hiện sai (False Positive Rejection)**:
   - Toán tử hình thái học (Morphology: Opening khử nhiễu đốm, Closing hàn gắn vùng).
   - Ghép vết đối tượng (Region Tracking qua IoU).
   - Bỏ phiếu thời gian (Sliding Window Persistence): đối tượng phải duy trì qua nhiều khung hình.
   - Hàm chấm điểm tổng hợp đa đặc trưng (Multi-feature scoring).

---

## 📁 Cấu trúc thư mục

```text
XLA/
├── configs/
│   └── default.yaml            # Toàn bộ cấu hình ngưỡng (RGB, YCbCr, MOG2, Scoring...)
├── data/
│   ├── raw/                    # Video thô thu thập từ dataset / quay thực tế (không commit)
│   ├── samples/                # Video ngắn phục vụ demo / test nhanh
│   └── annotations/            # Nhãn ground truth định dạng JSON theo frame
├── src/
│   ├── config.py               # Module tải và nạp đè cấu hình YAML
│   ├── structures.py           # Định nghĩa cấu trúc Region, Detection, BBox, IoU
│   ├── video_io/               # Đọc và ghi video tuần tự (VideoReader, VideoWriter)
│   ├── preprocessing/          # Resize, làm mờ Gaussian, chuyển đổi RGB/HSV/YCrCb
│   ├── color/                  # Luật lọc màu lửa (Chen, Celik) và khói
│   ├── motion/                 # Tách nền MOG2/KNN và Optical Flow
│   ├── tracking/               # Theo dõi liên kết vùng qua thời gian (RegionTracker)
│   ├── temporal/               # Phân tích Flicker (FFT, Wavelet), biến thiên diện tích, LBP
│   ├── filtering/              # Morphology, lọc hình học, persistence và hàm chấm điểm
│   ├── detectors/
│   │   ├── base.py             # Interface trừu tượng BaseDetector (cầu nối mở rộng DL)
│   │   ├── classical.py        # Pipeline xử lý ảnh truyền thống hoàn chỉnh
│   │   └── deep.py             # Chỗ dành sẵn cho YOLO / CNN xác minh (Giai đoạn sau)
│   ├── evaluation/             # Đo Precision, Recall, F1, FAR, độ trễ phát hiện
│   └── visualization/          # Vẽ bounding box, ghép lưới ảnh trung gian (Debug Grid)
├── app/
│   └── streamlit_app.py        # Giao diện trực quan Streamlit (tinh chỉnh slider, xem mask)
├── scripts/
│   ├── check_env.py            # Kiểm tra môi trường thư viện
│   ├── make_synthetic_video.py # Sinh video mẫu giả lập để kiểm thử pipeline
│   ├── run_video.py            # CLI chạy phát hiện trên video / webcam
│   ├── run_ablation.py         # Thực nghiệm bóc tách (Ablation Study) từng tầng
│   └── analyze_color.py        # Trích xuất và vẽ biểu đồ phân bố màu Cb-Cr, H-S
├── tests/                      # Bộ kiểm thử tự động pytest cho từng module
├── requirements.txt            # Danh sách thư viện Python
└── pytest.ini                  # Cấu hình kiểm thử
```

---

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Kích hoạt môi trường ảo

Dự án đã được cấu hình môi trường ảo Python 3.12 tại `.venv`:

```powershell
# Kích hoạt môi trường (trên Windows PowerShell):
.\.venv\Scripts\Activate.ps1

# Hoặc chạy trực tiếp với:
.\.venv\Scripts\python.exe <tên_script>.py
```

### 2. Kiểm tra môi trường & chạy unit tests

```powershell
.\.venv\Scripts\python.exe scripts/check_env.py
.\.venv\Scripts\python.exe -m pytest -v
```

### 3. Sinh video tổng hợp kiểm thử

Nếu chưa có video dữ liệu thực tế, sinh video mẫu kiểm thử (bao gồm ngọn lửa nhấp nháy, cột khói mờ lan tỏa, xe xám và đèn đỏ di chuyển để kiểm tra lọc sai):

```powershell
.\.venv\Scripts\python.exe scripts/make_synthetic_video.py
```

### 4. Chạy trên video / webcam (CLI)

```powershell
# Chạy video mẫu và hiển thị lưới debug mask trung gian:
.\.venv\Scripts\python.exe scripts/run_video.py --input data/samples/synthetic_fire_smoke.mp4 --show --debug

# Chạy với Webcam máy tính:
.\.venv\Scripts\python.exe scripts/run_video.py --input 0 --show --debug

# Xuất kết quả ra file video:
.\.venv\Scripts\python.exe scripts/run_video.py --input data/samples/synthetic_fire_smoke.mp4 --output outputs/result.mp4 --debug
```

### 5. Giao diện trực quan tương tác (Streamlit)

Giao diện cho phép xem trực quan từng khung hình, các mask màu, mask chuyển động, bản đồ flicker và chỉnh các ngưỡng (threshold) bằng slider theo thời gian thực:

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

### 6. Chạy đánh giá bóc tách (Ablation Study)

Chứng minh đóng góp khoa học của từng tầng thuật toán (chỉ màu -> màu + chuyển động -> + persistence -> + đặc trưng thời gian):

```powershell
.\.venv\Scripts\python.exe scripts/run_ablation.py --videos data/samples --annotations data/annotations
```

---

## 👥 Phân công công việc nhóm (4 thành viên)

| Thành viên | Trách nhiệm chính | Module phụ trách |
|---|---|---|
| **Thành viên 1** (Trưởng nhóm) | Tích hợp hệ thống, Kiến trúc luồng, Demo Streamlit | `src/video_io`, `src/preprocessing`, `src/detectors/classical.py`, `app/` |
| **Thành viên 2** | Mô hình hóa & Phân tích màu sắc, Thu thập dữ liệu | `src/color/`, `scripts/analyze_color.py`, thu thập dataset Bilkent/MIVIA |
| **Thành viên 3** | Chuyển động & Phân tích thời gian (Flicker, Wavelet) | `src/motion/`, `src/tracking/`, `src/temporal/flicker.py` |
| **Thành viên 4** | Đặc trưng khói, Lọc phát hiện sai & Đánh giá thực nghiệm | `src/temporal/shape_features.py`, `src/temporal/smoke_features.py`, `src/filtering/`, `src/evaluation/` |

---

## 🔮 Hướng mở rộng sang Học sâu (Deep Learning)

Kiến trúc đã chuẩn bị sẵn lớp trừu tượng `BaseDetector`:
1. **Mô hình lai (Hybrid approach - Khuyến nghị)**: Giữ nguyên toàn bộ pipeline xử lý ảnh truyền thống để trích xuất các vùng ứng viên (Candidate Proposals) với tốc độ cao, sau đó chuyển vùng crop vào mạng nơ-ron tích chập nhẹ (MobileNetV3 hoặc SqueezeNet) để phân loại nhị phân (Fire / Non-fire, Smoke / Non-smoke). Cách này vừa giữ trọn vẹn điểm kỹ thuật XLA vừa hạn chế tối đa báo động sai.
2. **End-to-End Object Detection**: Huấn luyện YOLOv8 / YOLO11 trên tập dữ liệu D-Fire và so sánh hiệu năng, độ trễ và số lượng tham số với pipeline cổ điển.
