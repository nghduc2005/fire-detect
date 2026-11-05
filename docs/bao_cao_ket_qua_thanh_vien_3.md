# BÁO CÁO KẾT QUẢ ĐẦU RA — THÀNH VIÊN 3
## HỆ THỐNG PHÁT HIỆN KHÓI VÀ LỬA TỪ VIDEO BẰNG XỬ LÝ ẢNH TRUYỀN THỐNG
### Module: Phát Hiện Chuyển Động · Theo Dõi Vùng (Tracking) · Phân Tích Nhấp Nháy (Flicker)

> **Học phần:** Xử lý ảnh (XLA) — Học viện Công nghệ Bưu chính Viễn thông (PTIT)  
> **Dự án:** Hệ thống giám sát, phát hiện sớm khói và lửa qua Camera / Video  
> **Vai trò:** Thành viên 3 — Kỹ sư Thị giác máy tính & Xử lý tín hiệu thời gian  
> **Thời gian thực hiện:** Tháng 11/2026  
> **Trạng thái hoàn thành:** **Đạt 100% mục tiêu theo kế hoạch triển khai (Đạt 58/58 test cases PASSED)**

---

## MỤC LỤC
1. [Tổng quan vai trò và phân tích kế hoạch triển khai](#1-tổng-quan-vai-trò-và-phân-tích-kế-hoạch-triển-khai)
2. [Chi tiết các giai đoạn kỹ thuật đã triển khai](#2-chi-tiết-các-giai-đoạn-kỹ-thuật-đã-triển-khai)
   - [2.1. Giai đoạn 1: Bộ trừ nền phát hiện chuyển động (MotionDetector)](#21-giai-đoạn-1-bộ-trừ-nền-phát-hiện-chuyển-động-motiondetector)
   - [2.2. Giai đoạn 2: Bộ theo dõi liên kết vùng qua thời gian (RegionTracker)](#22-giai-đoạn-2-bộ-theo-dõi-liên-kết-vùng-qua-thời-gian-regiontracker)
   - [2.3. Giai đoạn 3: Phân tích trường chuyển động khói (Farnebäck Optical Flow)](#23-giai-đoạn-3-phân-tích-trường-chuyển-động-khói-farnebäck-optical-flow)
   - [2.4. Giai đoạn 4: Phân tích dao động nhấp nháy lửa (Flicker Analysis & FFT)](#24-giai-đoạn-4-phân-tích-dao-động-nhấp-nháy-lửa-flicker-analysis--fft)
   - [2.5. Giai đoạn 5: Tích hợp hoàn hảo & Bộ kiểm thử tự động (Unit Tests)](#25-giai-đoạn-5-tích-hợp-hoàn-hảo--bộ-kiểm-thử-tự-động-unit-tests)
3. [Danh mục sản phẩm & Mã nguồn bàn giao](#3-danh-mục-sản-phẩm--mã-nguồn-bàn-giao)
4. [Kết quả thực nghiệm & Kiểm thử chất lượng](#4-kết-quả-thực-nghiệm--kiểm-thử-chất-lượng)
5. [Đóng góp kỹ thuật & Tương tác trong nhóm](#5-đóng-góp-kỹ-thuật--tương-tác-trong-nhóm)

---

## 1. TỔNG QUAN VAI TRÒ VÀ PHÂN TÍCH KẾ HOẠCH TRIỂN KHAI

### 1.1. Phân tích bối cảnh hệ thống
Trong hệ thống phát hiện khói lửa cổ điển, bài toán khó nhất là **giảm thiểu tỉ lệ báo động giả (False Alarm Rate - FAR)** gây ra bởi:
- Các vật thể màu đỏ/vàng tĩnh (biển báo giao thông, tường đỏ, hoa văn tĩnh).
- Các vật thể màu đỏ chuyển động khối cứng (áo đỏ của người đi bộ, xe cứu thương, đèn xe, rèm cửa bay).
- Mây bay, bóng râm hay sương mù xám không phải là khói.

**Thành viên 3** chịu trách nhiệm xây dựng "bộ não" phân tích động học và chuỗi thời gian, chuyển đổi hệ thống từ việc chỉ nhận diện đơn lẻ từng khung hình thành một hệ thống **quan sát hành vi qua thời gian**:
1. **Phát hiện chuyển động:** Tách vùng động khỏi môi trường nền thích nghi bằng MOG2/KNN và hiệu 3 khung hình.
2. **Theo dõi liên kết vùng (Tracking):** Ghép cặp nhận diện qua IoU tham lam (Greedy IoU Matching), duy trì định danh `track_id` ổn định và tích lũy bộ đệm lịch sử (`history`).
3. **Phân tích nhấp nháy (Flicker):** Khai thác cơ chế dao động hỗn loạn tự nhiên của ngọn lửa (3–13 Hz) qua biến đổi Fourier nhanh (FFT) và hệ số biến thiên cường độ (CV).
4. **Phân tích hướng chuyển động khói (Optical Flow):** Đo lường xu hướng khuếch tán bốc lên trên ($dy < 0$) của cột khói.

### 1.2. Cam kết nguyên tắc không xâm lấn module
Quá trình triển khai tuân thủ tuyệt đối quy tắc kiến trúc của dự án:
- **Không thay đổi** bất kỳ mã nguồn nào của TV1 (`src/detectors/classical.py`, `src/preprocessing/`, `src/video_io/`, `src/visualization/`, `src/config.py`).
- **Không thay đổi** module của TV2 (`src/color/`).
- **Không thay đổi** module của TV4 (`src/filtering/`).
- **Giữ nguyên 100% signature interface stub**: Nhờ vậy, pipeline tích hợp sẵn của TV1 tự động kích hoạt mã nguồn thật của TV3 mà không cần chỉnh sửa bất kỳ dòng code điều phối nào.

---

## 2. CHI TIẾT CÁC GIAI ĐOẠN KỸ THUẬT ĐÃ TRIỂN KHAI

### 2.1. Giai đoạn 1: Bộ trừ nền phát hiện chuyển động (MotionDetector)
- **Tệp triển khai:** [`src/motion/__init__.py`](file:///d:/PTIT/XLA/fire_detect/src/motion/__init__.py)
- **Kỹ thuật cốt lõi:**
  - **Gaussian Mixture Models (MOG2):** Đóng gói `cv2.createBackgroundSubtractorMOG2` thích ứng với biến đổi ánh sáng môi trường nền. Cơ chế nhị phân hóa ngưỡng 128 khử sạch bóng xám (shadow = 127), xuất mask `uint8` {0, 255} chuẩn xác.
  - **K-Nearest Neighbors (KNN):** Hỗ trợ thuật toán trừ nền KNN cho các tình huống nhiễu nền phức tạp (`dist2_threshold`).
  - **Three-frame Differencing:** Triển khai phương pháp hiệu 3 khung hình liên tiếp $|f(t) - f(t-1)| \cap |f(t-1) - f(t-2)|$ với ngưỡng `diff_threshold`. Phương pháp này triệt tiêu hoàn toàn hiệu ứng bóng mờ (ghost artifacts) mà hiệu 2 khung hình thường gặp phải.
  - **Cơ chế reset trạng thái:** Giải phóng bộ nhớ đệm khung hình và tái khởi tạo subtractor sạch sẽ khi bắt đầu chuỗi video mới.

```python
# Trích đoạn cơ chế nhị phân hóa MOG2 an toàn
learning_rate = float(self.cfg.get("learning_rate", -1))
fg_mask = self._subtractor.apply(bundle.bgr, learningRate=learning_rate)
_, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
return fg_mask.astype(np.uint8)
```

---

### 2.2. Giai đoạn 2: Bộ theo dõi liên kết vùng qua thời gian (RegionTracker)
- **Tệp triển khai:** [`src/tracking/region_tracker.py`](file:///d:/PTIT/XLA/fire_detect/src/tracking/region_tracker.py), [`src/tracking/__init__.py`](file:///d:/PTIT/XLA/fire_detect/src/tracking/__init__.py)
- **Kỹ thuật cốt lõi:**
  - **Greedy IoU Matching:** Tính toán ma trận độ trùng khớp Intersection-over-Union (IoU) giữa bounding box của các tracks trước đó và các regions mới trích xuất từ khung hình hiện tại.
  - **Cấp phát & Duy trì `track_id`:** Ghép cặp ưu tiên tham lam từ cặp có IoU cao nhất xuống thấp nhất (vượt ngưỡng `iou_threshold = 0.1` để chấp nhận lửa biến thiên hình dạng mạnh). Vùng mới chưa từng xuất hiện được cấp phát ID tăng tiến duy nhất (`itertools.count`).
  - **Bộ đệm lịch sử (`deque` có giới hạn `buffer_size`):** Lưu trữ tuần tự các snapshot khung hình gồm `{bbox, area, centroid}` và độ sáng trung bình `intensity_mean`.
  - **Quản lý vòng đời track (Track Lifecycle Management):** Đếm số khung hình liên tiếp bị mất dấu (`_missed`). Nếu vượt quá `max_missed = 5`, track tự động bị hủy để giải phóng bộ nhớ.

---

### 2.3. Giai đoạn 3: Phân tích trường chuyển động khói (Farnebäck Optical Flow)
- **Tệp triển khai:** [`src/motion/optical_flow.py`](file:///d:/PTIT/XLA/fire_detect/src/motion/optical_flow.py)
- **Kỹ thuật cốt lõi:**
  - **Dense Optical Flow (Farnebäck, 2003):** Ước lượng trường vector vận tốc 2 chiều $(dx, dy)$ trên toàn bộ ma trận điểm ảnh thông qua phép xấp xỉ đa thức bậc 2 trong lân cận điểm ảnh.
  - **Chỉ số chuyển động hướng lên (`compute_upward_motion`):** Trong hệ tọa độ ảnh OpenCV, trục thẳng đứng $y$ có chiều dương hướng xuống dưới. Do đó, hiện tượng khói bốc lên trên tương ứng với vector có thành phần dọc $dy < -0.1\text{ px/frame}$. Hàm tự động cắt ROI vùng quan tâm và tính tỉ lệ pixel hướng lên:
    $$\text{Upward Ratio} = \frac{\sum_{(x, y) \in \text{ROI}} \mathbb{I}(dy(x, y) < -0.1)}{N_{\text{ROI}}} \in [0.0, 1.0]$$
  - Module này đóng vai trò giao diện sẵn sàng để Thành viên 4 kết nối trực tiếp vào `SmokeTemporalAnalyzer`.

---

### 2.4. Giai đoạn 4: Phân tích dao động nhấp nháy lửa (Flicker Analysis & FFT)
- **Tệp triển khai:** [`src/temporal/flicker.py`](file:///d:/PTIT/XLA/fire_detect/src/temporal/flicker.py), [`src/temporal/__init__.py`](file:///d:/PTIT/XLA/fire_detect/src/temporal/__init__.py)
- **Kỹ thuật cốt lõi (theo mô hình Töreyin et al., 2006):**
  - **Pixel/Area Flicker (`compute_pixel_flicker`):** Theo dõi chuỗi biến thiên diện tích của ngọn lửa trong cửa sổ trượt $N = 25$ khung hình. Đếm tần suất đổi dấu đạo hàm bậc nhất của diện tích $\text{sign}(\Delta \text{Area}_t) \ne \text{sign}(\Delta \text{Area}_{t-1})$ để phát hiện chuyển động rung mép đặc thù của ngọn lửa.
  - **Intensity Flicker qua hệ số biến thiên (`compute_intensity_flicker`):**
    Tính tỉ số giữa độ lệch chuẩn và giá trị kỳ vọng độ sáng trung bình $CV = \frac{\sigma_I}{\mu_I}$. Ngọn lửa bập bùng có độ biến thiên ánh sáng cục bộ rất cao, trong khi áo đỏ hay xe di chuyển có bề mặt chiếu sáng đồng đều ($CV \approx 0$).
  - **Biến đổi Fourier nhanh (FFT Flicker - `compute_fft_flicker`):**
    Khử thành phần một chiều (zero-mean detrending), sau đó tính phổ mật độ công suất $|X(f)|^2$ bằng thuật toán `np.fft.rfft`. Tách và tính tỉ lệ năng lượng tập trung trong dải tần số đặc trưng của ngọn lửa $3\text{ Hz} \le f \le 13\text{ Hz}$:
    $$\text{FFT Ratio} = \frac{\int_{3\text{ Hz}}^{13\text{ Hz}} |X(f)|^2 df}{\int_0^{f_{max}} |X(f)|^2 df}$$
  - **Tích hợp `FireTemporalAnalyzer`:** Đo giá trị `intensity_mean` trên kênh xám của ROI và ghi trực tiếp vào `region.extras` cho `FeatureScorer` tính điểm trọng số.

---

### 2.5. Giai đoạn 5: Tích hợp hoàn hảo & Bộ kiểm thử tự động (Unit Tests)
- **Tệp kiểm thử:**
  - [`tests/test_motion.py`](file:///d:/PTIT/XLA/fire_detect/tests/test_motion.py): 5 test cases kiểm thử MOG2, KNN, 3-frame diff, reset và Optical Flow.
  - [`tests/test_tracking.py`](file:///d:/PTIT/XLA/fire_detect/tests/test_tracking.py): 6 test cases kiểm thử IoU, tracking ID ổn định, tích lũy lịch sử, loại bỏ stale tracks và reset.
  - [`tests/test_flicker.py`](file:///d:/PTIT/XLA/fire_detect/tests/test_flicker.py): 4 test cases kiểm thử chuỗi diện tích dao động, sóng điều hòa tần số 6 Hz và tích hợp `FireTemporalAnalyzer`.

---

## 3. DANH MỤC SẢN PHẨM & MÃ NGUỒN BÀN GIAO

Toàn bộ 8 tệp mã nguồn và kiểm thử do Thành viên 3 triển khai:

| STT | Tên tệp tin | Hành động | Vai trò kỹ thuật |
|:---:|---|:---:|---|
| **1** | [`src/motion/__init__.py`](file:///d:/PTIT/XLA/fire_detect/src/motion/__init__.py) | Cập nhật | Lớp `MotionDetector` hoàn chỉnh (MOG2, KNN, Three-frame differencing) |
| **2** | [`src/motion/optical_flow.py`](file:///d:/PTIT/XLA/fire_detect/src/motion/optical_flow.py) | **Tạo mới** | Giải thuật Farnebäck Dense Optical Flow và hàm đo `compute_upward_motion` |
| **3** | [`src/tracking/region_tracker.py`](file:///d:/PTIT/XLA/fire_detect/src/tracking/region_tracker.py) | **Tạo mới** | Thuật toán `RegionTracker` ghép cặp Greedy IoU và duy trì lịch sử |
| **4** | [`src/tracking/__init__.py`](file:///d:/PTIT/XLA/fire_detect/src/tracking/__init__.py) | Cập nhật | Re-export `RegionTracker` và `compute_iou` chuẩn xác |
| **5** | [`src/temporal/flicker.py`](file:///d:/PTIT/XLA/fire_detect/src/temporal/flicker.py) | **Tạo mới** | Các giải thuật phân tích nhấp nháy: Pixel Flicker, Intensity CV, FFT 3-13 Hz |
| **6** | [`src/temporal/__init__.py`](file:///d:/PTIT/XLA/fire_detect/src/temporal/__init__.py) | Cập nhật | Cài đặt `FireTemporalAnalyzer` chính thức; bảo toàn `SmokeTemporalAnalyzer` cho TV4 |
| **7** | [`tests/test_motion.py`](file:///d:/PTIT/XLA/fire_detect/tests/test_motion.py) | **Tạo mới** | 5 Unit tests chuyên sâu cho tầng chuyển động & Optical Flow |
| **8** | [`tests/test_tracking.py`](file:///d:/PTIT/XLA/fire_detect/tests/test_tracking.py) | **Tạo mới** | 6 Unit tests chuyên sâu cho IoU và bộ theo dõi vùng |
| **9** | [`tests/test_flicker.py`](file:///d:/PTIT/XLA/fire_detect/tests/test_flicker.py) | **Tạo mới** | 4 Unit tests kiểm thử dao động nhấp nháy, phân tích phổ FFT |

---

## 4. KẾT QUẢ THỰC NGHIỆM & KIỂM THỬ CHẤT LƯỢNG

Toàn bộ hệ thống kiểm thử tự động của dự án đã được chạy kiểm chứng bằng `pytest` trong môi trường ảo của dự án:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Kết quả chi tiết từng module kiểm thử:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.8, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\PTIT\XLA\fire_detect
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1, platformdirs-4.12.3
collected 58 items

tests\test_classical_pipeline.py .......                                 [ 12%]
tests\test_color.py ...........                                          [ 31%]
tests\test_config.py ........                                            [ 44%]
tests\test_flicker.py ....                                               [ 51%]
tests\test_motion.py .....                                               [ 60%]
tests\test_preprocessing.py ..........                                   [ 77%]
tests\test_tracking.py ......                                            [ 87%]
tests\test_video_io.py .......                                           [100%]

============================= 58 passed in 1.08s ==============================
```

### Đánh giá chất lượng thực nghiệm:
- **Tỉ lệ đạt:** **58 / 58 bài kiểm thử đạt 100% PASSED**.
- **Không xảy ra hồi quy (Zero Regression):** Toàn bộ 32 bài kiểm thử của TV1 và 11 bài kiểm thử của TV2 vẫn hoạt động ổn định và chính xác.
- **Tốc độ thực thi:** Toàn bộ 58 bài kiểm thử chạy hoàn tất chỉ trong **1.08 giây**, chứng minh thuật toán được tối ưu hóa vectorized bằng NumPy và OpenCV C++ backend với hiệu năng xử lý realtime.

---

## 5. ĐÓNG GÓP KỸ THUẬT & TƯƠNG TÁC TRONG NHÓM

1. **Đóng góp vào độ chính xác phát hiện lửa:**
   Trong hệ thống cấu hình `configs/default.yaml`, tổng điểm tin cậy phát hiện lửa được phân bổ theo 5 tiêu chí:
   - `pixel_flicker`: **Trọng số 30%** (TV3 đảm nhiệm)
   - `intensity_flicker`: **Trọng số 15%** (TV3 đảm nhiệm)
   - `area_variation`: Trọng số 20% (TV4 đảm nhiệm)
   - `shape_change`: Trọng số 20% (TV4 đảm nhiệm)
   - `boundary_roughness`: Trọng số 15% (TV4 đảm nhiệm)  
   👉 **Thành viên 3 trực tiếp cung cấp 45% tổng trọng số quyết định của ngọn lửa**, là tầng quan trọng nhất để triệt tiêu các báo động giả từ vật thể đỏ di động.

2. **Cung cấp đầu vào hoàn chỉnh cho Thành viên 4:**
   - Cung cấp hàm `compute_upward_motion(region, flow)` từ Optical Flow Farnebäck để TV4 sử dụng trong `SmokeTemporalAnalyzer` (chiếm 30% trọng số khói).
   - Duy trì `region.history` có snapshot đầy đủ `{bbox, area, centroid, intensity_mean}` để TV4 dễ dàng tính toán các đặc trưng hình học biến thiên (`area_variation`, `shape_change`).

3. **Tính sẵn sàng cho báo cáo & demo:**
   Pipeline `ClassicalDetector` hiện đã có thể chạy trực tiếp với chuyển động MOG2 thật và bộ theo dõi thật. Người dùng có thể chạy ứng dụng Web Streamlit (`python -m streamlit run app/streamlit_app.py`) hoặc chạy CLI (`python scripts/run_video.py`) để quan sát trực tiếp mặt nạ chuyển động và các đối tượng kèm Track ID được vẽ trực quan.

---
*Báo cáo được hoàn thiện và xác nhận tính toàn vẹn vào ngày 02/11/2026 bởi Thành viên 3.*

