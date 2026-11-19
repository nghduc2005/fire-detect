# Báo cáo kết quả Thành viên 4 — Đặc trưng khói, lọc sai & Đánh giá thực nghiệm

## 1. Các module đã tích hợp vào repo

| Hạng mục | File | Nội dung |
|---|---|---|
| Đặc trưng khói | `src/temporal/smoke_temporal.py` | `SmokeTemporalAnalyzer`: `edge_blur` (Sobel so với nền trượt, đóng băng nền trong vùng khói, đo ở lõi mask), `texture_smoothness` (entropy LBP 8 láng giềng), `upward_motion` (Optical Flow Farnebäck của TV3, fallback theo trọng tâm), `area_growth`. Các đặc trưng được làm trơn trượt 5 frame. |
| Đặc trưng hình dạng lửa | `src/temporal/shape_features.py` | `area_variation`, `shape_change` (1 − IoU bbox liên tiếp), `boundary_roughness`; được gọi tự động từ `FireTemporalAnalyzer` của TV3. |
| Lọc & chấm điểm | `src/filtering/__init__.py` | `morphology_clean`, `extract_candidates` (thêm `solidity`, `boundary_roughness`, `aspect_ratio`, `area_ratio`, giữ mask ROI), `FeatureScorer`, `PersistenceVoter` (có `begin_frame`/`end_frame` ghi nhận cả frame trượt). |
| Đánh giá | `src/evaluation/__init__.py` | `compute_iou`, `DetectionEvaluator` (Acc/Prec/Recall/F1/FPR/mIoU/FPS), `evaluate_video`. |
| Thực nghiệm bóc tách | `scripts/run_ablation.py` | 4 cấu hình, in bảng + xuất `outputs/ablation_results.csv`. |

Các chỉnh sửa tích hợp nhỏ: `RegionTracker.active_ids` (TV3), `ClassicalDetector` gọi
`begin_frame`/`end_frame` + reset voter, lazy import `ClassicalDetector` để tránh vòng import,
thêm `temporal.smoke` và tinh chỉnh `persistence`/`scoring` trong `configs/default.yaml`.

## 2. Cách chạy

```bash
python -m pytest -q                       # toàn bộ unit test (pass)
python -m scripts.run_ablation            # đọc data/raw, xuất outputs/ablation_results.csv
python scripts/run_video.py -i data/raw/pos_fire.avi --debug
```

`scripts/make_synthetic_dataset.py` sinh 4 video demo (khói, lửa, đèn giao thông, hoàng hôn)
để chạy thử khi chưa có dữ liệu thật.

## 3. Kết quả trên bộ video demo (4 video × 90 frame)

| Cấu hình | Acc | Prec | Recall | F1 | FPR | FPS |
|---|---|---|---|---|---|---|
| 1. Only Color (Baseline) | 46.7 | 48.3 | 93.3 | 63.6 | 100.0 | 48.6 |
| 2. Color + Motion | 79.7 | 79.6 | 80.0 | 79.8 | 20.6 | 40.7 |
| 3. Color + Motion + Temporal | **81.1** | 87.3 | 72.8 | **79.4** | 10.6 | 20.6 |
| 4. Full Pipeline (+ Persistence) | 80.0 | **92.2** | 65.6 | 76.6 | **5.6** | 21.6 |

Nhận xét:

- Mask chuyển động loại bỏ phần lớn báo động giả tĩnh/đèn: FPR 100% → 20.6%.
- Đặc trưng thời gian (FFT/lửa, Sobel-LBP/khói, hình dạng) tăng Precision 79.6% → 87.3%
  và giảm FPR còn 10.6% (đèn giao thông, ô tô trắng, hoàng hôn bị loại).
- Persistence đạt Precision/FPR tốt nhất (92.2% / 5.6%) nhưng tốn vài frame "khởi động"
  cho mỗi track nên Recall/F1 trên clip ngắn bị giảm. Với video dài hơn, chi phí này
  gần như không đáng kể.
- **Mean IoU = 0** vì bộ video demo chưa có ground-truth bounding box; khi truyền
  `gt_boxes` vào `DetectionEvaluator.update_frame`, IoU được tính bình thường
  (đã có unit test đạt 100%).

## 4. Ghi chú kỹ thuật

- `edge_blur` dùng thang đảo `[0.30, 1.00, 0.50]`: giá trị < 1 (mờ hơn nền) được điểm cao.
- Nền xám cập nhật trượt với `bg_alpha` nhưng đóng băng bên trong vùng khói để tránh
  "tự hấp thụ" đối tượng cần đo.
- Optical Flow Farnebäck gần như bằng 0 với khói mịn/ít texture, nên `SmokeTemporalAnalyzer`
  tự fallback sang dịch chuyển trọng tâm khi tín hiệu flow < 0.05.
- `FeatureScorer` gán điểm trung tính 0.5 cho đặc trưng chưa đủ lịch sử, tránh phạt oan
  các track mới; `PersistenceVoter` ghi 0 cho track hoạt động nhưng không được vote.
- Các ngưỡng trong `configs/default.yaml` (fire threshold 0.30, persistence 8/2) được
  tinh chỉnh trên bộ demo; cần chạy lại trên Bilkent/MIVIA/FIRESENSE để chốt số liệu báo cáo.
