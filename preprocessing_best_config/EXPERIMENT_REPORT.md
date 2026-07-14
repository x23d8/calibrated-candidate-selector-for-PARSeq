# Thử nghiệm tiền xử lý ảnh trước PARSeq

## Thiết lập

- Checkpoint: `outputs/refinement_finetune_20260710_142307/best_official_parseq_anpr.pt`, epoch 26.
- Validation: 397 ảnh; test khóa: 411 ảnh.
- Đầu vào PARSeq: 32 x 128, autoregressive decode, `refine_iters=2`.
- Baseline đúng với lúc fine-tune: grayscale -> CLAHE 2.0/8x8 -> bilateral -> unsharp nhẹ.
- Tổng cộng 47 cấu hình từ xử lý mức xám, tương phản, lọc không gian/tần số, khử nhiễu, morphology, khôi phục chiếu sáng, đa kênh và nội suy.
- Xếp hạng trên validation bằng exact-match trước, character accuracy sau; chỉ sáu finalist validation được xác nhận trên test.
- So sánh theo cặp trên cùng ảnh và bootstrap 2.000 lần.

## Kết quả tốt nhất

| Phương pháp | Val exact | Val char acc | Test exact | Test char acc | Delta test so với baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLAHE nhẹ, clip 1.0/tile 4x4 | 93.20% | 98.74% | **93.19%** | **99.08%** | **+1.22 / +0.21 điểm %** |
| Autocontrast | 93.20% | 98.74% | 92.70% | 98.96% | +0.73 / +0.09 điểm % |
| Kênh green | 93.20% | 98.74% | 92.70% | 98.96% | +0.73 / +0.09 điểm % |
| Percentile stretch 2-98% | 93.45% | 98.74% | 92.70% | 98.96% | +0.73 / +0.09 điểm % |
| Gamma 1.1 | 93.20% | 98.81% | 92.70% | 98.96% | +0.73 / +0.09 điểm % |
| Homomorphic filter | **93.70%** | **99.05%** | 92.46% | 98.87% | +0.49 / 0.00 điểm % |
| Baseline lúc fine-tune | 92.70% | 98.65% | 91.97% | 98.87% | mốc so sánh |

CLAHE nhẹ sửa đúng 9 ảnh baseline nhận sai và làm sai 4 ảnh baseline vốn đúng, tăng
ròng 5/411 ảnh. Trên test, số lỗi ký tự giảm từ 38 xuống 31.

Theo loại biển, CLAHE nhẹ giữ nguyên exact-match trên quân đội và ngoại giao,
tăng biển thường từ 93,24% lên 93,80%, và tăng biển vàng từ 58,33% lên 83,33%
(7/12 lên 10/12 ảnh đúng). Nhóm biển vàng còn rất nhỏ nên cần kiểm tra thêm.

Khoảng tin cậy bootstrap 95% cho mức tăng CLAHE nhẹ:

- Exact-match: [-0,49; +2,92] điểm %.
- Character accuracy: [-0,03; +0,47] điểm %.

Hai khoảng vẫn cắt qua 0, vì vậy đây là candidate tốt nhất hiện tại nhưng chưa
phải bằng chứng thống kê đủ mạnh cho dữ liệu ngoài mẫu.

## Đánh giá các nhóm phương pháp mới

### Có khả năng áp dụng

1. **CLAHE nhẹ 1.0/4x4**: candidate triển khai tốt nhất, chi phí thấp và cải thiện đồng thời exact-match lẫn lỗi ký tự.
2. **Percentile stretch 2-98%**: đứng thứ hai validation, đơn giản và bền vững trước pixel quá sáng/tối.
3. **Gamma 1.1, autocontrast hoặc kênh green**: cùng đạt 92,70% exact trên test; phù hợp làm augmentation khi fine-tune hơn là chạy nhiều nhánh inference.
4. **Homomorphic filtering**: thắng validation và CI character accuracy validation nằm trên 0, nhưng không giữ mức tăng character trên test, đồng thời chậm hơn khoảng ba lần; chưa nên dùng production.
5. **NLM**: exact validation khá nhưng character accuracy thấp hơn baseline và throughput chỉ khoảng 47 ảnh/giây; không đáng chi phí.

### Không phù hợp với checkpoint hiện tại

- Letterbox giữ tỉ lệ: 38,04% exact validation; checkpoint đã học trên ảnh kéo giãn trực tiếp 32x128.
- Morphological closing ngang/dọc: 77,83%/85,64%; làm dính nét ký tự.
- Otsu/adaptive threshold: 82,87%/79,85%; mất thông tin anti-alias và chi tiết nét.
- Median/wavelet: 86,65%/87,66%; làm mờ nét ở crop nhỏ.
- Wiener, Retinex, Laplacian, chọn kênh tự động: đều thấp hơn baseline.
- Resize bilinear/Lanczos thay bicubic: đều giảm accuracy.

Các phương pháp video/motion, compression, segmentation, feature extraction và
object detection không nằm trong sweep vì đầu vào đã là crop biển số và PARSeq
tự học đặc trưng. Deblurring nghịch đảo/least-squares cũng không được dùng vì
không có point-spread function đã biết; kernel sai dễ tạo ringing.

## Khuyến nghị triển khai

Dùng `clahe_clip1_tile4` làm candidate và giữ `train_baseline` làm fallback. Cần
xác nhận trên một holdout mới, đặc biệt bổ sung biển vàng, quân đội, ngoại giao,
ảnh mờ và thiếu sáng. Không tiếp tục chọn tham số bằng test 411 ảnh hiện tại.

Áp dụng cho một ảnh PIL:

```python
from preprocessing_best_config.preprocessing import preprocess_plate_image

image_for_parseq = preprocess_plate_image(image, "clahe_clip1_tile4")
```

Fine-tune với cùng domain tiền xử lý:

```powershell
python train_no_refinement\parseq_official_anpr_pipeline.py `
  --preprocess --preprocess-config clahe_clip1_tile4
```

Kết quả đầy đủ và dự đoán từng ảnh nằm tại
`outputs/preprocessing_course_benchmark/`.
