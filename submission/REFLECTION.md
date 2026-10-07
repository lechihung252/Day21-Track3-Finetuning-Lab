# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Toàn bộ 6 lỗi của bản fine-tune đều là cùng một lỗi: ticket có "Khi nào tiện" bị đoán
`urgency = trung_binh`, dù trong dữ liệu train cụm này luôn đi với `thap` (35/35). Điểm
0.970 trông gần như hoàn hảo, nhưng thực chất là "đúng 100% trừ đúng một loại ticket thì
sai 100%". Tôi không ngờ một lỗi có hệ thống lại nấp gọn trong một con số trung bình cao như vậy.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Lần chạy đầu tôi để `EVAL_LIMIT=8` (chế độ thử) nên `make verify` báo FAIL và phải chạy
lại toàn bộ. Tôi đã nghĩ phần tốn thời gian sẽ là train, nhưng thực tế phần tốn nhất là
NB4 (ba run đối chứng, ~21 phút) và việc chờ sinh văn bản để đánh giá. Lần chạy thử
cũng cho kết luận khác hẳn (regression Δ −0.083 trên 8 câu so với −0.247 trên 15 câu),
điều này cho thấy 8 mẫu là quá ít để tin.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ loss thấp hơn nghĩa là model tốt hơn, và rank càng cao thì adapter càng
mạnh. `attn_only` với r = 283 có loss thấp nhất nhưng không thắng `correct` trên tập
target. Còn `wrong_lr` có loss vẫn giảm đều nhưng target bằng 0. Giờ tôi chỉ tin điểm đo
trên tập eval đã đóng băng.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code để tóm tắt rubric, phân tích file `results/` (tìm ra lỗi "Khi nào
tiện") và soạn nháp report; việc chạy Colab và chỉnh sửa cuối là tôi làm. Chỗ nó sai: lấy
đường loss của `wrong_lr` từ log lần chạy thử 8 mẫu nhưng viết như số của lần chạy đầy đủ
— tôi phải sửa lại cho khớp `results/`; điền họ tên theo tài khoản git; và không phát
hiện `EVAL_LIMIT=8` trước khi tôi chạy — `make verify` mới là chốt chặn thật.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng tập eval và đo baseline prompt tối ưu trước. Ở lab này, prompt tốt đã đạt 0.765
mà không tốn gì. Tôi cũng sẽ đưa vào eval một nhóm câu hỏi chung ngay từ đầu, vì nếu
không có nó thì tôi đã không phát hiện ra model mất 31% năng lực chung.
