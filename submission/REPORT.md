# Lab 21 — Evaluation Report

**Họ tên**: Lê Chí Hùng  **MSSV**: 2A202602863  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (14.6 GB khả dụng, fp16 — T4 không có bf16)

> Mọi con số dưới đây lấy từ `results/` của lần chạy **đầy đủ** (`EVAL_LIMIT` không đặt:
> 50 mẫu target, 15 câu regression). Một lần chạy thử với `EVAL_LIMIT=8` trước đó **không**
> được dùng cho bất kỳ con số nào trong report này.

---

## 0. Lựa chọn thí nghiệm & lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định tier T4) | Lớn nhất vừa T4 16 GB ở 16-bit LoRA (peak 8.78 GB). Giữ mặc định để mask đã được chứng minh trong NB1 áp dụng nguyên vẹn, không phải chạy lại `check_mask_agreement.py`. |
| Dataset | 250 ticket CSKH tiếng Việt → JSON 4 trường (mặc định) | Mọi nhóm điểm có thang **khách quan** (so khớp trường, keyword recall, parse JSON, ms) — không cần LLM judge. Không đổi tập eval nên checksum giữ nguyên. |
| `OPTIMIZED_PROMPT` (b) | **Không sửa** | SHA `719e74d3b6232053` — `make verify` xác nhận không đổi. |

Thứ tự thực hiện: NB1 → NB2 (đóng băng mốc) → NB3 → NB4 → NB5, trên cùng base model.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| Eval | 50 ticket target (`eval_target.jsonl`) · 15 câu regression (`eval_regression.jsonl`) |
| `max_length` | 1024 (mặc định tier) — p95 đo được là **98**, p99 = 100, max = 101, gợi ý = 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / **30** (batch 1 × grad-accum 16 = batch hiệu dụng 16 < 32) |
| LoRA `correct` | `text-linear` (12 loại module), r = 16, α = 32, LR 1e-4, cosine, fp16 |

**Vì sao giữ `max_length=1024` thay vì 256?** Với `per_device_train_batch_size=1` và
`packing=False`, mỗi batch chỉ có một mẫu nên **không có padding**: `max_length` ở đây chỉ
là ngưỡng cắt. Mẫu dài nhất là 101 token, nên dù đặt 256 hay 1024 thì không mẫu nào bị cắt
và chi phí tính toán/VRAM y hệt nhau. Lệch tier là vô hại *cho dataset này*; với dataset
có đuôi dài hơn hoặc khi bật packing, tôi sẽ đặt theo p95 (≈256).

**Template có giữ khối `<think>` không?** **Có** — `template_check.json`:
`open_tag_present = true`, `body_present = true`, verdict *"reasoning preserved — safe to
train on traces"*. Template của Qwen3.5 render nguyên phần thân suy luận
(`<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>`). Dữ liệu triage không có trace
nên câu trả lời được huấn luyện có khối `<think>` **rỗng** — xem hệ quả ở §5
(`valid_trace_rate`).

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn **được** tính loss (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **bị** che (`masked_preview`):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

Giải mã ngược xác nhận: system + ticket + phần mở `<think>` nằm ngoài loss; JSON đáp án
và `<|im_end|>` (EOS) nằm trong loss. 0.41 ≪ 0.95 nên không phải trường hợp "tính loss
cả trên prompt" (so với `MASK_MODE=everything` = 94/94 = 100%).

---

## 3. Ba baseline (NB2 — mốc đóng băng)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3245.2 |
| (b) base + optimized prompt | **0.765** | 0.7911 | 1.000 | 1016.1 |
| (c) LoRA fine-tune *(naive prompt)* | **0.970** | **0.5444** | 1.000 | 1405.6 |

*(a), (b) từ `baselines_frozen.json` (n_target = 50, `smoke_mode = false`); (c) từ `verdict.json`.*

**(b) có thật sự mạnh hơn (a) không?** **Có**, rất rõ: 0.000 → 0.765. Prompt sơ sài
không cho ra một JSON hợp lệ nào (format 0.0) và chậm gấp ~3,2 lần vì model viết lan man
đến hết `max_new_tokens`. Prompt tối ưu liệt kê nhãn hợp lệ và ép "chỉ trả JSON" nên
format đạt 1.0.

**Có sửa `OPTIMIZED_PROMPT` không?** Không. SHA giữ nguyên `719e74d3b6232053`.

**Ghi chú trung thực về thứ tự chạy:** lần chạy thử đầu tiên dùng `EVAL_LIMIT=8`. Lần
chạy đầy đủ được làm lại **từ NB1**, và NB2 (đóng băng mốc) chạy xong trước NB3, đúng như
yêu cầu. Regression của (a) và (b) bằng nhau (0.7911) là hợp lý: nhóm regression được hỏi
**không có system prompt**, nên với cùng base model hai baseline sinh ra cùng câu trả lời.

Lưu ý: bản fine-tune (c) được chấm với **naive prompt**, giống (a). Đó là chủ ý: hành vi
đã chuyển vào trọng số nên prompt dài không còn cần thiết. So (a) với (c) cho thấy riêng
fine-tune đưa target từ 0.000 lên 0.970 với cùng một prompt.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s train | VRAM GB | latency ms |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12) | 16 | 32,464,896 | 1e-4 | 0.6250 | **0.970** | 396.8 | 8.78 | 1405.6 |
| `attn_only` | q,v (2) | 283 *(matched)* | 32,456,704 | 1e-4 | **0.5376** | 0.965 | 266.8 | 8.79 | 900.1 |
| `wrong_lr` | text-linear (12) | 16 | 32,464,896 | **1e-5** | 1.5702 | **0.000** | 402.7 | 8.78 | 5268.2 |
| `qlora` | text-linear (12) | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 477.8 | **3.86** | 1767.1 |

Cả 4 run dùng chung **30 step** (`make verify`: "all runs share ONE step budget").
`attn_only` lệch **8,192** tham số so với `correct` (0.025% < 5%).

**Mỗi run chỉ đổi một biến so với `correct`:**
- `attn_only`: **vị trí gắn adapter** (12 loại module → chỉ `q_proj`, `v_proj`). Rank được
  nâng lên 283 bằng `matched_rank()` (α = 2r = 566, giữ nguyên tỷ lệ α/r) để giữ **ngân
  sách tham số** cố định. Rank ở đây là hệ quả của việc khớp ngân sách, không phải biến
  độc lập.
- `wrong_lr`: **learning rate** 1e-4 → 1e-5 (thang full fine-tune).
- `qlora`: **độ chính xác của base** 16-bit → 4-bit NF4. Adapter được chấm trên đúng base
  4-bit mà nó được train cùng (`load_in_4bit` lấy từ spec của run).

**Xếp hạng — hai thước đo cho hai thứ tự khác nhau:**

| Thứ tự | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| theo train loss (thấp = "tốt") | attn_only 0.538 | correct 0.625 | qlora 0.706 | wrong_lr 1.570 |
| **theo target NB5** | **correct 0.970** | attn_only 0.965 | qlora 0.940 | wrong_lr 0.000 |

### 4.1 — `attn_only` vs `correct`: rank hay vị trí?

`attn_only` có cùng ngân sách (32.46 M) và **loss huấn luyện thấp nhất** trong bốn run
(0.538 vs 0.625). Nếu chấm bằng loss, ta sẽ kết luận "chỉ gắn vào attention với rank lớn
là tốt hơn". Nhưng trên tập target nó **hoà / thua sát** `correct`: 0.965 vs 0.970, tức
193 vs 194 trên 200 trường — khác nhau đúng **một trường**, nằm trong nhiễu của 50 mẫu.
Hai thước đo cho **hai thứ tự khác nhau ở hạng 1–2**: loss thấp hơn của `attn_only` không
chuyển thành năng lực trên tác vụ. Với 225 mẫu và r = 283 trên chỉ 2 module, phần loss
giảm thêm nhiều khả năng là ghi nhớ tập train.

Về đòn bẩy: nâng rank **17,7 lần** (16 → 283) để bù cho việc bỏ 10/12 loại module vẫn
**không thắng** được — nên *rank không phải đòn bẩy* ở đây. Nhưng tôi cũng **không** đo
được lợi thế rõ của vị trí (chỉ 0.005). Kết luận trung thực: **trên một tác vụ hẹp, nhiều
khuôn mẫu như triage JSON, ở cùng ngân sách, vị trí và rank đều gần như không quan
trọng** — tác vụ quá dễ để tách hai yếu tố này. Muốn thấy lợi thế của all-linear, cần tác
vụ khó hơn (nhiều kiến thức mới cần MLP), hoặc so ở ngân sách nhỏ hơn. Một lợi ích **thực
tế** đo được của `attn_only`: train nhanh hơn 33% (266.8 s vs 396.8 s) và suy luận nhanh
hơn 36% (900.1 vs 1405.6 ms) khi chưa merge, vì chỉ 2 module có nhánh LoRA chạy thêm.

### 4.2 — `wrong_lr`: chỉ khác một con số

LR nhỏ hơn 10 lần, loss cuối **1.570** so với **0.625** của `correct`. Đường loss từng
bước không được lưu trong `results/` (chỉ có `final_loss`). Log huấn luyện của lần chạy thử
`EVAL_LIMIT=8`, vốn cùng cấu hình, cùng seed và cho đúng cùng `final_loss` 1.5702, cho thấy
loss của `wrong_lr` vẫn **giảm đều** (từ ~2.16 xuống ~1.12 ở bước log cuối). Vì vậy nếu chỉ
nhìn đường cong mà không biết LR, ta dễ kết luận *"model đang học, chỉ cần train lâu
hơn"* hoặc *"dữ liệu khó / rank quá nhỏ"* — rồi đi tăng rank hay thêm dữ liệu, tức sửa
sai chỗ. Thực tế trên tập target nó **sụp hoàn toàn**: target 0.000, format 0.000,
latency 5268 ms. 30 step ở LR 1e-5 chưa đủ để model học cả **định dạng** JSON hay token
EOS, nên nó sinh lan man đến hết giới hạn (vì thế chậm gấp 3,7 lần `correct`). Đây là
nút vặn có biên độ ảnh hưởng **lớn nhất** trong lab: Δtarget = −0.970, so với −0.005 khi
đổi vị trí.

### 4.3 — `qlora`: tiết kiệm VRAM, trả giá bằng gì?

Peak VRAM giảm từ **8.78 → 3.86 GB (−56%)**. Cái giá đo được:
- target **0.940 vs 0.970** (−0.030, tức 188 vs 194 trường đúng);
- train **chậm hơn 20%** (477.8 vs 396.8 s) do phải giải lượng tử hoá trọng số mỗi bước;
- suy luận **chậm hơn 26%** (1767.1 vs 1405.6 ms).

Số đo của tôi **ủng hộ có điều kiện** khuyến nghị "không dùng QLoRA cho Qwen3.5": khi
16-bit đã vừa (T4 còn dư ~6 GB), QLoRA chỉ làm mọi thứ tệ hơn — kém hơn, chậm hơn, không
được gì. Nhưng mức hư hại trên tác vụ này là nhỏ (−3 điểm %), nên nếu phần cứng **không
vừa** 16-bit (ví dụ base 9B trên T4), QLoRA vẫn là lựa chọn hợp lý, miễn là đo lại.

**Xếp hạng ba nút vặn theo mức ảnh hưởng lên target:** LR (−0.970) ≫ lượng tử hoá 4-bit
(−0.030) > vị trí ở cùng ngân sách (−0.005).

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.247` (ngưỡng ±0.020) · `valid_trace_rate = 0.00`

Bản fine-tune **thắng rõ** ở tác vụ đích: 0.970 so với 0.765 của prompt tối ưu (+0.205,
tức 194 vs 153 trên 200 trường), format giữ 1.0. Nhưng nó **làm hỏng năng lực chung**:
keyword recall trên 15 câu hỏi phổ thông rơi từ 0.791 xuống 0.544 — mất 31% tương đối, gấp
hơn 12 lần ngưỡng cho phép. Cổng yêu cầu **cả hai** điều kiện, nên phán quyết là FAILED,
và tôi không nới ngưỡng hay sửa tập eval.

Vì sao regression tụt? Lời giải thích nhân quả hợp lý nhất là **quên thảm hoạ do dữ liệu
đơn điệu**: toàn bộ 225 mẫu train có cùng một dạng — ticket vào, JSON 4 trường ra, khối
`<think>` rỗng. Tôi cập nhật **cả 12 loại module tuyến tính ở mọi lớp** với LR 1e-4 trong
2 epoch mà không có lấy một mẫu dữ liệu phổ thông nào (replay = 0%). Model được đẩy mạnh về
phía "mọi câu hỏi → trả lời ngắn kiểu triage", nên với câu hỏi phổ thông nó trả lời cụt
hơn hoặc lệch, và bỏ lỡ từ khoá. Giả thuyết này **nhất quán** với `valid_trace_rate = 0.00`:
model đã học khép `<think>` ngay lập tức. Nhưng tôi **chưa kiểm chứng trực tiếp**, vì NB5
không lưu từng câu trả lời regression; đó là bước tiếp theo đầu tiên. Thêm một chi tiết:
tôi không có `valid_trace_rate` của base để so, nên chưa thể khẳng định là *fine-tune* đã
xoá trace (đó chính là thí nghiệm B3).

Ngoài ra fine-tune **chậm hơn (b) 38%** (1405.6 vs 1016.1 ms) dù prompt ngắn hơn, vì
adapter chưa merge thêm một nhánh tính ở 12 loại module. Merge (NB6) sẽ xoá overhead này.

**Điều này nói gì về bài toán?** Prompt engineering đã đạt 0.765 *miễn phí*. Phần
fine-tune mua thêm (+0.205) là có thật, nhưng phải đánh đổi bằng năng lực chung. Kết luận
phụ thuộc cách triển khai: nếu adapter chỉ phục vụ **một endpoint triage riêng** (chỉ nhận
ticket, không bao giờ nhận câu hỏi chung), regression gần như không ảnh hưởng thực tế. Còn
nếu cùng model phải vừa triage vừa trả lời khách, bản này **không được ship**.

---

## 6. Định tính — có cả ca THUA

Toàn bộ 6 lỗi của bản fine-tune trên 50 ticket nằm ở **cùng một trường, cùng một kiểu**
(`qualitative.json`, điểm 0.75 = sai 1/4 trường). Không có lỗi nào ở `intent`, `product`
hay `format`.

| # | i | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | 4 | "…đèn bàn LED… Vỡ khi nhận. **Gấp.** Shop xem giúp." | san_pham_loi · **cao** · đèn bàn LED · trung_tinh | san_pham_loi · cao · đèn bàn LED · … (1.0) | ✅ FT đúng cả 4 trường |
| 2 | 8 | "…chuột không dây… Bảo hành bao lâu. **Không vội.** Mình vẫn tin tưởng shop." | hoi_thong_tin · **thap** · chuột không dây · tich_cuc | hoi_thong_tin · thap · … (1.0) | ✅ "Không vội" → thap đúng |
| 3 | 30 | "…đèn bàn LED… **Hoàn lại.** Sớm nhé. Lần cuối mua ở đây." | **doi_tra** · trung_binh · đèn bàn LED · tieu_cuc | doi_tra · trung_binh · … (1.0) | ✅ phân biệt được "Hoàn lại" (đổi trả) với "Hoàn tiền" — quy ước riêng của dataset |
| 4 | 3 | "…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều." | hoan_tien · **thap** · bình giữ nhiệt · tich_cuc | hoan_tien · **trung_binh** · … (0.75) | ❌ **FT thua** — urgency |
| 5 | 5 | "…nồi chiên không dầu… Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi." | san_pham_loi · **thap** · nồi chiên không dầu · trung_tinh | san_pham_loi · **trung_binh** · … (0.75) | ❌ **FT thua** — urgency |
| 6 | 39 | "…nồi chiên không dầu… Hoàn tiền. **Khi nào tiện.** Quá tệ." | hoan_tien · **thap** · nồi chiên không dầu · tieu_cuc | hoan_tien · **trung_binh** · … (0.75) | ❌ **FT thua** — urgency |
| 7 | 46 | "…đèn bàn LED… Sai màu. **Khi nào tiện.** Shop hỗ trợ tốt." | san_pham_loi · **thap** · đèn bàn LED · tich_cuc | san_pham_loi · **trung_binh** · … (0.75) | ❌ **FT thua** — urgency |

*`qualitative.json` cắt dự đoán ở 90 ký tự nên trường `sentiment` thường bị cắt ("…").
Điểm 0.75 cùng với ba trường đầu đã khớp nhãn cho thấy trường sai duy nhất là `urgency`.
NB5 không lưu dự đoán từng mẫu của baseline (b), nên bảng không có cột (b).*

**Có mẫu chung nào ở các ca FT thua không?** **Có, và rất sắc nét.** Cả 6/6 ticket eval
chứa cụm *"Khi nào tiện"* đều bị đoán `urgency = trung_binh` thay vì `thap`. Đó cũng là
**toàn bộ** lỗi của model (6 trường sai = 200 − 194). Điều đáng chú ý là trong
`train_seed.jsonl`, *"Khi nào tiện"* xuất hiện 35 lần và **luôn** gán `thap` (35/35); các
cụm "chậm" khác như *"Không vội"* (34/34 → thap) thì model học đúng hoàn toàn trên eval.

Giả thuyết (chưa kiểm chứng): bản fine-tune bám vào token **"Khi nào"**. Trong dữ liệu
train, cụm *"Khi nào có tiền về"* (yêu cầu hoàn tiền) đi với `trung_binh` 7/13 lần, và là
ngữ cảnh "Khi nào" duy nhất không thuộc cụm "Khi nào tiện". Model học một quy tắc nông
dựa trên mặt chữ thay vì nghĩa ("khi nào *tiện*" = không gấp). Đây đúng là kiểu lỗi mà
điểm trung bình 0.970 che mất: một lỗi **có hệ thống** chứ không phải ngẫu nhiên.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** deploy bản fine-tune này như một model đa dụng, và chỉ cân
nhắc deploy nó cho một endpoint triage riêng biệt sau khi sửa hai vấn đề. Lý do đầu tiên
là cổng hồi quy FAILED: model giành thêm 0.205 điểm target nhưng mất 0.247 điểm năng lực
chung. Nguyên nhân nhân quả rất có thể là dữ liệu train hoàn toàn đơn điệu (một định dạng,
khối `<think>` rỗng, 0% replay) kết hợp với việc cập nhật mọi module tuyến tính ở LR 1e-4.
Cách sửa rẻ nhất là trộn 1–5% dữ liệu phổ thông (deck §6.3), hoặc phục vụ adapter qua
hot-swap để câu hỏi chung luôn đi vào base. Lý do thứ hai là lỗi `urgency` có hệ thống
với cụm "Khi nào tiện": điểm trung bình 0.970 trông gần hoàn hảo nhưng che một quy tắc
nông, sẽ tái diễn với 100% ticket loại đó trong production.

Về đòn bẩy: thí nghiệm đối chứng cho thấy **learning rate** là nút vặn quyết định
(đặt sai một bậc là target về 0.000). Còn vị trí adapter và rank ở cùng ngân sách gần như
không tạo khác biệt (0.970 vs 0.965) trên tác vụ hẹp này. Lượng tử hoá 4-bit tốn 3 điểm %
để đổi lấy 56% VRAM. Mask đúng (41% token được giám sát) là điều kiện tiên quyết — nó
không phải "đòn bẩy" để tăng điểm, nhưng sai thì mọi số liệu phía sau đều vô nghĩa. Bài
học lớn nhất: **train loss xếp sai thứ tự hai cấu hình tốt nhất**, và **điểm trung bình
che lỗi có hệ thống**. Cả hai chỉ lộ ra khi tôi chấm trên tập target và đọc từng ca thua.

**Ba điều tôi học được:**
1. Loss huấn luyện thấp nhất (`attn_only`, 0.538) không phải cấu hình tốt nhất trên tác vụ.
   Với 225 mẫu, rank 283 có thể ép loss xuống bằng ghi nhớ; chỉ điểm target trên tập đóng
   băng mới phân xử được.
2. Một LR sai một bậc (1e-5) vẫn cho đường loss giảm "đẹp" nhưng model không học nổi định
   dạng JSON và EOS. Nhìn loss mà không nhìn output, tôi đã có thể đi tăng rank hay thêm dữ
   liệu — sửa sai chỗ.
3. 0.970 trung bình giấu một lỗi chiếm 100% ticket có "Khi nào tiện". Từ giờ tôi sẽ luôn
   nhóm lỗi theo trường và theo cụm từ trước khi tin một con số trung bình.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Lưu từng câu trả lời regression của (b) và (c) để kiểm chứng giả thuyết quên thảm hoạ
  (model trả lời cụt hay trả về JSON cho câu hỏi chung?).
- Train lại `correct` với ~3–5% mẫu replay phổ thông, rồi xem cổng hồi quy có PASS mà
  target vẫn > 0.765 không.
- Thêm ticket "Khi nào tiện" đa dạng ngữ cảnh, hoặc giảm còn 1 epoch, để xem lỗi urgency
  là do dữ liệu hay do overfit mặt chữ.
- B3: đo `valid_trace_rate` của base và so `MASK_MODE=response-only` với `assistant-only`.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link: _(chưa làm)_
