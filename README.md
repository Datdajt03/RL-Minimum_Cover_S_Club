# Phương pháp lựa chọn thích nghi các toán tử tiến hóa dựa trên Reinforcement Learning cho bài toán Minimum s-club cover

Thư mục này chứa mã nguồn thực nghiệm và các tập dữ liệu cho bài nghiên cứu đề xuất giải pháp học tăng cường (Reinforcement Learning) nhằm lựa chọn toán tử di truyền thích nghi cho giải thuật tiến hóa (Evolutionary Algorithm - EA) giải quyết bài toán **Minimum $s$-Club Cover (Bao $s$-Club tối tiểu)**.

---

## 1. Tên chuyên ngành của các thuật toán & Khái niệm cốt lõi

### 1.1. Bài toán mục tiêu
* **Tên tiếng Anh:** Minimum $s$-Club Cover Problem (MSCCP)
* **Tên tiếng Việt:** Bài toán Bao $s$-Club tối tiểu
* **Mô tả:** Tìm số lượng phân hoạch đỉnh tối thiểu của đồ thị sao cho mỗi phân hoạch tạo thành một đồ thị con có đường kính không vượt quá $s$ (gọi là một $s$-club).

### 1.2. Các thuật toán & Chiến lược đối chiếu
Dự án thực nghiệm thực hiện so sánh 4 chiến lược chọn toán tử di truyền:
1. **EVO-RL (PPO) - *Thuật toán đề xuất*:**
   * *Tên khoa học:* Evolutionary Algorithm with Reinforcement Learning-based Adaptive Operator Selection (PPO-AOS).
   * *Mô tả:* Sử dụng tác tử học tăng cường **PPO (Proximal Policy Optimization)** để quan sát trạng thái của quần thể (Population State) và đưa ra quyết định chọn toán tử tiến hóa tối ưu một cách thích nghi tại mỗi thế hệ.
2. **No-RL (Không học tăng cường):**
   * *Tên khoa học:* Evolutionary Algorithm with Uniform Operator Selection.
   * *Mô tả:* Lựa chọn các toán tử tiến hóa với xác suất đồng đều cố định (1/3 cho mỗi toán tử).
3. **Fixed (Lịch trình tuần tự cố định):**
   * *Tên khoa học:* Evolutionary Algorithm with Deterministic Sequential Operator Schedule.
   * *Mô tả:* Áp dụng tuần tự các toán tử theo chu kỳ xoay vòng cố định (Crossover → Mutation 1 → Mutation 2).
4. **Random (Ngẫu nhiên hoàn toàn):**
   * *Tên khoa học:* Evolutionary Algorithm with Random Operator Selection.
   * *Mô tả:* Chọn toán tử ngẫu nhiên hoàn toàn tại mỗi thế hệ mà không có xác suất định trước.

### 1.3. Các toán tử tiến hóa chuyên ngành
Tác tử RL lựa chọn giữa 3 toán tử di truyền chính:
* **Structural Crossover (Lai ghép cấu trúc):** Kết hợp các phần tử từ hai cá thể cha mẹ dựa trên cấu trúc đồ thị con để tạo thế hệ con kế thừa đặc trưng hình học tốt.
* **Feasibility-preserving Mutation (Đột biến duy trì tính khả thi):** Đột biến cục bộ trên các s-club nhưng đảm bảo nghiệm sinh ra luôn duy trì tính khả thi (không vi phạm ràng buộc đường kính $s$).
* **Diversification Mutation (Đột biến đa dạng hóa):** Đột biến ngẫu nhiên trên quy mô rộng nhằm đưa quần thể thoát khỏi các cực trị địa phương (local optima).

---

## 2. Cấu trúc thư mục dự án

```text
├── pbodulieu/           # Thư mục chứa các đồ thị benchmark (.clq, .gml)
└── pcode/ban2/
    ├── env.py           # Định nghĩa Môi trường tiến hóa (SClubEnvironment) & lớp cá thể (Individual)
    ├── agent.py         # Cấu trúc mạng thần kinh và tác tử PPO (PPO-AOS Agent) cùng các Baseline
    ├── train.py         # Hàm huấn luyện (train_agent) và đánh giá (evaluate_agent)
    ├── experiment_config.py # File chứa các siêu tham số thực nghiệm (Max Generations, Stagnation Limit...)
    ├── main.py          # Luồng chạy thực nghiệm chính, thu thập dữ liệu và xuất báo cáo (Markdown & LaTeX)
    ├── generate_charts.py   # Script vẽ các biểu đồ phân tích (PNG/PDF) từ kết quả thực nghiệm
    ├── start/
    │   └── start.py     # Lối tắt khởi chạy nhanh thực nghiệm
    └── results/         # Thư mục tự động tạo chứa kết quả đầu ra (.json, .tex, .md, biểu đồ)
```

---

## 3. Siêu tham số thực nghiệm chính
Các siêu tham số được định nghĩa trong `experiment_config.py` và tối ưu hóa theo bài báo:
* **`pop_size = 50`**: Quy mô quần thể giải thuật di truyền.
* **`max_epochs = 500`**: Số episode huấn luyện tác tử RL.
* **`max_generations = 1000` ($T$)**: Số thế hệ tối đa chạy trong 1 lần tiến hóa (1 episode).
* **`stagnation_limit = 200` ($L$)**: Điều kiện dừng sớm khi quần thể không cải thiện nghiệm tốt nhất sau $L$ thế hệ liên tiếp.
* **`penalty_lambda = 10.0` ($\lambda$):** Hệ số hình phạt khi vi phạm ràng buộc đường kính $s$-club.

---

## 4. Hướng dẫn chạy thực nghiệm

### Cài đặt thư viện bắt buộc
Cài đặt các gói phụ thuộc cần thiết:
```bash
pip install -r requirements.txt
```
*(Gồm các thư viện: `torch`, `numpy`, `igraph`, `matplotlib`, `pandas`)*

### Chạy chương trình
1. **Chạy chế độ tương tác (Interactive Mode):**
   Bạn có thể chạy trực tiếp để chọn đồ thị cần chạy bằng tay:
   ```bash
   python start/start.py
   ```
2. **Chạy tự động (Auto Mode):**
   Chạy tự động đồ thị cụ thể mà không cần nhập liệu (ví dụ chạy đồ thị thứ 4 `johnson8-2-4`):
   ```bash
   python start/start.py --run 4
   ```

---

## 5. Kết quả đầu ra (Outputs)
Khi thực nghiệm hoàn thành, kết quả sẽ nằm trong thư mục `results/<tên_đồ_thị>/`:
* **`results_summary.md`**: Báo cáo dạng Markdown, hiển thị bảng so sánh tổng hợp trực quan.
* **`results_table.tex`**: Code bảng LaTeX so sánh kích thước trung bình và thời gian chạy (chèn trực tiếp vào nội dung chính của bài viết).
* **`results_table_detailed.tex`**: Code bảng LaTeX chi tiết 20 lần chạy của cả 20 hạt giống (chèn vào Phụ lục bài viết).
* **`results_chart.tex`**: Code vẽ biểu đồ hội tụ và xác suất chọn toán tử dạng vector TikZ/PGFPlots cực kỳ sắc nét cho LaTeX.
* **`charts_output/`**: Chứa 5 biểu đồ phân tích dạng ảnh PNG/PDF vẽ tự động từ Python.

---

## 6. Phương pháp thống kê tổng hợp kết quả (20 hạt giống độc lập)
Để đánh giá tính ổn định và hiệu quả của các thuật toán, báo cáo tổng hợp sử dụng các công thức thống kê sau trên $N = 20$ lần chạy ngẫu nhiên độc lập (seeds):

1. **Kích thước lớp phủ trung bình (Mean - $\mu$):**
   $$\mu = \frac{1}{N} \sum_{i=1}^{N} K_i$$
   *(Với $K_i$ là kích thước bao s-club nhỏ nhất tìm được ở lần chạy thứ $i$)*

2. **Độ lệch chuẩn (Standard Deviation - $\sigma$):**
   $$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (K_i - \mu)^2}$$
   *(Đo lường độ ổn định của giải thuật. $\sigma$ càng nhỏ, thuật toán càng ổn định)*

3. **Kết quả tốt nhất tìm được (Best / Min):**
   $$\text{Best} = \min_{i=1}^{N} K_i$$

4. **Thời gian chạy trung bình (Average Time - $\bar{T}$):**
   $$\bar{T} = \frac{1}{N} \sum_{i=1}^{N} t_i$$
   *(Với $t_i$ là thời gian thực thi tính bằng giây của lần chạy thứ $i$)*

5. **Đường cong hội tụ trung bình tại thế hệ $t$ ($\bar{f}(t)$):**
   $$\bar{f}(t) = \frac{1}{N} \sum_{i=1}^{N} f_i(t)$$
   *(Với $f_i(t)$ là giá trị fitness tốt nhất tại thế hệ $t$ của lần chạy thứ $i$)*

6. **Xác suất chọn toán tử trung bình tại thế hệ $t$ ($\bar{P}(a, t)$):**
   $$\bar{P}(a, t) = \frac{1}{N} \sum_{i=1}^{N} P_i(a, t)$$
   *(Với $P_i(a, t)$ là xác suất tác tử RL chọn toán tử $a$ tại thế hệ $t$ ở lần chạy thứ $i$)*

