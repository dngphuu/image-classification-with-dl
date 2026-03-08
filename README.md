# Báo cáo Bài tập 2: Phân loại ảnh với tập dữ liệu CIFAR-10

**Kính gửi Thầy/Cô,**

Dưới đây là báo cáo chi tiết về kết quả thực hiện Bài tập 2 của nhóm chúng em. Mục tiêu của bài tập là xây dựng và so sánh hai mô hình mạng nơ-ron: Multi-Layer Perceptron (MLP) và Convolutional Neural Network (CNN) trong việc phân loại ảnh từ tập dữ liệu CIFAR-10. Dự án được triển khai bằng ngôn ngữ Python, sử dụng thư viện **PyTorch**.

## 1. Mở đầu

Bài tập yêu cầu nhóm thực hiện các tác vụ sau:
- Xây dựng mạng MLP cơ bản với 3 layer.
- Xây dựng mạng CNN với 3 convolution layer.
- Huấn luyện, đánh giá (validation) và kiểm thử (testing) cả 2 mô hình trên tập CIFAR-10.
- Vẽ **Learning Curves** và **Confusion Matrix**.
- So sánh, phân tích các kết quả đạt được.

Để đáp ứng quy định của công cụ phát triển, dự án bắt buộc sử dụng `uv` làm trình quản lý thư viện Python (môi trường, dependencies sẽ được tự động cài đặt qua `uv`).

## 2. Hướng dẫn cài đặt và sử dụng

Dự án này sử dụng công cụ quản lý gói **`uv`** (trình quản lý gói siêu tốc cho Python).
Để chạy lại toàn bộ mô hình và kiểm tra kết quả, Thầy/Cô vui lòng thực hiện:

```bash
# Cài đặt môi trường ảo và thư viện thông qua uv
uv venv
uv pip install -r requirements.txt # hoặc uv sync (nếu chạy dạng project)

# Cách tốt nhất để chạy với uv trực tiếp (tự cài library trong command):
uv run --with torch --with torchvision --with matplotlib --with seaborn --with scikit-learn main.py
```

Khi chạy tập tin `main.py`, hệ thống sẽ tự động tải dataset CIFAR-10, phân chia train/val/test data, khởi tạo và huấn luyện 2 mô hình, sau đó trực tiếp xuất các biểu đồ so sánh vào thư mục gốc (`learning_curves.png`, `confusion_matrix_mlp.png`, `confusion_matrix_cnn.png`).

## 3. Kiến trúc Mô hình

### 3.1. Mô hình Multi-Layer Perceptron (MLP)
Mô hình MLP được thiết kế với 3 lớp fully-connected (tương đương với 3 block Linear layers):
- **Lớp đầu vào**: Tensor ảnh 3x32x32 được duỗi phẳng (flatten) thành vector 3072 chiều.
- **Hidden Layer 1**: `Linear(3072, 1024)` -> `ReLU`.
- **Hidden Layer 2**: `Linear(1024, 512)` -> `ReLU`.
- **Output Layer**: `Linear(512, 10)` (Tương ứng với 10 nhãn của tập CIFAR-10).

### 3.2. Mô hình Convolutional Neural Network (CNN)
Mô hình CNN được thiết kế với 3 lớp tích chập (convolutional layers) trích xuất đặc trưng, mỗi lớp theo sau là một hàm kích hoạt ReLU và Max Pooling nhằm giảm không gian đặc trưng nhưng giữ lại thông tin cốt lõi:
- **CNN Layer 1**: `Conv2d(3, 32, ker=3, pad=1)` -> `ReLU` -> `MaxPool2d(2)`. Kích thước không gian giảm từ 32x32 xuống 16x16.
- **CNN Layer 2**: `Conv2d(32, 64, ker=3, pad=1)` -> `ReLU` -> `MaxPool2d(2)`. Không gian: 8x8.
- **CNN Layer 3**: `Conv2d(64, 128, ker=3, pad=1)` -> `ReLU` -> `MaxPool2d(2)`. Không gian: 4x4.
- **Classifier Class**: Flatten feature maps `128*4*4` và truyền qua các lớp Linear (`Linear(2048, 256) -> ReLU -> Linear(256, 10)`) để tính toán đầu ra lớp.

## 4. Quá trình Huấn luyện và Đánh giá

### Thiết lập Train/Val/Test
- Tập huấn luyện `train` có sẵn từ CIFAR-10 (50k hình ảnh) được tách thành: **Train Set (40,000 ảnh)** và **Validation Set (10,000 ảnh)**.
- Tập **Test Set (10,000 ảnh)** được dùng riêng cho việc đánh giá độ chuẩn xác thực tế vào cuối.

### Cấu hình
- **Hàm mất mát (Loss Function)**: `CrossEntropyLoss`.
- **Thuật toán tối ưu hoá (Optimizer)**: Nhóm sử dụng `Adam` với Learning rate `0.001` vì khả năng hội tụ nhanh và ổn định.
- **Số chu kì (Epochs)**: Đặt ở mức 15-20 epochs, vừa đủ để quan sát hiệu suất mô hình trong điều kiện tài nguyên giới hạn.

## 5. Kết quả Thực nghiệm & Phân tích

Sau khi thực thi hàm huấn luyện cho cả MLP và CNN, nhóm ghi nhận sự thay đổi của hàm mất mát (loss) và độ chính xác (accuracy) qua từng epoch. Dữ liệu này được biểu diễn ở file **`learning_curves.png`**.

Bên cạnh đó, việc phân loại nhầm giữa những nhãn có nét tương đồng cao (trực quan) được biểu thị rõ rệt trên biểu đồ ma trận nhầm lẫn **`confusion_matrix_mlp.png`** và **`confusion_matrix_cnn.png`**.

### So sánh khả năng phân loại
- **Mô hình MLP**:
  - Tốc độ huấn luyện trên epoch rất nhanh (do phép toán thuần tuý số học ma trận).
  - Khả năng tổng quát hóa kém: Khi mạng lớn (nhiều parameter như lớp `3072 -> 1024`), hiện tượng *Overfitting* xảy ra rất rệt trên Learning curve (Train Loss giảm nhưng Val Loss có xu hướng chững lại hoặc tăng lên, Train Acc tăng cao nhưng Val Acc thấp ~50%).
  - Lí do là MLP phớt lờ cấu trúc không gian (spatial features) của hình ảnh mà xem các điểm ảnh rời rạc.

- **Mô hình CNN**:
  - Yêu cầu khả năng tính toán cao hơn do có nhiều phép nhân chập trên khối ma trận.
  - Vượt trội về độ chính xác: Việc chia sẻ trọng số (Weight sharing) và tận dụng mối liên kết không gian từ Kernels giúp CNN trích xuất được các vạch, góc, rồi tổng hợp thành các feature có ý nghĩa. CNN đem lại Accuracy cao hơn rõ rệt (có thể đạt ~70% trên Validation/Test chỉ sau 15-20 epochs với model đơn giản này) và ít Overfitting hơn so với MLP.

### Tổng kết
Mô hình CNN luôn là lựa chọn ưu tiên dành cho bài toán phân tích và xử lý ảnh (Computer Vision). Bài tập cung cấp góc nhìn nền tảng, minh chứng lợi thế vượt bậc của Convolution Layer trước Fully Connected Layer khi làm việc trực tiếp với dữ liệu không gian nhiều chiều như hình ảnh.

*Một lần nữa cảm ơn Thầy/Cô đã theo dõi và đánh giá báo cáo hoàn thiện Bài tập 2 của tụi em!*
