# Báo cáo Assignment 2: Phân loại ảnh với tập dữ liệu CIFAR-10

## 1. Chuẩn bị dữ liệu (Data Pipeline)
Tập dữ liệu CIFAR-10 gồm 50.000 ảnh tập huấn luyện (train) và 10.000 ảnh tập kiểm thử (test) với kích thước 32x32 pixel. Tôi đã chia 50.000 ảnh này thành tập huấn luyện (40.000 ảnh) và tập xác thực (10.000 ảnh) để tiện cho quá trình theo dõi hiệu năng mô hình, tránh việc bị overfitting (quá khớp).

**Phương pháp áp dụng:**
*   **Data Augmentation (Tăng cường dữ liệu):** Áp dụng phép biến đổi ảnh: `RandomHorizontalFlip` (lật ngang ảnh) và `RandomRotation(10)` (xoay ngẫu nhiên 10 độ).
    *   *Tại sao lại dùng?* Việc làm phong phú dạng dữ liệu đầu vào giúp mô hình học được nhiều đặc trưng hơn, nâng cao độ tổng quát hóa của mô hình trên tập dữ liệu chưa từng thấy. Tôi chỉ áp dụng phép xoay, lật cho tập Train, còn Valid và Test vẫn giữ nguyên để đánh giá khách quan nhất hiệu suất.
*   **Chuyển đổi Tensor & Chuẩn hóa (Normalization):** Dùng `(0.5, 0.5, 0.5)` trên 3 kênh màu RGB để biến các giá trị của ảnh về miền `[-1, 1]`.
    *   *Tại sao lại dùng?* Việc chuẩn hóa giúp cho giá trị gradient lan truyền ngược (backpropagation) được ổn định hơn, từ đó cải thiện tốc độ hội tụ của tiến trình gradient descent.

## 2. Xây dựng cấu trúc mạng (Models)

### 2.1 Cấu trúc MLP (Multi-Layer Perceptron)
*   **Thiết kế:** Dữ liệu ảnh đầu vào ($3 \times 32 \times 32 = 3072$) được đưa qua lớp `Flatten()` thành vector 1D (3072 chiều). Sau đó đưa qua 3 lớp kết nối đầy đủ (Fully Connected/Linear Layers). Số lượng perceptron trên các hidden layer lần lượt theo chuỗi: `3072 -> 1024 -> 512 -> 10`. Đưa thêm hàm kích hoạt `ReLU` và `Dropout (0.3)` chèn vào giữa các tầng ẩn.
*   **Tại sao lại chọn cấu trúc này?** Việc giảm dần số lượng perceptron giúp mạng nén thông tin từ từ, tự chắt lọc các biểu diễn mức cao (high-level representations) về con số 10 lớp phân loại đầu ra cơ bản.
*   **Dropout (0.3):** Ngẫu nhiên "tắt" đi 30% kết nối nhằm tránh việc mạng phụ thuộc quá nhiều vào một lượng cực nhỏ đặc trưng nhiễu, buộc các nơ-ron còn lại cũng phải học chung nhiều thuộc tính tốt. Đây là một cơ chế giúp cho MLP không bị "thuộc lòng" tập điểm số ảo từ tập train.

### 2.2 Cấu trúc CNN (Convolutional Neural Network)
*   **Thiết kế:** CNN gồm 3 lớp Tích chập (Convolution Layers), số lượng filters tăng dần: `32 -> 64 -> 128`. Kích thước bộ lọc (kernel_size) là bằng 3x3, kết hợp với `padding=1` để không bị thu nhỏ kích thước ngay sau việc quét ma trận. Sau mỗi lớp Conv2d là hàm non-linear `ReLU` và lớp `MaxPool2d` kích thước 2x2.
*   Sau 3 lần MaxPooling, vector đầu ra sẽ được dẹt lại qua `Flatten()` (kích thước $128 \times 4 \times 4 = 2048$) rồi phân loại qua hai tầng linear: `2048 -> 256 -> 10`.
*   **Tại sao lại chọn cấu trúc này?** CNN vốn rất vượt trội trong việc xử lý hình ảnh nhờ khả năng trích xuất các đặc trưng không gian (như góc, cạnh, đường cong mặt, hình khối) thông qua các ma trận quét đặc trưng. Kích thước MaxPooling giảm số lượng tham số cần đào tạo khổng lồ, đồng thời đảm bảo được tính kháng nhẹ dịch chuyển trong bức ảnh (bất biến cục bộ / translation invariance).

## 3. Huấn luyện (Training) & Đánh giá (Testing)

**Phương pháp huấn luyện:**
*   **Hàm mất mát (Loss Function) - `CrossEntropyLoss`:** Tại sao? Hai mô hình đều thuộc bài toán phân loại đa lớp độc lập. Hàm Loss này rất phù hợp nhờ việc kết hợp tối ưu lớp Softmax với hàm Log-Loss. Kết quả mang tính thống kê cao để cập nhật sai số.
*   **Thuật toán tối ưu (Optimizer) - `Adam`:** Tại sao? Thay vì dùng SGD thông thường, Adam sử dụng cả 2 mô-men hàm động lượng. Quá trình hội tụ diễn ra tự thích ứng và nhạy bén với từng trọng số hơn so với SGD, dễ tránh được các trũng gradient xấu.
*   **Học suất (Learning Rate):** Set cứng bằng `0.0005`.
*   **Vòng lặp (Epochs) = 15 & Batch size = 64:** Batch lớn giúp tối ưu tốt hơn tài nguyên phần cứng, nhưng 64 thay vì cao hơn giúp duy trì thuật toán học còn tính cực bộ nhiễu.

## 4. Tổng kết đánh giá (Learning Curves & Confusion Matrix)
*   **Đường cong học tập (Learning curves):** Đồ thị này theo dõi Accuracy/Loss ở Train và Validation (nằm ở thư mục `result/learning_curves.png`). Phân tích trực diện biểu đồ có thể nhận diện ngay trạng thái chuẩn hay Underfitting, Overfitting.
*   **Ma trận nhầm lẫn (Confusion Matrix):** Sử dụng thư viện vẽ lên ma trận trực quan (nằm ở `result/confusion_matrix_...`). Phương pháp này chỉ ngay mẫu/nhãn cụ thể nào mô hình thường xuyên gán sai vào những class nào nhất (VD: con chó hay bị nhìn ra con mèo do tương quan lớn so với máy bay).

*   **So sánh tổng quan hai cấu trúc mạng:**
    Theo biểu đồ, **CNN** chứng minh khả năng vượt qua xa với Test Accuracy vượt trội và ổn định hơn so với **MLP**. Lý do là MLP thực chất phá dỡ trật tự không gian mảng (chia làm vector 1D) nên mất sạch thông tin quan hệ lân cận giữa các điểm ảnh kề nhau, trong khi CNN duy trì trọn vẹn đặc trưng lưới 2 chiều này hiệu quả cao nhất. Dù vậy, cả 2 mô hình đã phản ánh rõ rệt các lý thuyết kinh điển đối với bài toán này.
