import os
import warnings
warnings.filterwarnings("ignore", category=Warning, message=".*align should be passed as Python or NumPy boolean.*")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Nạp model và hàm tiện ích vẽ biểu đồ
from models import MLP, CNN
from utils import plot_learning_curves, plot_confusion_matrix

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=15, device='cpu', model_name='Model'):
    """
    Hàm thực thi chung cho việc huấn luyện (Train) và xác thực (Validate).
    """
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # -------- BƯỚC TRAIN --------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # -------- BƯỚC VALIDATION --------
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"[{model_name}] Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
    return history

def evaluate_model(model, test_loader, device='cpu'):
    """
    Hàm đánh giá mô hình bằng tập Test và trả về danh sách dự đoán.
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = 100 * correct / total
    return acc, all_labels, all_preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===== Đang chạy với thiết bị phần cứng: {device} =====")
    
    # 1. Pipeline Chuẩn bị dữ liệu
    # Bias-variance tradeoff: áp dụng Data Augmentation cho Train
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Không dùng Data Augmentation cho Valid và Test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("\n--- TẢI TẬP DỮ LIỆU CIFAR-10 ---")
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_eval_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Phân chia 50k train set thành tập Train (40k) và Validation (10k)
    train_size = 40000
    val_size = 10000
    
    # Cố định seed cho generator để đảm bảo Validation set được tách ra giống hệt nhau ở cả 2 transform
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    
    generator = torch.Generator().manual_seed(42) # Reset seed
    _, val_dataset = random_split(val_eval_dataset, [train_size, val_size], generator=generator)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Siêu tham số
    epochs = 15
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0005 # Giảm learning rate theo feedback từ 0.001 -> 0.0005
    
    # 2. Khởi tạo và Huấn luyện MLP
    print("\n[INFO] Bắt đầu huấn luyện mô hình MLP (Multi-Layer Perceptron)...")
    mlp_model = MLP().to(device)
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=learning_rate)
    
    history_mlp = train_model(mlp_model, train_loader, val_loader, criterion, optimizer_mlp, epochs, device, model_name="MLP")
    
    # 3. Khởi tạo và Huấn luyện CNN
    print("\n[INFO] Bắt đầu huấn luyện mô hình CNN (Convolutional Neural Network)...")
    cnn_model = CNN().to(device)
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=learning_rate)
    
    history_cnn = train_model(cnn_model, train_loader, val_loader, criterion, optimizer_cnn, epochs, device, model_name="CNN")
    
    # 4. Đánh giá kiểm thử (Testing) và Xuất Biểu Đồ
    print("\n--- ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ (TEST SET) ---")
    
    test_acc_mlp, y_true_mlp, y_pred_mlp = evaluate_model(mlp_model, test_loader, device)
    print(f"-> Độ chính xác của MLP (Test Acc): {test_acc_mlp:.2f}%")
    
    test_acc_cnn, y_true_cnn, y_pred_cnn = evaluate_model(cnn_model, test_loader, device)
    print(f"-> Độ chính xác của CNN (Test Acc): {test_acc_cnn:.2f}%")
    
    print("\n--- LƯU KẾT QUẢ (MÔ HÌNH VÀ BIỂU ĐỒ) VÀO THƯ MỤC 'RESULT' ---")
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    
    # Lưu trọng số mô hình
    torch.save(mlp_model.state_dict(), os.path.join(result_dir, 'mlp_model.pth'))
    torch.save(cnn_model.state_dict(), os.path.join(result_dir, 'cnn_model.pth'))
    print(f"-> Đã lưu trọng số mô hình vào thư mục '{result_dir}/'")

    # Lưu biểu đồ
    plot_learning_curves(history_mlp, history_cnn, filename=os.path.join(result_dir, 'learning_curves.png'))
    plot_confusion_matrix(y_true_mlp, y_pred_mlp, classes, title='MLP Confusion Matrix', filename=os.path.join(result_dir, 'confusion_matrix_mlp.png'))
    plot_confusion_matrix(y_true_cnn, y_pred_cnn, classes, title='CNN Confusion Matrix', filename=os.path.join(result_dir, 'confusion_matrix_cnn.png'))
    
    print("\n===== HOÀN TẤT BÀI TẬP LỚN ÚNG DỤNG CIFAR-10 =====")

if __name__ == "__main__":
    main()
