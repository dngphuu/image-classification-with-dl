import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_learning_curves(history_mlp, history_cnn, filename='learning_curves.png'):
    epochs = range(1, len(history_mlp['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # --- MLP Mức độ giảm độ lỗi ---
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history_mlp['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history_mlp['val_loss'], label='Val Loss', color='orange')
    plt.title('MLP - Learning Curve (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    
    # --- MLP Accuracy ---
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history_mlp['train_acc'], label='Train Acc', color='blue')
    plt.plot(epochs, history_mlp['val_acc'], label='Val Acc', color='orange')
    plt.title('MLP - Learning Curve (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # --- CNN Mức độ giảm độ lỗi ---
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history_cnn['train_loss'], label='Train Loss', color='green')
    plt.plot(epochs, history_cnn['val_loss'], label='Val Loss', color='red')
    plt.title('CNN - Learning Curve (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    
    # --- CNN Accuracy ---
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history_cnn['train_acc'], label='Train Acc', color='green')
    plt.plot(epochs, history_cnn['val_acc'], label='Val Acc', color='red')
    plt.title('CNN - Learning Curve (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Bản vẽ Learning Curves đã được lưu tại {filename}")

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', filename='cm.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Nhãn Thực tế (True Label)')
    plt.xlabel('Nhãn Dự đoán (Predicted Label)')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Bản vẽ Confusion Matrix đã được lưu tại {filename}")
