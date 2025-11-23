import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_logs(log_path):
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_top5_accuracies = []
    epochs = []

    with open(log_path, 'r') as f:
        content = f.read()
        
    # Extract metrics using regex
    # Pattern: Epoch X/Y ... Train Loss: A ... Val Loss: B ... Val Accuracy: C ... Val Top-5 Accuracy: D
    
    # We'll iterate line by line to be robust
    current_epoch = 0
    for line in content.split('\n'):
        if "Epoch" in line and "/" in line:
            try:
                current_epoch = int(line.split("Epoch")[1].split("/")[0].strip())
                if current_epoch not in epochs:
                    epochs.append(current_epoch)
            except:
                pass
                
        if "Train Loss:" in line:
            train_losses.append(float(line.split("Train Loss:")[1].strip()))
        if "Val Loss:" in line:
            val_losses.append(float(line.split("Val Loss:")[1].strip()))
        if "Val Accuracy:" in line:
            val_accuracies.append(float(line.split("Val Accuracy:")[1].strip().replace('%', '')))
        if "Val Top-5 Accuracy:" in line:
            val_top5_accuracies.append(float(line.split("Val Top-5 Accuracy:")[1].strip().replace('%', '')))

    return epochs, train_losses, val_losses, val_accuracies, val_top5_accuracies

def plot_metrics(epochs, train_losses, val_losses, val_accuracies, val_top5_accuracies):
    plt.figure(figsize=(12, 10))

    # 1. Loss Curve
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy Curve
    plt.subplot(2, 1, 2)
    plt.plot(val_accuracies, label='Top-1 Accuracy', marker='o')
    plt.plot(val_top5_accuracies, label='Top-5 Accuracy', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Plot saved to training_curves.png")

if __name__ == "__main__":
    log_path = Path("logs/training.log")
    if not log_path.exists():
        print(f"Error: {log_path} not found.")
    else:
        data = parse_logs(log_path)
        # Handle case where lists might be slightly different lengths due to interrupted runs
        min_len = min(len(data[1]), len(data[2]), len(data[3]))
        plot_metrics(
            data[0][:min_len], 
            data[1][:min_len], 
            data[2][:min_len], 
            data[3][:min_len], 
            data[4][:min_len]
        )
