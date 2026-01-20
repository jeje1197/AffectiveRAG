import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss(metrics_path: Path, output_path: Path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    metrics = data["metrics"]
    train_loss = metrics.get("train_loss", [])
    val_loss = metrics.get("val_loss", [])
    
    if not train_loss or not val_loss:
        # Check for multistage metrics
        if "stage1_train_loss" in metrics:
            train_loss = metrics["stage1_train_loss"] + metrics.get("stage2_train_loss", [])
            val_loss = metrics["stage1_val_loss"] + metrics.get("stage2_val_loss", [])
        else:
            print("No loss data found in metrics.")
            return

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training Loss", color="blue", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", color="orange", linestyle="--", linewidth=2)
    
    plt.title("ALS Model Training", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("BCE Loss (Weighted)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Loss curve saved to {output_path}")

if __name__ == "__main__":
    metrics_file = Path("v1/artifacts/pretrained/training_metrics.json")
    output_file = Path("v1/artifacts/pretrained/loss_curve.png")
    plot_loss(metrics_file, output_file)
