import pandas as pd
import matplotlib.pyplot as plt
import time
# If your CSV data is in a file named "training_log.csv", read it like this:
# df = pd.read_csv('training_log.csv')

# Alternatively, if your CSV is exactly as given in your question and not saved yet,
# you can store it in a multiline string and use pd.read_csv with StringIO:
from io import StringIO


df = pd.read_csv('training_log.csv')

# ------------------
# Plot Loss & IoU
# ------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1) Training & Validation Loss
axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
axes[0].plot(df['epoch'], df['valid_loss'], label='Valid Loss', marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Dice Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True)

# 2) Training & Validation IoU
axes[1].plot(df['epoch'], df['train_iou'], label='Train IoU', marker='o')
axes[1].plot(df['epoch'], df['valid_iou'], label='Valid IoU', marker='o')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('IoU')
axes[1].set_title('Training & Validation IoU')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(time.strftime("graph/loss_iou_%Y%m%d_%H%M%S.png"), bbox_inches='tight')