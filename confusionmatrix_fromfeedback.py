import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Read the file
true_labels = []
pred_labels = []

with open("feedback.log", "r") as f:
    for line in f:
        match = re.search(r"Predicted:\s*(\d+),\s*Correction:\s*(\w+)", line)
        if match:
            pred = match.group(1)
            correction = match.group(2)

            if correction.lower() == "correct":
                # Treat pred as both true and predicted label
                true_labels.append(pred)
                pred_labels.append(pred)
            else:
                true_labels.append(correction.lower())
                pred_labels.append(pred)

# Get all unique labels
labels = sorted(set(true_labels + pred_labels))

# Generate confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

# Display it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
