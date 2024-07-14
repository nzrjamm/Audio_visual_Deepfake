import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchvision import models
from my_dataset import MyDataset  # Replace with your dataset class
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the validation data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the validation dataset
val_dataset = MyDataset(root='path_to_val_data', transform=transform)  # Replace with your dataset path and class
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the trained model
model = models.resnet50(pretrained=False)  # Replace with your model architecture
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust for the number of classes
model.load_state_dict(torch.load('path_to_trained_model.pth'))  # Replace with your model path
model = model.to(device)
model.eval()

# Initialize metrics
all_preds = []
all_labels = []

# Evaluation loop
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # Collect predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy = np.mean(all_preds == all_labels)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)

# Print results
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')
print('Confusion Matrix:')
print(cm)

# Save results to a file (optional)
with open('validation_results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy * 100:.2f}%\n')
    f.write(f'Precision: {precision * 100:.2f}%\n')
    f.write(f'Recall: {recall * 100:.2f}%\n')
    f.write(f'F1 Score: {f1 * 100:.2f}%\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm))
