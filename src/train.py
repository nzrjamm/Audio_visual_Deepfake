import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from pytorch_lightning.loggers import WandbLogger
from dataset import DeepfakeDataset, transform
from model import initialize_model
import wandb

# Log in to W&B
wandb.login(key='e3452ce10803404c28ed7778b50c619e2a84e49c')

# Initialize W&B logger
wandb_logger = WandbLogger(log_model='all', project="deceptive_realities_dfd", name="nzrjama")

# Define the directory for saving model checkpoints
checkpoint_dir = 'C:/Users/RIzan/Documents/dfd/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Load datasets
train_dataset = DeepfakeDataset(data_dir='C:/Users/RIzan/Documents/dfd/dataset/train_processed', transform=transform)
val_dataset = DeepfakeDataset(data_dir='C:/Users/RIzan/Documents/dfd/dataset/val_processed', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model and feature extractor
model, feature_extractor = initialize_model()

class DeepfakeDetectionModel(pl.LightningModule):
    def __init__(self, model, feature_extractor):
        super(DeepfakeDetectionModel, self).__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, frames, mels):
        # Move tensors to the appropriate device
        device = frames.device
        frames = self.feature_extractor(images=frames, return_tensors="pt")['pixel_values'].to(device)

        # Ensure mel spectrograms have 1 channel
        if mels.dim() == 4 and mels.size(1) == 4:  # If mels have 4 channels, convert to 1 channel
            mels = mels[:, 0:1, :, :]

        # Print shapes for debugging
        print(f"Frames shape: {frames.shape}")
        print(f"Mel spectrograms shape: {mels.shape}")

        outputs = self.model(frames, mels)
        return outputs

    def training_step(self, batch, batch_idx):
        frames, mels, labels = batch
        outputs = self(frames, mels)
        loss = self.criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        train_accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        train_auc = roc_auc_score(labels.cpu().numpy(), predicted.cpu().numpy())

        # Print training metrics to the terminal
        print(f"Training - Loss: {loss.item()}, Accuracy: {train_accuracy}, AUC: {train_auc}")

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_auc', train_auc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        frames, mels, labels = batch
        outputs = self(frames, mels)
        loss = self.criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        val_accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        val_auc = roc_auc_score(labels.cpu().numpy(), predicted.cpu().numpy())

        # Print validation metrics to the terminal
        print(f"Validation - Loss: {loss.item()}, Accuracy: {val_accuracy}, AUC: {val_auc}")

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True)
        self.log('val_auc', val_auc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)  # Reduced learning rate
        return optimizer

# Initialize the model for training
deepfake_model = DeepfakeDetectionModel(model, feature_extractor)

# Define a custom callback to print a message when training finishes
class TrainingFinishedCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        print("Training finished!")

# Initialize Trainer with custom callback
trainer = pl.Trainer(
    max_epochs=10,
    logger=wandb_logger,
    callbacks=[
        pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=1, monitor='val_loss', mode='min'),
        TrainingFinishedCallback()
    ],
    default_root_dir=checkpoint_dir,
    accelerator='gpu',  # Use GPU for training
    devices=1  # Number of GPUs to use
)

# Train the model
trainer.fit(deepfake_model, train_loader, val_loader)
