# dataset.py
import os

import torch
from datasets import Dataset
from transformers import DeiTFeatureExtractor
import cv2


def preprocess_frames(frames, feature_extractor):
    preprocessed_frames = []
    for frame in frames:
        inputs = feature_extractor(images=frame, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        # Ensure pixel_values has the correct shape
        if len(pixel_values.shape) == 4:  # Ensure it's a 4D tensor (batch_size, num_channels, height, width)
            batch_size, num_channels, height, width = pixel_values.shape
            # Example: Normalize, resize, etc.
            # Add to preprocessed_frames
            preprocessed_frames.append(pixel_values)

    return torch.cat(preprocessed_frames)


class DeepfakeDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = {'real': 0, 'fake': 1}
        self._data = self.load_dataset()  # Use a private attribute for data

    @property
    def data(self):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        visual_path, audio_path, label = self.data[idx]
        visual_features = torch.load(visual_path)
        audio_features = torch.load(audio_path)
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is converted to tensor
        return visual_features, audio_features, label

    def load_dataset(self):
        data = []
        for label in self.classes:
            label_idx = self.classes[label]
            label_path = os.path.join(self.dataset_path, label)
            for visual_file in os.listdir(os.path.join(label_path, 'visual')):
                if visual_file.endswith('.pt'):
                    visual_path = os.path.join(label_path, 'visual', visual_file)
                    audio_file = os.path.splitext(visual_file)[0] + '.pt'
                    audio_path = os.path.join(label_path, 'audio', audio_file)
                    data.append((visual_path, audio_path, label_idx))
        return data
