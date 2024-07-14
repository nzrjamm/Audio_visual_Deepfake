import torch
import torch.nn as nn
from transformers import DeiTModel, DeiTFeatureExtractor

class AudioTransformer(nn.Module):
    def __init__(self):
        super(AudioTransformer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 768)  # Match DeiT output size

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, embed_size):
        super(CrossModalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads=8)

    def forward(self, audio_feats, visual_feats):
        attn_output, _ = self.attention(audio_feats.unsqueeze(0), visual_feats.unsqueeze(0), visual_feats.unsqueeze(0))
        return attn_output.squeeze(0)

class UnifiedModel(nn.Module):
    def __init__(self, deit_model, audio_model, cross_modal_attention):
        super(UnifiedModel, self).__init__()
        self.deit_model = deit_model
        self.audio_model = audio_model
        self.cross_modal_attention = cross_modal_attention
        self.fc = nn.Linear(768 * 2, 2)  # Combine audio and visual features

    def forward(self, images, audio_specs):
        visual_feats = self.deit_model(pixel_values=images).last_hidden_state[:, 0, :]  # Extract CLS token representation
        audio_feats = self.audio_model(audio_specs)
        cross_feats = self.cross_modal_attention(audio_feats, visual_feats)
        combined_feats = torch.cat((visual_feats, cross_feats), dim=1)
        output = self.fc(combined_feats)
        return output

def initialize_model():
    feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    deit_model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    audio_model = AudioTransformer()
    cross_modal_attention = CrossModalAttention(embed_size=768)
    unified_model = UnifiedModel(deit_model, audio_model, cross_modal_attention)
    return unified_model, feature_extractor
