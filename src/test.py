import torch
import os
from dataset import DeepfakeDataset, transform
from model import initialize_model
from PIL import Image
import librosa
import matplotlib.pyplot as plt


def load_model(checkpoint_path):
    model, feature_extractor = initialize_model()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, feature_extractor


def preprocess_video(video_path):
    # Extract frames
    frames = extract_frames(video_path)
    processed_frames = [image_transform(Image.fromarray(frame)) for frame in frames]

    # Extract audio and generate Mel-spectrogram
    audio_path = video_path.replace('.mp4', '.wav')
    extract_audio_from_video(video_path, audio_path)
    mel_spec = generate_mel_spectrogram(audio_path)

    return torch.stack(processed_frames), torch.tensor(mel_spec).unsqueeze(0)


def detect_deepfake(model, feature_extractor, video_path):
    frames, mel_spec = preprocess_video(video_path)
    frames = feature_extractor(frames, return_tensors="pt")['pixel_values']

    with torch.no_grad():
        output = model(frames, mel_spec)
        _, predicted = torch.max(output.data, 1)
        return 'fake' if predicted.item() == 1 else 'real'


if __name__ == "__main__":
    model, feature_extractor = load_model('C:/Users/RIzan/Documents/dfd/checkpoints/best-checkpoint.ckpt')
    video_path = 'path_to_video.mp4'
    result = detect_deepfake(model, feature_extractor, video_path)
    print(f'The video is detected as: {result}')
