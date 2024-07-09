import os
import cv2
import librosa
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from multiprocessing import Pool, cpu_count
import json

# Set the environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def extract_frames(video_path, frame_rate=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        frame_id = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def extract_audio_from_video(video_path, output_audio_path):
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(output_audio_path)

def generate_mel_spectrogram(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def save_mel_spectrogram(mel_spec, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=22050, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_processed_videos(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_video(log_file, video_file):
    processed_videos = load_processed_videos(log_file)
    processed_videos.add(video_file)
    with open(log_file, 'w') as f:
        json.dump(list(processed_videos), f)

def process_video(args):
    video_file, video_dir, output_image_dir, frame_rate, log_file = args

    # Check if the video has already been processed
    if video_file in load_processed_videos(log_file):
        print(f"Skipping already processed video: {video_file}")
        return

    print(f"Processing video: {video_file}")

    video_path = os.path.join(video_dir, video_file)
    frames = extract_frames(video_path, frame_rate)
    print(f"Extracted {len(frames)} frames from {video_file}")

    for i, frame in enumerate(frames):
        frame_img = Image.fromarray(frame)
        frame_img.save(os.path.join(output_image_dir, f'{os.path.splitext(video_file)[0]}_frame_{i}.png'))
        print(f"Saved frame {i + 1}/{len(frames)} for {video_file}")

    audio_path = os.path.join(output_image_dir, f'{os.path.splitext(video_file)[0]}.wav')
    try:
        extract_audio_from_video(video_path, audio_path)
        print(f"Extracted audio for {video_file}")

        mel_spec = generate_mel_spectrogram(audio_path)
        save_mel_spectrogram(mel_spec, os.path.join(output_image_dir, f'{os.path.splitext(video_file)[0]}_mel.png'))
        print(f"Generated Mel-spectrogram for {video_file}")
    except Exception as e:
        print(f"Failed to process audio for {video_file}: {e}")
        # Remove partially processed files if necessary
        if os.path.exists(audio_path):
            os.remove(audio_path)
        for i in range(len(frames)):
            frame_file = os.path.join(output_image_dir, f'{os.path.splitext(video_file)[0]}_frame_{i}.png')
            if os.path.exists(frame_file):
                os.remove(frame_file)
        return

    # Log the processed video
    save_processed_video(log_file, video_file)

    return f"Processed {video_file}"

def preprocess_dfdc_data(input_dir, output_dir, frame_rate=1, log_file='processed_videos.json'):
    for label in ['real', 'fake']:
        video_dir = os.path.join(input_dir, label)
        output_image_dir = os.path.join(output_dir, label)
        os.makedirs(output_image_dir, exist_ok=True)

        video_files = os.listdir(video_dir)
        total_videos = len(video_files)

        pool_args = [(video_file, video_dir, output_image_dir, frame_rate, log_file) for video_file in video_files]

        with Pool(cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(process_video, pool_args), total=total_videos, desc=f"Processing {label} videos"):
                pass

if __name__ == "__main__":
    input_dir =  # Adjust this path to your DFDC dataset
    output_dir =  # Adjust this path to where you want to save processed data
    preprocess_dfdc_data(input_dir, output_dir, frame_rate=1, log_file='train_processed_videos.json')
