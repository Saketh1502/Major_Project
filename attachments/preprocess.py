import os
import cv2
from path import train_path, frames


def preprocess_data(video_folder, output_folder):
    count = 1
    for video in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video)
        output_path = os.path.join(output_folder, video.split('.')[0])
        os.makedirs(output_path, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_preprocessed = cv2.resize(frame_rgb, (224, 224))
            output_file = os.path.join(output_path, f'frame_{i:04d}.jpg')
            cv2.imwrite(output_file, cv2.cvtColor(frame_preprocessed, cv2.COLOR_RGB2BGR))
        cap.release()
        print(count)
        count += 1


preprocess_data(train_path, frames)
