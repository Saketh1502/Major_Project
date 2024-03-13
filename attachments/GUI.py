import cv2
import gradio as gr
import numpy as np


def predict_video(model, video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_preprocessed = cv2.resize(frame_rgb, (224, 224))
        frames.append(frame_preprocessed)

    cap.release()

    frames = np.array(frames)
    prediction = model.predict(frames)

    return "Real" if np.mean(prediction) < 0.5 else "Fake"


# Gradio UI
iface = gr.Interface(predict_video,
                     inputs="file",
                     outputs="text",
                     title="Deepfake Detection",
                     description="Upload a video to detect if it's real or fake.")
iface.launch()
