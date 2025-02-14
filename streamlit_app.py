import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import json

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known faces from face_embeddings.json
known_faces = {}
try:
    with open('face_embeddings.json', 'r') as f:
        data = json.load(f)
        for name, embedding in data.items():
            known_faces[name] = torch.tensor(embedding)
except FileNotFoundError:
    st.error("face_embeddings.json not found. Please provide the file and restart the application.")
    st.stop()

def recognize_face(face_embedding):
    """Match a face embedding to the closest known face."""
    best_match = None
    best_distance = float("inf")

    if face_embedding.dim() == 1:
        face_embedding = face_embedding.unsqueeze(0)

    for name, known_embedding in known_faces.items():
        if known_embedding.dim() == 1:
            known_embedding = known_embedding.unsqueeze(0)

        distance = torch.nn.functional.pairwise_distance(face_embedding, known_embedding)[0].item()
        if distance < best_distance:
            best_distance = distance
            best_match = name

    confidence = 1 - best_distance
    return (best_match if best_distance < 0.7 else "Unknown", confidence)


class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Detect faces
        faces = mtcnn(pil_img)

        if faces is not None:
            with torch.no_grad():
                face_embedding = facenet(faces.unsqueeze(0))

            # Recognize the face
            name, confidence = recognize_face(face_embedding)

            # Draw boxes and labels
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{name} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return img


st.title("Live Facial Recognition with Streamlit & FaceNet")

# Start webcam with streamlit-webrtc
webrtc_streamer(key="face-recognition", video_transformer_factory=FaceRecognitionTransformer)