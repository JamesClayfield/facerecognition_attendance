import streamlit as st
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

    # Ensure the face_embedding is 2D: [1, 512]
    if face_embedding.dim() == 1:
        face_embedding = face_embedding.unsqueeze(0)

    for name, known_embedding in known_faces.items():
        # Ensure known_embedding is also 2D
        if known_embedding.dim() == 1:
            known_embedding = known_embedding.unsqueeze(0)

        # Calculate the distance and convert to scalar
        distance = torch.nn.functional.pairwise_distance(face_embedding, known_embedding)[0].item()

        if distance < best_distance:
            best_distance = distance
            best_match = name

    confidence = 1 - best_distance  # Higher confidence if distance is smaller
    return (best_match if best_distance < 0.7 else "Unknown", confidence)


st.title("Live Facial Recognition with Streamlit & FaceNet")

# Start webcam
run_webcam = st.checkbox("Start Webcam")

if run_webcam:
    video_capture = cv2.VideoCapture(0)
    frame_display = st.empty()

    while run_webcam:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Detect face and get embeddings
        face = mtcnn(img)
        if face is not None:
            face = face.squeeze(0)  # Correct the tensor shape
            with torch.no_grad():
                face_embedding = facenet(face.unsqueeze(0))  # Now pass a 4D tensor

            # Recognize the face
            name, confidence = recognize_face(face_embedding)

            # Draw rectangles around detected faces
            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the video stream in Streamlit
        frame_display.image(frame, channels="BGR")

    video_capture.release()
else:
    st.write("Enable the checkbox above to start the webcam.")