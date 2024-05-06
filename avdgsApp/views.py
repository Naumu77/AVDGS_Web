# views.py

from django.shortcuts import render
import cv2
import numpy as np
import torch
from torchvision import transforms
 # Assuming plane.pt is the PyTorch model file
 
def Plane():
    model_path = 'models/plane.pt'

# Load the PyTorch model

onnx_model = Plane()
onnx_model.load_state_dict(torch.load(onnx_model))
onnx_model.eval()

# Define any necessary transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640,640)),
    transforms.ToTensor(),
])

def process_frame(frame):
    # Perform any necessary preprocessing on the frame
    # (e.g., resizing, normalization)
    processed_frame = transform(frame).unsqueeze(0)

    # Perform inference using the PyTorch model
    with torch.no_grad():
        outputs = onnx_model(processed_frame)

    # Process the model outputs (e.g., draw bounding boxes)
    # You'll need to implement this based on your model's output format
    # and how you want to display the results

    return processed_frame

def camera_feed(request):
    # Initialize the camera
    camera = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the processed frame
        cv2.imshow('Camera Feed', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()

    return render(request, 'avdgsApp/camera_feed.html')
