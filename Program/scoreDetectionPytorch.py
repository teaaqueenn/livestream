import os
import cv2
import time
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import imutils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
model.load_state_dict(torch.load(r"C:\Users\27GracieF\Documents\livestream\Program\results\modelSDG_epoch10.pth"))
model.eval()


# Define transformations for pre-trained model
transform = transforms.Compose([
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def upload_digit(digit_image, predicted_digit, confidence, data_dir):
    print("[INFO] uploading digits...")
    """
    Saves the digit image and data (predicted digit, confidence) to a file.
    """
    filename = f"digit_{time.time()}.jpg"
    filepath = os.path.join(data_dir, filename)
    digit_img_np = digit_img.cpu().numpy().squeeze(0)
    print(digit_img_np.shape)
    cv2.imwrite(filepath, digit_img_np)

    with open(os.path.join(data_dir, f"data_{filename}.txt"), "w") as f:
        f.write(f"{predicted_digit},{confidence}")

    print(f"Digit saved to: {filepath}, Data saved to: data_{filename}.txt")


# Define paths and variables
current_directory = os.getcwd()
#servingTeam = 1
currentframe = 0
score1 = 10
#score2 = 5
#data_dir = 'data'
#os.makedirs(data_dir, exist_ok=True)  # Create data directory if it doesn't exist

# Video capture
video_path = r"C:\Users\27GracieF\Documents\livestream\Program\practiceFootage.mp4"

pracVid = cv2.VideoCapture(video_path)

while True:
    print("[INFO] While Loop Started...")
    time.sleep(1)

    # Read video frame
    ret, frame = pracVid.read()
    print("[INFO] There is a Video File:", ret)
    # Process frame if video exists
    if ret:

        # Convert to grayscale, crop, threshold, and apply contrast/blur
        print("[INFO] Transforming Frame...")
        crop = frame[40:100, 45:80]
        cv2.imshow("crop",crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 70, 220, cv2.THRESH_BINARY)[1]
        blur = cv2.medianBlur(thresh, 7)


        
        # Extract digit image and preprocess
        print("[INFO] Processing Potential Regions...")
        digit_img = transform(Image.fromarray(blur)).unsqueeze(0)

        # Run prediction and get results
        with torch.no_grad():
            print("[INFO] predicting digit...")
            output = model(digit_img)
            print("[INFO] Output Shape:", output.shape)
            predicted_digit = torch.argmax(output, dim=1).item()
            score1 = predicted_digit
            print("[INFO] Prediction:", predicted_digit)
            confidence = torch.nn.functional.softmax(output, dim=1).max().item()
            confidence *= 100
            print("[INFO] Confidence:", confidence)
            print("[INFO] Transformed...")
            
        with open(r"C:\Users\27GracieF\Documents\livestream\Program\Score1.txt", "w") as team1File:
            if ret:
                if confidence >= 45:
                    team1File.truncate(0)  # Clear existing content
                    team1File.write(str(score1))
                    team1File.flush()
            else:
                print("[INFO] No Video...")
        
       # Display the frame with bounding boxes
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):  # Add a key press to exit
           break
            

    # Exit loop if video ends
    if not ret:
        break

# Release resources
pracVid.release()
cv2.destroyAllWindows()