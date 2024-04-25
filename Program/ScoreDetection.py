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

eastFile = r"C:\Users\27GracieF\Documents\livestream\Program\frozen_east_text_detection.pb"
videoFile = r"C:\Users\27GracieF\Documents\livestream\Program\practiceFootage.mp4"

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
def decode_predictions(scores, geometry):
	# grab  number of rows and columns from the scores volume
	# initialize bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract probabilities
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# i stole the math for this so i can't really explain it
			# not have enough probability --> ignore it
			if scoresData[x] < 0.6:
				continue
			#  offset factor
    		# maps 4x smaller than input 
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# rotation angle for prediction --> compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use geometry volume to derive the width and height of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add bounding box coordinates + probability score to respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of bounding boxes + associated confidences
	return (rects, confidences)

# initialize og frame dimensions + new dimensions + ratio between
(W, H) = (None, None)
(newW, newH) = (320, 320)
(rW, rH) = (None, None)

# define 2 output layer names for the model
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(eastFile)

# load the video file
print("[INFO] loading video file...")
cap = cv2.VideoCapture(videoFile)

# start the FPS estimator
fps = FPS().start()

while True:
    print("[INFO] While Loop Started...")
    time.sleep(1)

    # Read video frame
    ret, frame = cap.read()
    print("[INFO] There is a Video File:", ret)
    # Process frame if video exists
    if ret:
        
        # resize the frame (maintain the aspect ratio)
        frame = imutils.resize(frame, width=1000)
        orig = frame.copy()
        # if our frame dimensions are None, we still need to compute the
        # ratio of old frame dimensions to new frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            rW = W / float(newW)
            rH = H / float(newH)
        # resize the frame, this time ignoring aspect ratio
        frame = cv2.resize(frame, (newW, newH))
    
        # construct a blob from the frame and then perform a forward pass
        blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        # decode the predictions, then apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW * 1.2)
            startY = int(startY * rH * 1.2)
            endX = int(endX * rW * 1.2)
            endY = int(endY * rH * 1.2)

        # Convert to grayscale, crop, threshold, and apply contrast/blur
            print("[INFO] Transforming Frame...")
            crop = frame[startY:endY, startX:endX]
            if crop.size == 0:
                print("[WARNING] Crop region is empty. Skipping...")
                continue
            else:
                print("[INFO] crop locations...", startY, endY, startX, endX)
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
                    team1File.truncate(0)  # Clear existing content
                    team1File.write(str(score1))
                    team1File.flush()
                else:
                    print("[INFO] Could not upload scores...")
                    break
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # update the FPS counter
        fps.update()
        # show the output frame
        cv2.imshow("Text Detection", orig)
        
       # Display the frame with bounding boxes
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):  # Add a key press to exit
           break
            

    # Exit loop if video ends
    if not ret:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()