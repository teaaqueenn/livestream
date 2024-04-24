# imports
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import time
import cv2

eastFile = r"C:\Users\27GracieF\Documents\livestream\Program\frozen_east_text_detection.pb"
videoFile = r"C:\Users\27GracieF\Documents\livestream\Program\practiceFootage.mp4"

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
    ret, frame = cap.read()
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
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # draw the bounding box on the frame
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # update the FPS counter
        fps.update()
        # show the output frame
        cv2.imshow("Text Detection", orig)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# release the file pointer
cap.release()
# close all windows
cv2.destroyAllWindows()