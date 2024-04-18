import os
from PIL import Image
import cv2
import time
import pytesseract

# Correct Tesseract path (replace with your actual path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\27GracieF\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Get current directory
current_directory = os.getcwd()

# Set up variables
servingTeam = 1
currentframe = 0
score1 = 10
score2 = 5

# Change directory (check for errors)
try:
    os.chdir(r"C:\Users\27GracieF\OneDrive - Taipei American School\Desktop\Livestream\Program")
except FileNotFoundError:
    print("Error: Target directory not found!")
    exit()

# Choose video output
pracVid = cv2.VideoCapture('practice footage.mp4')

# Create data folder if it doesn't exist
try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('Error: Creating directory of data')

# Process
while True:
    time.sleep(1)

    # Read video
    ret, frame = pracVid.read()

    # If video exists, run the following
    if ret:
        print("Frame read successfully.")  # Print for debugging

        # Process frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 70, 220, cv2.THRESH_BINARY_INV)[1]
        contrast = cv2.convertScaleAbs(thresh, 3, 140)
        blur = cv2.medianBlur(contrast, 7)

        # Create filename
        name = os.path.join('data', f'frame{currentframe}.jpg')

        # Save image (check success)
        success = cv2.imwrite(name, blur)

        if success:
            # Proceed with text extraction only if saving is successful
            text = pytesseract.image_to_string(blur, lang="name(.traineddata)")
            print(text)

        # Open score files for writing (append mode)
        with open("Team 1 Score.txt", "a") as team1File, open("Team 2 Score.txt", "a") as team2File:
            if servingTeam == 1:
                team1File.truncate(0)  # Clear existing content
                team1File.write(str(score1))
                team1File.flush()
            elif servingTeam == 2:
                team2File.truncate(0)  # Clear existing content
                team2File.write(str(score2))
                team2File.flush()

            else:
                print(f'Error score for image: {name}')

    currentframe += 1
    # Exit loop if video ends
    if not ret:
        break

# Release resources
pracVid.release()
cv2.destroyAllWindows()