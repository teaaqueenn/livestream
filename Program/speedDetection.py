import cv2
import numpy as np

# Define netHeight based on gender (modify as needed)
menNetHeight = 2.43  # meters
womenNetHeight = 1.8  # meters
netHeight = menNetHeight  # Change this for gender


def calculateSpeed(center, previousCenter, frameRate):
  # Calculate displacement based on frameRate
  displacement = abs(center[0] - previousCenter[0])
  # Convert pixel displacement to meters based on netHeight as reference
  distancePerPixel = netHeight / center[1]
  # Speed in meters per second
  speedMS = displacement * distancePerPixel / frameRate
  speedKMPH = speedMS * 3600/1000
  return speedKMPH


def detectBall(frame):
  # Image processing
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)
  crop = blur[0:1080, 0:1800]

  # Detect circles using HoughCircles (adjust parameters as needed)
  circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, 1.2, 100,
                             param1=50, param2=30, minRadius=10, maxRadius=30)

  if circles is not None:
    # Assume the largest circle is the ball
    circles = np.round(circles[0, :]).astype("int")
    largestCircle = max(circles, key=cv2.contourArea)
    return largestCircle[0]  # Return center coordinates
  else:
    return None


def main():
  # Initialize video capture
  cap = cv2.VideoCapture(0)  # Change 0 to video file path if needed

  # Get frameRate (might need adjustment based on video source)
  frameRate = cap.get(cv2.CAP_PROP_FPS)

  # Track previous ball center
  previousCenter = None

  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect ball using HoughCircles
    center = detectBall(frame)

    # Process and display results if ball is detected
    if center is not None:
      # Calculate speed if previous center exists
      if previousCenter is not None:
        speed = calculateSpeed(center, previousCenter, frameRate)
        # Place speed into text file
        with open("ballSpeed.txt", "a") as speedFile:
            if speed > 30:
                speedFile.truncate(0)  # Clear existing content
                speedFile.write(str(speed))
                speedFile.flush()
            else:
                print("speed was too slow")

      # Update previous center for next frame
      previousCenter = center

      # Draw circle around ball center (for proofing only)
      #cv2.circle(frame, (center[0], center[1]), 5, (0, 0, 255), -1)

    # Display the resulting frame
    #cv2.imshow('Volleyball Speed Detection', frame)

    # Switch netHeight with 'n' key press (modify key as needed)
    if cv2.waitKey(1) == ord('n'):
      netHeight = menNetHeight if netHeight == womenNetHeight else womenNetHeight
      print(f"Net height switched to: {netHeight} meters")

    # Quit with 'q' key press
    if cv2.waitKey(1) == ord('q'):
      break

  # Release capture
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()