import os
from PIL import Image
import cv2
import time
import pytesseract

# Correct Tesseract path (replace with your actual path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\27GracieF\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Get current directory
current_directory = os.getcwd()


#try to read frame 
img = cv2.imread('frame0.jpg') 

#text = pytesseract.image_to_data(Image.open('testpic5.png'))
 
#print(text)

text = pytesseract.image_to_string(Image.open('testpic5.png'))
 
print(text)