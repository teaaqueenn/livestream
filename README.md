# livestream


**Project Overview**

Over a few weeks, I have been working on a solution designed to enhance the IASAS Live Stream—a major event that connects student communities across international schools for various activities and competitions.
The goal of this project was to improve text detection during the live stream, which is crucial for displaying accurate and real-time information. 


**Investigating Solutions**

I began by exploring a range of technologies to address this need.
Initially, I tried using Google Tesseract for Optical Character Recognition (OCR). Unfortunately, it struggled with detecting numbers and provided inaccurate bounding boxes around the text. Next, I turned to TensorFlow and PyTorch, comparing both to find the most effective tool. TensorFlow’s complexity was a barrier, so I shifted my focus to PyTorch, which proved to be more manageable and suitable for my needs.


**Technical Challenges**

Even with PyTorch, the process wasn’t straightforward. I developed a custom text detection algorithm, building various layers and running extensive training sessions with different images. Initially, I experimented with pre-trained datasets like MNIST, but these did not work as well. I also looked into YOLO-V8, but its focus on object detection rather than digit detection wasn’t ideal.


**Developing the Model**

Ultimately, I decided to build the model from scratch. After multiple training epochs, starting with 5 and achieving 97% accuracy, I found that increasing to 10 epochs improved the accuracy to 99%. This enhancement led me to finalize the model’s configuration. To visualize the training progress, I used Matplotlib to plot loss metrics, which was instrumental in tracking the model’s performance throughout its development.
In the final stages, I explored an additional program for generating bounding boxes around text to streamline the process. While this approach showed some promise, it was less reliable than manually cropping the text, so I opted to retain the cropping method in the final implementation.


**Final Thoughts**

In summary, the model is effective and accurate, though it has some inconsistencies. Moving forward, I plan to create a custom dataset to further improve the model’s accuracy for the IASAS Live Stream.
