Project Overview

The goal of this system is to combine computer vision and machine learning to detect a user's facial emotion and recommend songs accordingly. For instance, if the user appears happy, upbeat music is played; if sad, calming music is suggested.

Key Features

- Real-time facial emotion detection using webcam.
- Emotion classification into categories such as Happy, Sad, Angry, Neutral, and Surprise.
- Music mapping based on recognized emotions.
- Uses trained emotion datasets saved in `.npy` format.
- Generates emotion classification metrics and confusion matrix.

Methodology

1. Data Collection & Preprocessing
   - Used labeled facial emotion datasets.
   - Converted and saved preprocessed emotions as `.npy` arrays for fast loading.

2. Model Training
   - Facial features extracted using convolutional neural networks (CNNs).
   - Trained model saved and used for real-time emotion detection.

3. Emotion Detection
   - Real-time webcam feed is processed.
   - Detected face is analyzed for expression classification (Happy, Sad, Angry, etc.).

4. Music Recommendation
   - Based on detected emotion, the system fetches corresponding songs from a predefined playlist.

5. Visualization & Evaluation
   - Model performance evaluated using metrics and confusion matrix.
   - Metrics saved in `expression_metrics.txt`.

Installation & Setup

Prerequisites

- Python 3.7+
- OpenCV (for image capture and face detection)
- TensorFlow / Keras (for CNN modeling)
- NumPy (for handling emotion arrays)
- Matplotlib (for visualizations)

Steps to Run

1. Clone the repository or unzip the project  
   Place the project in your working directory.

2.Install required libraries
pip install opencv-python numpy matplotlib tensorflow

Run the Application

Make sure your webcam is working.

Launch the main script:

python main.py
(Assuming main.py is the script that runs the emotion detection & recommendation logic.)

Customize the Playlist

Modify the emotion-to-music mapping section inside the script to include your favorite songs or audio paths.

Libraries Used

Library	                              Purpose
OpenCV	                      Webcam access & face detection
TensorFlow/Keras	    Building and training emotion model
NumPy	                          Array manipulation
Matplotlib	           Confusion matrix & metrics plot