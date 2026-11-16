# Sign Language Detection and Hand Gestures Recognition System

## Real-Time Indian Sign Language (ISL) Recognition and Speech Conversion

### About The Project

The Sign Language Detection and Hand Gestures Recognition System is a Python-based project developed to recognize Indian Sign Language (ISL) hand gestures in real time.

The system uses computer vision, hand landmark detection, and deep learning techniques to identify sign language gestures and convert recognized signs into text and voice output.

The main objective of this project is to reduce communication barriers between people who use sign language and people who do not understand it.

### Problem Statement

People with hearing and speech impairments often use sign language as their primary communication method. However, communication becomes difficult when others do not understand sign language.

This project aims to develop a real-time recognition system that detects hand gestures, identifies sign language words, generates meaningful text, and converts the output into speech.

### Project Objectives

- To detect hand gestures using a live camera.
- To extract hand landmarks for gesture recognition.
- To recognize trained Indian Sign Language words.
- To convert recognized words into sentences.
- To generate voice output from recognized sentences.
- To display prediction confidence.
- To maintain the history of recently recognized sentences.

### Implemented Features

- **Live Hand Detection:** Detects hand movements using a webcam.
- **Hand Landmark Extraction:** Extracts hand keypoints using MediaPipe.
- **Real-Time Sign Recognition:** Recognizes trained sign language gestures.
- **Confidence Level:** Displays the confidence score of predictions.
- **Word-to-Sentence Conversion:** Combines recognized words into sentences.
- **Sentence-to-Voice Conversion:** Converts recognized sentences into speech.
- **Sentence History:** Maintains a history of the previous 2–3 recognized sentences.

### Technologies and Tools Used

- **Python 3.11:** Core programming language.
- **OpenCV:** Live camera access and image processing.
- **MediaPipe:** Hand detection and landmark extraction.
- **NumPy:** Numerical operations and feature processing.
- **Graph Convolutional Network (GCN):** Spatial hand landmark relationship learning.
- **Bidirectional Long Short-Term Memory (BiLSTM):** Sequential gesture information processing.
- **Text-to-Speech (TTS):** Conversion of recognized sentences into voice output.
- **Visual Studio Code:** Development environment.
- **Git and GitHub:** Version control and project hosting.

### Algorithms Used

**1. Graph Convolutional Network (GCN)**

GCN is used to learn spatial relationships between hand landmarks. It helps analyze the structural relationships between different hand keypoints.

**2. Bidirectional Long Short-Term Memory (BiLSTM)**

BiLSTM processes sequential data in both forward and backward directions. It helps learn temporal patterns from gesture sequences.

### System Workflow

1. Live Camera Input
2. Hand Detection
3. Hand Landmark Extraction
4. Gesture Feature Processing
5. Sign Word Recognition
6. Confidence Score Calculation
7. Word-to-Sentence Formation
8. Sentence-to-Voice Conversion
9. Sentence History Management

### Project Structure

The project is organized into the following main directories and files:

- **dataset:** Contains custom gesture training samples.
- **models:** Stores trained recognition models.
- **module1_sign_to_speech:** Contains the camera, live prediction, and model training scripts.
- **pretrained_isl:** Contains the pretrained ISL project resources.
- **hand_landmarker.task:** MediaPipe hand landmark detection model.
- **requirements.txt:** Required Python dependencies.
- **.gitignore:** Specifies files and folders excluded from Git tracking.
- **README.md:** Project documentation.

### Custom Dataset

A custom dynamic gesture dataset was created for initial model training.

The dataset contains four sign classes:

- HELLO – 20 samples
- PLEASE – 20 samples
- SORRY – 20 samples
- THANK_YOU – 20 samples

**Total Samples:** 80

Each sample contains a sequence of 30 frames with 126 hand landmark features.

Dataset Shape: (80, 30, 126)

The current trained vocabulary is limited to these four sign classes. Additional gesture classes can be incorporated through further dataset collection and model training.

### Installation and Execution

**Prerequisites**

- Python 3.11
- Visual Studio Code
- Webcam
- Windows Operating System

**Step 1: Clone the Repository**

    git clone https://github.com/suhanishende06-rgb/Sign-Language-Detection-and-Hand-gestures-Recognition-System.git

**Step 2: Navigate to the Project Directory**

    cd Sign-Language-Detection-and-Hand-gestures-Recognition-System

**Step 3: Create a Virtual Environment**

    python -m venv .venv

**Step 4: Activate the Virtual Environment**

    .venv\Scripts\Activate.ps1

**Step 5: Install Required Libraries**

    pip install -r requirements.txt

**Step 6: Run the Camera Module**

    python module1_sign_to_speech/camera.py

**Step 7: Run Live Sign Recognition**

    python module1_sign_to_speech/live_prediction.py

**Step 8: Train the Word Recognition Model**

    python module1_sign_to_speech/train_word_model.py

Note: Run the training script when model training or retraining is required. Ensure that the required model files and dependencies are available.

### Future Scope

- Increase the number of recognized Indian Sign Language gestures.
- Improve continuous sign language sentence recognition.
- Expand the dataset with more daily-use signs.
- Improve recognition accuracy under different lighting and camera conditions.
- Add multilingual text and voice output.
- Develop a user-friendly interface.

### Applications

- Communication assistance for deaf and hard-of-hearing individuals.
- Educational institutions.
- Sign language learning and practice.
- Accessibility-focused applications.
- Human-computer interaction.

### Developer

**Suhani Shende**

GitHub: https://github.com/suhanishende06-rgb

### Project Information

**Project Title:** Sign Language Detection and Hand Gestures Recognition System

**Domain:** Artificial Intelligence, Machine Learning, Deep Learning and Computer Vision.

**Project Type:** Academic / Engineering Project

**Repository:** https://github.com/suhanishende06-rgb/Sign-Language-Detection-and-Hand-gestures-Recognition-System

### License

This project is developed for educational and academic purposes.
