Aura
========

AI-Powered Assistive Vision System
----------------------------------

Aura is a real-time AI vision assistant designed to support visually impaired individuals in understanding their surroundings and locating objects through voice guidance.

The system integrates computer vision, speech recognition, and speech synthesis to create a fully hands-free assistive navigation experience.

* * * * *

1\. Overview
------------

Aura operates in two primary modes:

### 1\. Scene Understanding

Continuously analyzes the environment and provides periodic voice descriptions of what is in front of the user.

### 2\. Object Search Mode

Allows the user to search for a specific object using voice input and receive directional guidance (left, right, straight).

The system runs in real time using:

-   A laptop webcam, or

-   A mobile phone camera via DroidCam

* * * * *

2\. Key Features
----------------

### Real-Time Scene Description

-   Continuous environmental scanning

-   Voice output every 10 seconds

-   Natural language descriptions generated using image captioning

Example output:

> "I see a person sitting at a desk with a laptop."

* * * * *

### Voice-Activated Object Search

-   Press **ENTER** to activate search mode

-   System prompts: *"What would you like to find?"*

-   User speaks the object name

-   System detects and tracks the object

-   Provides real-time voice guidance:

    -   Move left

    -   Move right

    -   Straight ahead

-   Confirms when object is centered

* * * * *

### Fully Voice-Based Interaction

-   All system responses are spoken aloud

-   No screen interaction required

-   Designed for accessibility and hands-free use

* * * * *

3\. System Architecture
-----------------------

Aura integrates multiple AI components into a unified pipeline.

### Core Components

#### 1\. BLIP -- Image Captioning

-   Generates natural-language descriptions from camera frames

-   Used in Scene Understanding mode

#### 2\. YOLOv8 -- Object Detection

-   Performs real-time object detection

-   Calculates object position within the frame

-   Enables directional guidance

#### 3\. Speech Recognition

-   Captures user voice input

-   Extracts object name for search mode

#### 4\. Speech Synthesis

-   Converts system responses to voice

-   Uses Windows SpeechSynthesizer

* * * * *

4\. Visual Program Flow
-----------------------

```

+-------------------+
|   Start Program   |
+-------------------+
          |
          v
+-------------------+
|   Load AI Models  |
|  (YOLOv8 + BLIP)  |
+-------------------+
          |
          v
+---------------------------+
|       Normal Mode         |
|   Scene Description       |
|   (Every 10 Seconds)      |
+---------------------------+
          |
          | Press ENTER
          v
+---------------------------+
|     Object Search Mode    |
+---------------------------+
          |
          v
+---------------------------+
|    Capture Voice Input    |
|      (Object Name)        |
+---------------------------+
          |
          v
+---------------------------+
|   YOLO Object Detection   |
+---------------------------+
          |
          v
+---------------------------+
|  Calculate Object Position|
+---------------------------+
          |
          v
+---------------------------+
|   Voice Guidance Output   |
|  (Left / Right / Center)  |
+---------------------------+
          |
          v
+---------------------------+
|   Return to Normal Mode   |
+---------------------------+

```

* * * * *

5\. Technologies Used
---------------------

-   Python 3.10
  
-   OpenCV

-   YOLOv8 (Ultralytics)

-   BLIP Image Captioning Model

-   Hugging Face Transformers

-   PyTorch

-   SpeechRecognition

-   PyAudio

-   Windows SpeechSynthesizer

* * * * *

6\. System Requirements
-----------------------

-   Windows 10 or Windows 11

-   Python 3.10

-   Webcam or DroidCam

-   Microphone

-   Internet connection (first-time model download)

* * * * *

7\. Installation Guide
----------------------

### Step 1 -- Install Python

Download Python 3.10:

<https://www.python.org/downloads/>

Ensure that **"Add Python to PATH"** is selected during installation.

* * * * *

### Step 2 -- Install Required Libraries

Open Command Prompt and run:

`pip install opencv-python ultralytics transformers torch pillow speechrecognition pyaudio`

If PyAudio fails on Windows:

`pip install pipwin
pipwin install pyaudio`

* * * * *

8\. Camera Configuration
------------------------

### Option 1 -- Laptop Webcam

No configuration required.

### Option 2 -- Mobile Phone (DroidCam)

1.  Install DroidCam on your mobile device

2.  Connect phone and laptop to the same WiFi network

3.  Copy the IP address shown in DroidCam

4.  Modify the capture line in `main.py`:

`cap = cv2.VideoCapture("http://YOUR_IP:4747/video")`

Example:

`cap = cv2.VideoCapture("http://192.168.0.15:4747/video")`

* * * * *

9\. Running the Application
---------------------------

Navigate to the project directory and execute:

`python main.py`

* * * * *

10\. Controls
-------------

| Key | Function |
| --- | --- |
| ENTER | Activate object search mode |
| ESC | Exit application |

* * * * *

11\. Performance Considerations
-------------------------------

-   First run downloads approximately 1GB of AI models

-   Works best in well-lit environments

-   Background noise may reduce speech recognition accuracy

-   Ensure microphone permissions are enabled

* * * * *

12\. Future Improvements
------------------------

-   Distance estimation

-   Obstacle detection

-   Haptic feedback integration

-   Mobile application deployment

-   Offline speech recognition

-   Improved directional precision

* * * * *

13\. Project Objective
----------------------

Aura aims to provide an affordable and scalable AI-powered assistive solution that enhances independence and situational awareness for visually impaired individuals.
