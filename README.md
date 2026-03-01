# 🚁 TRINETRA: AI-Powered Multi-Mode Disaster Rescue and Human Detection System
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Problem Statement

In recent years, both natural and man-made disasters have increased significantly. Many people lose their lives not because rescue teams do not respond, but because victims are not located in time.

Traditional rescue operations rely heavily on human visual search. However, the human eye is limited to line-of-sight visibility, making it difficult to detect victims in:

* Disaster rubble
* Forest environments
* Marine regions
* Low visibility conditions
* Large-scale disaster zones

This delay in detection directly reduces rescue success rate.

---

## Our Solution: TRINETRA

TRINETRA is an AI-powered multi-mode human detection and rescue assistance system designed to detect humans and critical objects in real time using drone or mobile camera video.

Unlike traditional systems, TRINETRA does not depend on specialized drone hardware or same-network connectivity between drone and ground station.

Instead, TRINETRA uses a simple and powerful approach:

* A mobile phone is attached to the drone
* The phone streams live video via video call
* The laptop receives the video
* TRINETRA captures the screen in real time
* AI processes the video and detects humans instantly

This removes major hardware and network limitations.

---

## Key Innovation

### Traditional System Limitations

Previous systems required:

* Drone camera and laptop in same network
* Dedicated transmission hardware
* Expensive drone computing systems
* Edge processing with delayed result viewing

If drone is lost or camera fails, system fails.

---

### TRINETRA Advantages

TRINETRA removes these limitations:

* Works using any mobile phone camera
* No need for same WiFi network
* Works through video call streaming
* No special drone hardware required
* Low cost and highly scalable
* Works even with signal boosters or antennas
* Can use high-resolution modern phone cameras (50MP, 64MP)

This makes the system flexible, reliable, and practical.

---

## Multi-Mode Detection System

TRINETRA supports multiple detection modes based on environment.

User can select mode dynamically.

### Supported Modes

* Basic Human Detection
* Disaster Rescue Mode
* Forest Animal and Human Detection
* Marine Detection Mode
* Military Surveillance Mode
* Mining Safety Mode
* Vehicle Detection Mode

Each mode uses optimized AI model for specific environment.

---

## Thermal and Advanced Detection

TRINETRA includes:

* Simulated thermal vision detection
* COCO dataset human detection
* Multi-model detection pipeline
* Automatic capture of detected frames
* Evidence storage for rescue analysis

This improves detection in difficult environments.

---

## How the System Works

1. Mobile phone is attached to drone
2. Phone starts video call to laptop
3. Laptop runs TRINETRA AI system
4. System captures live screen video
5. AI processes video using YOLOv8 models
6. Humans and objects are detected in real time
7. Detected frames are automatically saved
8. Rescue teams receive immediate visual intelligence

---

## System Architecture

Mobile Camera (Drone Mounted)
↓
Video Call Transmission
↓
Laptop Screen Capture
↓
TRINETRA AI Detection Engine
↓
Human Detection Output
↓
Frame Capture and Storage
↓
Rescue Team Decision Support

---

## Technologies Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* PyTorch
* NumPy
* Flask
* MSS (Screen Capture)
* Pillow

---

## Project Structure

```
TRINETRA/
│
├── trinetra.py
├── README.md
├── LICENSE
├── requirements.txt
│
├── models/
│   ├── 01_basic_human.pt
│   ├── 02_disaster_real.pt
│   ├── 03_forest_animals.pt
│   ├── 04_marine.pt
│   ├── 05_vehicle.pt
│   ├── 06_army.pt
│   ├── 07_ship.pt
│   ├── 08_mining.pt
│
└── archived_frames/
```

---

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

Run the system:

```
python trinetra.py
```

---

## System Capabilities

TRINETRA can detect:

* Humans
* Animals
* Vehicles
* Marine objects
* Disaster survivors
* Military targets

The system automatically captures and stores detection evidence.

---

## Major Advantages

* Works with any phone camera
* No dedicated drone camera required
* No network restriction
* Low cost system
* Real-time detection
* Multi-mode environment detection
* Easy deployment
* Highly scalable

---

## Real-World Use Cases

* Disaster rescue operations
* Earthquake rescue
* Flood rescue
* Forest rescue missions
* Defense surveillance
* Wildlife monitoring
* Industrial safety

---

## Future Scope

Future improvements include:

* LiDAR integration
* Depth camera integration
* Victim distance estimation
* Rescue priority prediction using AI
* Fully autonomous drone integration
* Night vision enhancement
* Cloud-based rescue coordination

Even without drone, system can work using mobile camera mounted on long pole for ground rescue.

---

## Impact

TRINETRA significantly improves rescue efficiency by enabling faster and more accurate victim detection.

This can increase rescue success rate and save human lives.

---

## Developer

Teki Satya Bhargav
B.Tech Electronics and Communication Engineering
Vignan University
AI and Embedded Systems Developer

---

## Project Purpose

This project was developed for:

* AI Hackathon Submission
* Disaster Rescue Research
* Real-World Rescue Applications

---

## Project Tagline

TRINETRA – The Third Eye That Never Misses a Survivor

