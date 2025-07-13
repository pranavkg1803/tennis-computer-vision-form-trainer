🎾 Tennis Motion Analysis System
A real-time computer vision system for tennis coaching that compares a user's form to professional players and delivers immediate, personalized feedback for improvement. This project integrates pose estimation, object detection, and novel movement analysis to assist players in refining their technique.

✅ Key Features
Real-time Pose Comparison: Match user movements with professional players frame-by-frame.

Dual Transformation Methods: Adjusts for body type variations using Scaled and Basic transforms.

Interactive Feedback: Visual cues including color-coded joints and directional arrows.

Tennis Racket Detection: YOLOv8-based object detection for racket tracking.

Problem Segment Identification: Highlights where form deviates most significantly.

Chain-Based Similarity Algorithm: Compares key kinematic chains for robust motion analysis.

Joint-specific Feedback: Detailed corrective advice on individual joint movements.

Timeline Visualization: See performance progression and replay problem segments.

⚙️ Setup Instructions
Create Virtual Environment

bash
Copy
Edit
python -m venv venv
Activate the Environment

Windows:

bash
Copy
Edit
venv\Scripts\activate
macOS/Linux:

bash
Copy
Edit
source venv/bin/activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
💻 System Requirements
Python 3.8 or higher

Webcam access

Adequate CPU/GPU for real-time inference (Intel i5 or equivalent recommended)

📦 Main Dependencies
OpenCV – Image and video processing

MediaPipe – Pose estimation and landmark tracking

PyQt6 – GUI framework

Ultralytics YOLOv8 – Racket detection

NumPy – Numerical computing

DTAIDistance – Dynamic Time Warping library

Full list available in requirements.txt.

🚀 Running the Application
Start the GUI:

bash
Copy
Edit
python gui.py
Using the System:

Upload a video of a professional player via the "Upload" button.

Choose the shot type and player.

Select desired options. Ensure you include the Chain-Based Similarity Algorithm for feedback to function correctly.

Start recording your shot with the webcam and follow the real-time visual guidance.

After the shot, check your similarity score (lower is better).

Visit the "Post-Shot Analysis" tab to:

Analyze problematic segments.

Replay key movement issues.

Receive actionable feedback for individual joints.

Use this information to improve and try again.

🧠 System Architecture
🎛️ Main Interface
A PyQt6-based GUI providing live feedback, similarity scores, and post-analysis tools.

📐 Pose Estimation
Utilizes MediaPipe for accurate 33-point skeleton tracking in real-time.

🔍 Similarity Analysis
Implements a custom chain-based algorithm analyzing four major kinematic chains:

Left/Right Arms

Left/Right Legs
Includes joint weighting to reflect motion significance.

🧬 Transformation Methods
Scaled Transform: Matches user body proportions to reference.

Basic Transform: Keeps overlay consistent for baseline comparisons.

🏸 Racket Detection
YOLOv8n model trained for fast and accurate racket localization in live feed.

📊 Feedback & Visualization
Transformed skeleton overlays

Color-coded joints indicating deviation severity

Directional arrows for corrective movement

Timeline & replay of incorrect segments

🗂️ Project Structure
File	Description
SportsMotionGUI.py	Main application and PyQt6 GUI logic
duelWebCam.py	Live webcam processing and pose tracking
chainSimilarity.py	Custom chain-based similarity algorithm
formVisualisation.py	Renders visual feedback overlays
feedback_widget.py	GUI components for feedback display
visualization_widgets.py	Timeline and analysis widgets
similarity_metrics.py	Additional similarity calculation methods

📌 About the Project
 It showcases practical application of:

Computer Vision

Real-time Inference

GUI Development

Algorithm Design (Chain-Based Similarity)

It is designed to bridge the gap between elite athletic form and amateur practice by providing accessible, intelligent feedback.
