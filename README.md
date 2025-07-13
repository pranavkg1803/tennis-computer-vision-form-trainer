Tennis Motion Analysis System
A real-time computer vision system for tennis coaching that compares users' form to professional players and provides immediate visual feedback and personalized guidance for improvement.
Features

Real-time pose comparison with professional tennis players
Two transformation methods to accommodate different body types
Interactive visual feedback with color-coded joint indicators
Tennis racket detection using YOLOv8
Problem segment identification
Chain-based similarity algorithm for accurate movement analysis
Detailed feedback with specific joint improvement suggestions
Timeline visualization for performance review

Setup Instructions

Create a new virtual environment:
python -m venv venv

Activate the virtual environment:

Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate


Install dependencies:
pip install -r requirements.txt


Environment Requirements

Python 3.8 or higher
Webcam access
Sufficient processing power for real-time computer vision (Intel i5 or equivalent recommended)

Dependencies
The main dependencies include:

OpenCV
MediaPipe
PyQt6
Ultralytics (YOLOv8)
NumPy
DTAIDistance

A complete list is available in the requirements.txt file.
Usage

RUN THE MAIN APPLICAITON:
python gui.py

WHEN USING THE APPLICATION:
First upload a video by clicking upload which should open a folder select. Click on the VIDEOS folder
to see all the options. Click on the proffesional player and shot type you want to hit. 
Select the options you want and start Comparison and you should be able to see a video feed of your self. 
MAKE SURE TO SELECT CHAIN BASED SIMILARITY ALGORITHM, as one of the similarity algorithms as the post shot feedback can only work if
the chain based similairty is selected as one of the algorithms (the other algorithms are only there for comparison sake)

When you are ready to do the tennis shot click start recording and do the shot using the live graphics to guide you
Now you should see your similairty score (the lower the score the better).
Next go to the post shot analysis tab to visualise where you shot was wrong. 

Click "Analyze Movement" to see detailed feedback and problem areas
Use the timeline and problem segment replay to understand specific improvement areas
You can see actionable feedback for the worst joint, telling you how you should have moved them instea
Now using that feedback repeat the whole process.













System Components
Main Interface
The PyQt6-based interface provides an intuitive user experience with real-time visual feedback, similarity scores, and detailed analysis tools.
Pose Estimation
MediaPipe is used for accurate pose estimation, tracking 33 body landmarks in real-time.
Similarity Analysis
A novel chain-based algorithm analyzes movement by considering four kinematic chains (left/right arms and legs) with proper weighting of joint dependencies.
Transformation Methods

Scaled Transform: Adapts the reference skeleton to match user proportions
Basic Transform: Provides a consistent overlay without scaling

Racket Detection
Uses YOLOv8n for efficient tennis racket detection, optimized for real-time performance while maintaining accuracy.
Feedback Visualization

Transformed skeleton overlays
Color-coded joint indicators
Directional correction arrows
Timeline visualization of similarity
Problem segment identification and replay

Project Structure

SportsMotionGUI.py: Main application with PyQt6 interface
duelWebCam.py: Video processing and pose estimation
chainSimilarity.py: Chain-based similarity algorithm
formVisualisation.py: Visual feedback rendering
feedback_widget.py: Feedback interface components
visualization_widgets.py: Visualization widgets for analysis
similarity_metrics.py: Similarity calculation algorithms