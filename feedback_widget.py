from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QScrollArea, QProgressBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPalette
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

class FeedbackWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.feedback_data = None
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Movement Feedback")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Segment info
        self.segment_frame = QFrame()
        self.segment_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.segment_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 5px; padding: 5px;")
        segment_layout = QVBoxLayout(self.segment_frame)
        
        self.segment_label = QLabel("Analyzing movement...")
        self.segment_label.setStyleSheet("font-weight: bold;")
        segment_layout.addWidget(self.segment_label)
        
        self.score_label = QLabel("")
        segment_layout.addWidget(self.score_label)
        
        main_layout.addWidget(self.segment_frame)
        
        # Joint feedback
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.joint_container = QWidget()
        self.joint_layout = QVBoxLayout(self.joint_container)
        
        scroll.setWidget(self.joint_container)
        main_layout.addWidget(scroll)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.replay_button = QPushButton("Replay Problem Segment")
        self.replay_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.replay_button.setEnabled(False)
        self.replay_button.clicked.connect(self.on_replay_clicked)
        
        button_layout.addWidget(self.replay_button)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
    
    def update_feedback(self, feedback_data):
        """Update the widget with new feedback data"""
        self.feedback_data = feedback_data
        
        if not feedback_data:
            self.segment_label.setText("No feedback available")
            self.score_label.setText("")
            self.replay_button.setEnabled(False)
            self.clear_joint_feedback()
            return
        
        # Update segment information
        worst_segment = feedback_data.get("worst_segment", {})
        start_frame = worst_segment.get("start_frame")
        end_frame = worst_segment.get("end_frame")
        score = worst_segment.get("score")
        
        if start_frame is not None and end_frame is not None and score is not None:
            self.segment_label.setText(f"Problem segment: Frames {start_frame} to {end_frame}")
            self.score_label.setText(f"Average similarity score: {score:.1f}%")
            self.replay_button.setEnabled(True)
        else:
            self.segment_label.setText("No problem segment identified")
            self.score_label.setText("")
            self.replay_button.setEnabled(False)
        
        # Update joint feedback
        self.clear_joint_feedback()
        problem_joints = feedback_data.get("problem_joints", [])
        
        for joint_data in problem_joints:
            self.add_joint_feedback(joint_data)
    
    def clear_joint_feedback(self):
        """Clear all joint feedback widgets"""
        # Remove all widgets from joint layout
        while self.joint_layout.count():
            item = self.joint_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def add_joint_feedback(self, joint_data):
        """Add a joint feedback panel"""
        joint_frame = QFrame()
        joint_frame.setFrameShape(QFrame.Shape.StyledPanel)
        joint_frame.setStyleSheet("background-color: white; border-radius: 5px; margin: 5px; padding: 10px;")
        
        joint_layout = QVBoxLayout(joint_frame)
        
        # Joint name and score
        header_layout = QHBoxLayout()
        
        friendly_name = joint_data.get("friendly_name", "Unknown Joint")
        joint_name_label = QLabel(friendly_name)
        joint_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(joint_name_label)
        
        score = joint_data.get("avg_score", 0)
        score_label = QLabel(f"Score: {score:.1f}%")
        header_layout.addWidget(score_label)
        
        header_layout.addStretch()
        joint_layout.addLayout(header_layout)
        
        # Progress bar showing difference
        pro_avg = joint_data.get("pro_avg", 0)
        user_avg = joint_data.get("user_avg", 0)
        
        # Skip if values are too close
        if abs(pro_avg - user_avg) > 0.1:
            comparison_label = QLabel(f"Professional: {pro_avg:.1f}° | Your average: {user_avg:.1f}°")
            joint_layout.addWidget(comparison_label)
            
            # Add visualization of the difference
            self.add_comparison_visualization(joint_layout, pro_avg, user_avg)
        
        # Action feedback
        feedback_text = joint_data.get("feedback", "")
        if feedback_text:
            feedback_label = QLabel(feedback_text)
            feedback_label.setStyleSheet("color: #D32F2F; font-weight: bold;")
            feedback_label.setWordWrap(True)
            joint_layout.addWidget(feedback_label)
        
        # Add joint frame to main layout
        self.joint_layout.addWidget(joint_frame)
    
    def add_comparison_visualization(self, layout, pro_value, user_value):
        """Add a visual comparison between pro and user values"""
        # Calculate the range to show
        value_diff = abs(pro_value - user_value)
        display_min = min(pro_value, user_value) - (value_diff * 0.2)
        display_max = max(pro_value, user_value) + (value_diff * 0.2)
        
        # Scale values to 0-100 for progress bar
        scaled_pro = 100 * (pro_value - display_min) / (display_max - display_min)
        scaled_user = 100 * (user_value - display_min) / (display_max - display_min)
        
        # Create a container with scale labels
        scale_layout = QHBoxLayout()
        min_label = QLabel(f"{display_min:.0f}°")
        max_label = QLabel(f"{display_max:.0f}°")
        scale_layout.addWidget(min_label)
        scale_layout.addStretch()
        scale_layout.addWidget(max_label)
        layout.addLayout(scale_layout)
        
        # Pro value marker
        pro_bar = QProgressBar()
        pro_bar.setMaximumHeight(15)
        pro_bar.setTextVisible(False)
        pro_bar.setRange(0, 100)
        pro_bar.setValue(int(scaled_pro))
        
        # Set green color for pro bar
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 175, 80))
        pro_bar.setPalette(palette)
        
        pro_layout = QHBoxLayout()
        pro_layout.addWidget(QLabel("Pro:"))
        pro_layout.addWidget(pro_bar)
        layout.addLayout(pro_layout)
        
        # User value marker
        user_bar = QProgressBar()
        user_bar.setMaximumHeight(15)
        user_bar.setTextVisible(False)
        user_bar.setRange(0, 100)
        user_bar.setValue(int(scaled_user))
        
        # Set blue color for user bar
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Highlight, QColor(33, 150, 243))
        user_bar.setPalette(palette)
        
        user_layout = QHBoxLayout()
        user_layout.addWidget(QLabel("You:"))
        user_layout.addWidget(user_bar)
        layout.addLayout(user_layout)
    
    def on_replay_clicked(self):
        """Handle replay button click"""
        print("Replay button clicked")
        
        if self.feedback_data and hasattr(self.parent(), 'replay_segment'):
            worst_segment = self.feedback_data.get("worst_segment", {})
            start_frame = worst_segment.get("start_frame")
            end_frame = worst_segment.get("end_frame")
            
            print(f"Worst segment frames: {start_frame} to {end_frame}")
            
            if start_frame is not None and end_frame is not None:
                print(f"Calling replay_segment({start_frame}, {end_frame})")
                self.parent().replay_segment(start_frame, end_frame)
        else:
            print("Cannot replay: missing feedback data or parent does not have replay_segment method")
            print(f"feedback_data exists: {self.feedback_data is not None}")
            print(f"Parent has replay_segment: {hasattr(self.parent(), 'replay_segment')}")

class FeedbackTimeline(FigureCanvasQTAgg):
    """Widget to display similarity timeline with highlighted problem segments"""
    def __init__(self, parent=None, width=8, height=3):
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        
        # Set background to match application
        fig.patch.set_facecolor('#f0f0f0')
        self.axes.set_facecolor('#f0f0f0')
        
        self.feedback_data = None
        self.temporal_scores = {}
    
    def update_data(self, temporal_scores, feedback_data=None):
        """Update the timeline with new data and feedback"""
        self.temporal_scores = temporal_scores
        self.feedback_data = feedback_data
        self.plot_timeline()
    
    def plot_timeline(self):
        """Plot the similarity timeline with problem segments highlighted"""
        self.axes.clear()
        
        if not self.temporal_scores:
            return
        
        # Plot overall timeline
        frames = list(self.temporal_scores.keys())
        scores = list(self.temporal_scores.values())
        
        self.axes.plot(frames, scores, 'b-', linewidth=2, label='Similarity')
        
        # Highlight problem segments if available
        if self.feedback_data:
            worst_segment = self.feedback_data.get("worst_segment", {})
            start_frame = worst_segment.get("start_frame")
            end_frame = worst_segment.get("end_frame")
            
            if start_frame is not None and end_frame is not None:
                # Get segment frames and scores
                segment_frames = [f for f in frames if start_frame <= f <= end_frame]
                segment_scores = [self.temporal_scores[f] for f in segment_frames]
                
                # Highlight problem segment
                self.axes.fill_between(segment_frames, 0, segment_scores, 
                                      color='#f44336', alpha=0.3, label='Problem Area')
                
                # Add vertical lines at boundaries
                self.axes.axvline(x=start_frame, color='#d32f2f', linestyle='--', linewidth=1)
                self.axes.axvline(x=end_frame, color='#d32f2f', linestyle='--', linewidth=1)
        
        # Format plot
        self.axes.set_xlabel('Frame')
        self.axes.set_ylabel('Similarity Score (%)')
        self.axes.set_title('Movement Similarity Over Time')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis to 0-100 scale
        self.axes.set_ylim(0, 100)
        
        # Add legend
        self.axes.legend(loc='lower left')
        
        self.draw()