from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox,
                              QFrame, QSlider, QScrollArea, QTabWidget, QSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QFont, QColor
import sys
import cv2
import os
import duelWebCam as vp
import chainSimilarity as cs 
from feedback_widget import FeedbackWidget, FeedbackTimeline

from visualization_widgets import SkeletonWidget, SimilarityGraph

class SportsMotionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sports Motion Analysis System")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 180px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QLabel {
                color: #333333;
            }
            QComboBox {
                padding: 6px;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                min-width: 180px;
                color: #333333;
                background-color: white;
            }
            QFrame {
                background-color: white;
                border-radius: 8px;
                margin: 2px;
            }
        """)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create main analysis tab
        self.main_tab = QWidget()
        self.main_layout = QVBoxLayout(self.main_tab)
        self.setup_main_tab()  # This will contain your current UI
        self.tab_widget.addTab(self.main_tab, "Main Analysis")

        # Create similarity analysis tab
        self.similarity_tab = QWidget()
        self.setup_similarity_tab()
        self.tab_widget.addTab(self.similarity_tab, "Similarity Analysis")
        self.setup_feedback_tab()
        self.setup_variables()
        

    def setup_variables(self):
        self.video_path = None
        self.pro = None
        self.user = None
        self.is_recording = False
        self.comparison_running = False
        self.should_run = True
        self.cap_webcam = None
        self.cap_video = None
        self.mirror_enabled = False
        self.similarity_algo1 = "Standard Similarity"
        self.similarity_algo2 = "Weighted Similarity"
        self.countdown_remaining = None
        self.countdown_timer = None
        self.video_duration = None
        self.feedback_data = None
        self.segment_replay_active = False
        self.replay_start_frame = None
        self.replay_end_frame = None
        self.replay_current_frame = None
        

    def create_options_panel(self):
        options_frame = QFrame()
        options_frame.setMaximumHeight(50)  # Minimal height for single row
        options_layout = QHBoxLayout(options_frame)
        options_layout.setContentsMargins(5, 5, 5, 5)
        options_layout.setSpacing(10)

        # Common style for all dropdowns
        dropdown_style = """
            QComboBox {
                background-color: white;
                color: #333333;
                padding: 6px;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                min-width: 180px;
            }
            QComboBox:drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                border: none;
                background: #666666;
                width: 8px;
                height: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #333333;
                selection-background-color: #2196F3;
                selection-color: white;
            }
        """

        # Transform selection
        self.transform_combo = QComboBox()
        self.transform_combo.setStyleSheet(dropdown_style)
        self.transform_combo.addItems(["Basic Transform", "Scaled Transform"])
        options_layout.addWidget(self.transform_combo)

        # Mirror toggle
        self.mirror_combo = QComboBox()
        self.mirror_combo.setStyleSheet(dropdown_style)
        self.mirror_combo.addItems(["Mirror: Disabled", "Mirror: Enabled"])
        self.mirror_combo.currentTextChanged.connect(lambda x: self.toggle_mirror(x.split(": ")[1]))
        options_layout.addWidget(self.mirror_combo)

        # First similarity algorithm selection
        self.sim1_combo = QComboBox()
        self.sim1_combo.setStyleSheet(dropdown_style)
        self.sim1_combo.addItems(["Similarity 1: Standard",
                                  "Similarity 1: Weighted",
                                  "Similarity 1: Chain-based"])
        self.sim1_combo.currentTextChanged.connect(self.update_similarity_algos)
        options_layout.addWidget(self.sim1_combo)

        # Second similarity algorithm selection
        self.sim2_combo = QComboBox()
        self.sim2_combo.setStyleSheet(dropdown_style)
        self.sim2_combo.addItems(["Similarity 2: Weighted",
                                  "Similarity 2: Standard",
                                  "Similarity 2: Chain-based"])
        self.sim2_combo.currentTextChanged.connect(self.update_similarity_algos)
        options_layout.addWidget(self.sim2_combo)

        # Add some stretch to keep dropdowns together
        options_layout.addStretch()

        return options_frame

    def setup_main_tab(self):
        #scroll = QScrollArea()
        #self.setCentralWidget(scroll)
        
        #main_widget = QWidget()
        #scroll.setWidget(main_widget)
        #scroll.setWidgetResizable(True)
        self.main_layout.setSpacing(2)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        # Compact header
        header = QFrame()
        header.setMaximumHeight(50)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 5, 5, 5)
        title = QLabel("Sports Motion Analysis")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1976D2;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        self.main_layout.addWidget(header)

        # Options panel
        options = self.create_options_panel()
        self.main_layout.addWidget(options)

        # Compact controls in one row
        controls = QFrame()
        controls.setMaximumHeight(70)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(10)

        # Upload section
        upload_button = QPushButton("Upload Reference Video")
        upload_button.clicked.connect(self.handle_upload)
        controls_layout.addWidget(upload_button)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #666666;")
        controls_layout.addWidget(self.file_label)

        # Action buttons
        self.comparison_button = QPushButton("Start Comparison")
        self.comparison_button.clicked.connect(self.start_comparison)
        self.comparison_button.setEnabled(False)
        controls_layout.addWidget(self.comparison_button)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        controls_layout.addWidget(self.record_button)
        
        self.stop_button = QPushButton("Stop Comparison")
        self.stop_button.clicked.connect(self.stop_comparison)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        self.main_layout.addWidget(controls)

        # Video displays
        displays = QFrame()
        displays_layout = QHBoxLayout(displays)
        displays_layout.setContentsMargins(5, 5, 5, 5)

        # Reference video
        ref_container = QVBoxLayout()
        ref_label = QLabel("Reference Video")
        ref_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        ref_label.setMaximumHeight(20)
        self.reference_display = QLabel()
        self.reference_display.setMinimumSize(600, 500)
        self.reference_display.setStyleSheet("border: 2px solid #BDBDBD; border-radius: 4px;")
        ref_container.addWidget(ref_label)
        ref_container.addWidget(self.reference_display)
        displays_layout.addLayout(ref_container)

        # Webcam display
        webcam_container = QVBoxLayout()
        webcam_label = QLabel("Your Movement")
        webcam_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        webcam_label.setMaximumHeight(20)
        self.webcam_display = QLabel()
        self.webcam_display.setMinimumSize(600, 500)
        self.webcam_display.setStyleSheet("border: 2px solid #BDBDBD; border-radius: 4px;")
        webcam_container.addWidget(webcam_label)
        webcam_container.addWidget(self.webcam_display)
        displays_layout.addLayout(webcam_container)

        self.main_layout.addWidget(displays)

        # Metrics section
        metrics = QFrame()
        metrics.setMaximumHeight(50)
        metrics_layout = QHBoxLayout(metrics)
        metrics_layout.setContentsMargins(5, 5, 5, 5)
        
        self.similarity_label = QLabel("Similarity Score: --")
        self.similarity_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.weighted_similarity_label = QLabel("Weighted Similarity: --")
        self.weighted_similarity_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(self.similarity_label)
        metrics_layout.addWidget(self.weighted_similarity_label)
        
        self.main_layout.addWidget(metrics)

        # Status
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #666666; font-size: 12px;")
        self.status_label.setMaximumHeight(30)
        self.main_layout.addWidget(self.status_label)
    
    def setup_feedback_tab(self):
        """Set up the feedback tab with detailed analysis widgets"""
        self.feedback_tab = QWidget()
        feedback_layout = QVBoxLayout(self.feedback_tab)
        
        # Create a horizontal layout for timeline and video display
        top_section = QHBoxLayout()
        
        # Left side: Timeline and controls
        timeline_container = QFrame()
        timeline_layout = QVBoxLayout(timeline_container)
        timeline_layout.setContentsMargins(5, 5, 5, 5)
        
        timeline_header = QLabel("Similarity Timeline")
        timeline_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #1976D2;")
        timeline_layout.addWidget(timeline_header)
        
        self.feedback_timeline = FeedbackTimeline(self)
        timeline_layout.addWidget(self.feedback_timeline)
        
        # Segment size selector - keep your existing controls
        segment_controls = QHBoxLayout()
        segment_controls.addWidget(QLabel("Problem segment size:"))
        
        self.segment_size_spinner = QSpinBox()
        self.segment_size_spinner.setRange(5, 50)
        self.segment_size_spinner.setValue(25)
        self.segment_size_spinner.setSuffix("%")
        self.segment_size_spinner.setToolTip("Size of problem segment as percentage of total motion")
        segment_controls.addWidget(self.segment_size_spinner)
        
        segment_controls.addWidget(QLabel("Problem joints to identify:"))
        
        self.problem_joints_spinner = QSpinBox()
        self.problem_joints_spinner.setRange(1, 5)
        self.problem_joints_spinner.setValue(3)
        segment_controls.addWidget(self.problem_joints_spinner)
        
        self.analyze_button = QPushButton("Analyze Movement")
        self.analyze_button.setStyleSheet("""
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
        self.analyze_button.clicked.connect(self.analyze_movement)
        self.analyze_button.setEnabled(False)
        segment_controls.addWidget(self.analyze_button)
        
        segment_controls.addStretch()
        timeline_layout.addLayout(segment_controls)
        
        # Add timeline to top section
        top_section.addWidget(timeline_container)
        
        # Right side: NEW Video display for problem segment replay
        video_container = QFrame()
        video_layout = QVBoxLayout(video_container)
        
        video_header = QLabel("Problem Segment Replay")
        video_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #1976D2;")
        video_layout.addWidget(video_header)
        
        # This is the new display area for the replayed video
        self.feedback_video_display = QLabel()
        self.feedback_video_display.setMinimumSize(400, 300)
        self.feedback_video_display.setStyleSheet("border: 2px solid #BDBDBD; border-radius: 4px;")
        self.feedback_video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback_video_display.setText("Click 'Replay Problem Segment' to view")
        video_layout.addWidget(self.feedback_video_display)
        
        # Add video container to top section
        top_section.addWidget(video_container)
        
        # Add top section to main layout
        feedback_layout.addLayout(top_section)
        
        # Feedback widget (existing code)
        self.feedback_widget = FeedbackWidget(self)
        feedback_layout.addWidget(self.feedback_widget)
        
        # Add to tab widget
        self.tab_widget.addTab(self.feedback_tab, "Feedback Analysis")
        
        # Variables for feedback replay
        self.feedback_data = None
        self.segment_replay_active = False
        self.replay_start_frame = None
        self.replay_end_frame = None
        self.replay_current_frame = None
        
        

    def toggle_mirror(self, value):
        self.mirror_enabled = (value == "Enabled")

    def update_similarity_algos(self):
        self.similarity_algo1 = self.sim1_combo.currentText().split(": ")[1] #to get rid of the first part, so we just have type
        self.similarity_algo2 = self.sim2_combo.currentText().split(": ")[1]    

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap with optional mirroring"""
        if self.mirror_enabled:
            cv_img = cv2.flip(cv_img, 1)
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        
        # If countdown is active, draw it on the pixmap
        if self.countdown_remaining is not None:
            painter = QPainter(pixmap)
            font = QFont()
            font.setPointSize(120)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(Qt.GlobalColor.white)
            painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 128))
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, str(self.countdown_remaining))
            painter.end()  # Add this to properly close the painter
            
        return pixmap

    def closeEvent(self, event):
        """Handle window close event"""
        self.should_run = False
        
        # Stop any active replay
        if hasattr(self, 'segment_replay_active') and self.segment_replay_active:
            self.stop_replay()
        
        if hasattr(self, 'cap_webcam') and self.cap_webcam is not None:
            self.cap_webcam.release()
        if hasattr(self, 'cap_video') and self.cap_video is not None:
            self.cap_video.release()
        if hasattr(self, 'cap_replay') and self.cap_replay is not None:
            self.cap_replay.release()
        
        cv2.destroyAllWindows()
        event.accept()

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Q:
            self.close()

    def handle_upload(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Video", "", "Video Files (*.mp4 *.avi)"
        )
        if file_name:
            self.video_path = file_name
            self.file_label.setText(f"Selected: {os.path.basename(file_name)}")
            self.status_label.setText("Status: Analyzing reference video...")
            
            self.video_duration = vp.get_video_duration(file_name)
            print(f"Loaded video duration: {self.video_duration} seconds") 
            self.pro = vp.PersonAngles()
            vp.analyzeVideo(self.pro, video_path=file_name)
            
            self.comparison_button.setEnabled(True)
            self.status_label.setText("Status: Reference video analyzed. Ready for comparison.")
    
    def toggle_recording(self):
        if not self.is_recording:
            # Start countdown
            self.start_countdown()
        else:
            self.record_button.setText("Start Recording")
            self.status_label.setText("Status: Processing recording...")

    #This allows us to go from recording to not recording and back
    def start_countdown(self):
        self.countdown_remaining = 5
        # Create and start countdown timer
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

    def start_countdown(self):
        self.countdown_remaining = 5
        # Create and start countdown timer
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)  # Fire every 1 second

    def update_countdown(self):
        self.countdown_remaining -= 1
        if self.countdown_remaining > 0:
            # Force a redraw of the video displays
            if self.reference_display.pixmap():
                self.redraw_countdown()
        else:
            self.countdown_timer.stop()
            self.countdown_remaining = None
            # Actually start recording
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            self.status_label.setText("Status: Recording in progress...")
            self.similarity_label.setText("Recording in progress...")
            self.weighted_similarity_label.setText("Recording in progress...")

    def redraw_countdown(self):
        if self.countdown_remaining is not None:
            # Get current pixmaps
            ref_pixmap = self.reference_display.pixmap()
            webcam_pixmap = self.webcam_display.pixmap()
            
            if ref_pixmap and webcam_pixmap:
                # Create painters for both displays
                ref_painter = QPainter(ref_pixmap)
                webcam_painter = QPainter(webcam_pixmap)
                
                # Set up text appearance
                font = QFont()
                font.setPointSize(120)
                font.setBold(True)
                
                # Add semi-transparent background
                ref_painter.fillRect(ref_pixmap.rect(), QColor(0, 0, 0, 128))
                webcam_painter.fillRect(webcam_pixmap.rect(), QColor(0, 0, 0, 128))
                
                # Draw countdown text
                ref_painter.setFont(font)
                ref_painter.setPen(Qt.GlobalColor.white)
                webcam_painter.setFont(font)
                webcam_painter.setPen(Qt.GlobalColor.white)
                
                countdown_text = str(self.countdown_remaining)
                
                # Center text in both displays
                ref_painter.drawText(ref_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, countdown_text)
                webcam_painter.drawText(webcam_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, countdown_text)
                
                # End painters properly
                ref_painter.end()
                webcam_painter.end()
                
                # Update displays
                self.reference_display.setPixmap(ref_pixmap)
                self.webcam_display.setPixmap(webcam_pixmap)
    
    def stop_comparison(self):
        self.should_run = False
        self.comparison_running = False
        self.comparison_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)
        
        # Stop any active replay
        if hasattr(self, 'segment_replay_active') and self.segment_replay_active:
            self.stop_replay()
        
        self.status_label.setText("Status: Comparison stopped")
        
        if hasattr(self, 'cap_webcam') and self.cap_webcam is not None:
            self.cap_webcam.release()
        if hasattr(self, 'cap_video') and self.cap_video is not None:
            self.cap_video.release()
        cv2.destroyAllWindows()

    def start_comparison(self):
        if not self.video_path or not self.pro:
            self.status_label.setText("Status: Please upload a reference video first")
            return

        self.comparison_running = True
        self.should_run = True
        self.status_label.setText("Status: Starting comparison...")
        self.comparison_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        processed_video_path = vp.store_reference_video(self.video_path)
        reference_pose_landmarks = vp.record_pose_landmarks(self.video_path)
        
        self.user = vp.PersonAngles()
        transform_method = self.transform_combo.currentText().lower().split()[0]
        self.record_button.setEnabled(True)

        def update_frames(ref_frame, webcam_frame):
            if self.comparison_running:
                ref_pixmap = self.convert_cv_qt(ref_frame)
                webcam_pixmap = self.convert_cv_qt(webcam_frame)
                
                self.reference_display.setPixmap(
                    ref_pixmap.scaled(
                        self.reference_display.size(),
                        Qt.AspectRatioMode.KeepAspectRatio
                    )
                )
                self.webcam_display.setPixmap(
                    webcam_pixmap.scaled(
                        self.webcam_display.size(),
                        Qt.AspectRatioMode.KeepAspectRatio
                    )
                )
        def on_recording_complete():
            if self.user and self.pro:
                path = vp.dtaidistance.dtw.warping_path(
                    self.pro.angles['rElbowAngles'],
                    self.user.angles['rElbowAngles']
                )
                print("Calculating chain-based similarity...")
                # Calculate similarities based on selected algorithms
                
                if self.similarity_algo1 == "Standard":
                    print("Calculating standard similarity...")
                    similarity1 = vp.sm.calculateSimilarity1(self.pro, self.user, path)
                    sim1_text = "Standard Similarity"
                elif self.similarity_algo1 == "Chain-based":
                    try:
                        print("Doing chain-based similarity...")
                        temporal_scores, chain_weights, joint_scores = cs.calculate_chain_based_similarity(
                            self.pro,
                            self.user,
                            path
                        )
                        
                        print(f"Got temporal scores: {len(temporal_scores)} frames")
                        print(f"Got joint scores for {len(joint_scores)} joints")
                        # Use average of temporal scores as overall similarity
                        similarity1 = sum(temporal_scores.values()) / len(temporal_scores)
                        sim1_text = "Chain-based Similarity"
                        
                        # Store results for visualization
                        self.temporal_scores = temporal_scores
                        self.joint_scores = joint_scores
                        self.timeline_slider.setMaximum(max(temporal_scores.keys()))
                        self.update_similarity_view(0)
                    except Exception as e:
                        print(f"Error in chain-based similarity calculation: {str(e)}")
                        print(f"Pro angles keys: {self.pro.angles.keys()}")
                        print(f"User angles keys: {self.user.angles.keys()}")
                else:
                    similarity1 = vp.sm.calculateSimilarityWeighted(self.pro, self.user, path)
                    print("Calculating weighted similarity...")
                    sim1_text = "Weighted Similarity"

                if self.similarity_algo2 == "Standard":
                    similarity2 = vp.sm.calculateSimilarity1(self.pro, self.user, path)
                    sim2_text = "Standard Similarity"
                elif self.similarity_algo2 == "Chain-based":
                    try:
                        print("Doing chain-based similarity...")
                        temporal_scores, chain_weights, joint_scores = cs.calculate_chain_based_similarity(
                            self.pro,
                            self.user,
                            path
                        )
                        
                        print(f"Got temporal scores: {len(temporal_scores)} frames")
                        print(f"Got joint scores for {len(joint_scores)} joints")
                        # Use average of temporal scores as overall similarity
                        similarity2 = sum(temporal_scores.values()) / len(temporal_scores)
                        sim2_text = "Chain-based Similarity"
                        
                        # Store results for visualization
                        self.temporal_scores = temporal_scores
                        self.joint_scores = joint_scores
                        self.timeline_slider.setMaximum(max(temporal_scores.keys()))
                        self.update_similarity_view(0)
                    except Exception as e:
                        print(f"Error in chain-based similarity calculation: {str(e)}")
                        print(f"Pro angles keys: {self.pro.angles.keys()}")
                        print(f"User angles keys: {self.user.angles.keys()}")
                else:
                    similarity2 = vp.sm.calculateSimilarityWeighted(self.pro, self.user, path)
                    sim2_text = "Weighted Similarity"
                
                self.similarity_label.setText(f"{sim1_text}: {similarity1:.2f}")
                self.weighted_similarity_label.setText(f"{sim2_text}: {similarity2:.2f}")
                
                self.analyze_button.setEnabled(True)
        
                self.status_label.setText("Status: Analysis complete. Click 'Analyze Movement' for detailed feedback.")
                self.record_button.setText("Start Recording")
                self.is_recording = False
                
                
    
            
        vp.analyze_video_and_webcam(
            self.user,
            processed_video_path,
            reference_pose_landmarks,
            transform_method=transform_method,
            frame_callback=update_frames,
            recording_callback=lambda: self.is_recording,
            completion_callback=on_recording_complete,
            should_run_callback=lambda: self.should_run,
            video_duration=self.video_duration
        )
    
    def setup_similarity_tab(self):
        """Set up the similarity analysis tab"""
        similarity_layout = QVBoxLayout(self.similarity_tab)
        
        # Timeline slider
        slider_container = QFrame()
        slider_layout = QHBoxLayout(slider_container)
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.valueChanged.connect(self.update_similarity_view)
        self.frame_label = QLabel("Frame: 0")
        slider_layout.addWidget(self.frame_label)
        slider_layout.addWidget(self.timeline_slider)
        similarity_layout.addWidget(slider_container)
        
        # Similarity graph
        self.similarity_graph = SimilarityGraph(self)
        similarity_layout.addWidget(self.similarity_graph)
        
        # Skeleton visualization
        self.skeleton_widget = SkeletonWidget(self)
        similarity_layout.addWidget(self.skeleton_widget)

    def update_similarity_view(self, frame_idx):
        # More comprehensive joint name mapping
        joint_name_mapping = {
            'lUpperArmVector': 'lShoulder',
            'rUpperArmVector': 'rShoulder',
            'lUpperLegVector': 'lHip',
            'rUpperLegVector': 'rHip',
            'lAnkleAngles': 'lAnkle',
            'rAnkleAngles': 'rAnkle',
            'lShoulderAngles': 'lShoulder',
            'rShoulderAngles': 'rShoulder',
            'lElbowAngles': 'lElbow',
            'rElbowAngles': 'rElbow',
            'lHipAngles': 'lHip',
            'rHipAngles': 'rHip',
            'lKneeAngles': 'lKnee',
            'rKneeAngles': 'rKnee'
        }
        
        print("Updating similarity view for frame:", frame_idx)
        
        # Plot temporal scores graph
        if hasattr(self, 'temporal_scores'):
            print("Temporal scores available")
            print("Temporal scores keys:", list(self.temporal_scores.keys()))
            print("Temporal scores values:", list(self.temporal_scores.values()))
            
            # Plot the similarity graph
            self.similarity_graph.plot_similarity(self.temporal_scores, frame_idx)
        else:
            print("No temporal scores available")
        
        #set joint colors
        if hasattr(self, 'joint_scores'):
            print("Joint scores available:", list(self.joint_scores.keys()))
            
            for joint_name, scores in self.joint_scores.items():
                # Map joint names, use original if no mapping
                mapped_joint_name = joint_name_mapping.get(joint_name, joint_name)
                
                try:
                    if frame_idx < len(scores):
                        score = scores[frame_idx]
                        
                        # Skip zero scores or very low scores
                        if score > 0:
                            print(f"Setting color for {joint_name} (mapped to {mapped_joint_name}): {score}")
                            self.skeleton_widget.set_joint_color(mapped_joint_name, score)
                        else:
                            print(f"Skipping zero score for {joint_name}")
                    else:
                        print(f"Frame index {frame_idx} out of range for {joint_name}")
                except Exception as e:
                    print(f"Error processing {joint_name}: {e}")
        else:
            print("No joint scores available")
    
    def analyze_movement(self): #this is for the feedback tab
        """Analyze the user's movement and generate feedback"""
        if not self.pro or not self.user:
            return
        
        # Show status
        self.status_label.setText("Status: Analyzing movement...")
        
        try:
            # Get warping path between user and pro
            path = vp.dtaidistance.dtw.warping_path(
                self.pro.angles['rElbowAngles'],
                self.user.angles['rElbowAngles']
            )
            
            # Get segment size and problem joints count
            segment_size_percent = self.segment_size_spinner.value()
            num_problem_joints = self.problem_joints_spinner.value()
            
            # Generate feedback
            self.feedback_data = cs.generate_feedback(
                self.pro, 
                self.user,
                path,
                segment_size_percent=segment_size_percent,
                num_problem_joints=num_problem_joints
            )
            
            # Update feedback widget
            self.feedback_widget.update_feedback(self.feedback_data)
            
            # Update timeline with highlighted problem areas
            if hasattr(self, 'temporal_scores'):
                self.feedback_timeline.update_data(self.temporal_scores, self.feedback_data)
            
            # Switch to feedback tab
            self.tab_widget.setCurrentWidget(self.feedback_tab)
            
            self.status_label.setText("Status: Movement analysis complete")
        except Exception as e:
            print(f"Error analyzing movement: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Status: Error during analysis - {str(e)}")
    
    def replay_segment(self, start_frame, end_frame):
        """Replay a specific segment of the reference video"""
        print(f"Replay segment called with frames {start_frame} to {end_frame}")
        
        if not self.pro or not self.video_path:
            print("No pro data or video path available")
            return
        
        # Store replay information
        self.replay_start_frame = start_frame
        self.replay_end_frame = end_frame
        self.replay_current_frame = start_frame
        
        # Set up video for replay
        self.cap_replay = cv2.VideoCapture(self.video_path)
        frame_position_result = self.cap_replay.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Set frame position result: {frame_position_result}")
        
        # Start replay timer
        self.segment_replay_active = True
        self.replay_timer = QTimer()
        self.replay_timer.timeout.connect(self.update_replay_frame)
        self.replay_timer.start(50)  # 20 fps
        
        self.status_label.setText(f"Status: Replaying problem segment (frames {start_frame}-{end_frame})")
        
    def update_replay_frame(self):
        """Update the replay frame during segment replay"""
        print(f"Update replay frame: {self.replay_current_frame}")
        
        if not self.segment_replay_active or not hasattr(self, 'cap_replay'):
            print("Replay not active or cap_replay not found")
            return
        
        ret, frame = self.cap_replay.read()
        print(f"Read frame result: {ret}")
        
        if not ret or self.replay_current_frame >= self.replay_end_frame:
            print("Reached end of segment, looping back to start")
            # Reached end of segment, loop back to start
            self.cap_replay.set(cv2.CAP_PROP_POS_FRAMES, self.replay_start_frame)
            self.replay_current_frame = self.replay_start_frame
            ret, frame = self.cap_replay.read()
        
        if ret:
            # Process frame with pose estimation
            with vp.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    vp.mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks, 
                        vp.mp_pose.POSE_CONNECTIONS,
                        vp.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        vp.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )
            
            # Add problem segment indicator
            cv2.putText(
                image,
                f"PROBLEM SEGMENT - Frame {self.replay_current_frame}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Convert to Qt format and display in the feedback video display
            pixmap = self.convert_cv_qt(image)
            self.feedback_video_display.setPixmap(
                pixmap.scaled(
                    self.feedback_video_display.size(),
                    Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            print("Updated feedback video display")
            
            self.replay_current_frame += 1
    
    def stop_replay(self):
        """Stop the segment replay"""
        print("Stopping replay")
        if hasattr(self, 'replay_timer') and self.replay_timer:
            self.replay_timer.stop()
        
        self.segment_replay_active = False
        
        if hasattr(self, 'cap_replay') and self.cap_replay:
            self.cap_replay.release()
        
        self.status_label.setText("Status: Replay stopped")

def main():
    app = QApplication(sys.argv)
    window = SportsMotionGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()