from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class SkeletonWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.joint_colors = {}  # Will store colors for each joint
        self.setMinimumSize(400, 600)
        self.debug_mode = True  # Added debug mode
        
    def set_joint_color(self, joint_name, similarity_score):
        """Set color for a specific joint based on similarity score"""
        #print(f"DEBUG: Setting color for {joint_name} with score {similarity_score}")
        
        # Remove Angles or Vector suffix if present
        for suffix in ['Angles', 'Vector']:
            joint_name = joint_name.replace(suffix, '')
        
        if similarity_score >= 90:
            color = QColor(76, 175, 80)  # Green
        elif similarity_score >= 70:
            color = QColor(255, 193, 7)  # Yellow
        else:
            color = QColor(244, 67, 54)  # Red
        
        #print(f"DEBUG: Storing color for {joint_name}: {color.getRgb()}")
        self.joint_colors[joint_name] = color
        self.repaint()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        w = self.width()
        h = self.height()
        center_x = w // 2
        
        # Draw skeleton
        # Torso
        painter.setPen(QPen(QColor(102, 102, 102), 4))
        painter.drawLine(center_x, 100, center_x, 250)
        
        # Right Arm and Shoulder
        self.draw_limb(painter, "rShoulder", center_x, 150, center_x + 60, 180)
        self.draw_limb(painter, "rElbow", center_x + 60, 180, center_x + 90, 220)
        self.draw_joint(painter, "rWrist", center_x + 90, 220)
        
        # Left Arm and Shoulder
        self.draw_limb(painter, "lShoulder", center_x, 150, center_x - 60, 180)
        self.draw_limb(painter, "lElbow", center_x - 60, 180, center_x - 90, 220)
        self.draw_joint(painter, "lWrist", center_x - 90, 220)
        
        # Right Leg and Hip
        self.draw_limb(painter, "rHip", center_x, 250, center_x + 30, 350)
        self.draw_limb(painter, "rKnee", center_x + 30, 350, center_x + 40, 450)
        self.draw_joint(painter, "rAnkle", center_x + 40, 450)
        
        # Left Leg and Hip
        self.draw_limb(painter, "lHip", center_x, 250, center_x - 30, 350)
        self.draw_limb(painter, "lKnee", center_x - 30, 350, center_x - 40, 450)
        self.draw_joint(painter, "lAnkle", center_x - 40, 450)
        
        # Draw legend
        self.draw_legend(painter)
        
    def draw_limb(self, painter, joint_name, x1, y1, x2, y2):
        """Draw a limb segment and its joint"""
        # Draw the limb segment
        painter.setPen(QPen(QColor(33, 150, 243), 4))  # Blue line
        painter.drawLine(x1, y1, x2, y2)
        
        # Draw the joint
        self.draw_joint(painter, joint_name, x1, y1)
        
    def draw_joint(self, painter, joint_name, x, y):
        """Draw a joint circle with its color based on similarity"""
        # Remove any suffixes
        for suffix in ['Angles', 'Vector']:
            joint_name = joint_name.replace(suffix, '')
        
        color = self.joint_colors.get(joint_name, QColor(128, 128, 128))  # Default gray
        
        #print(f"DEBUG: Drawing joint {joint_name}")
        #print(f"DEBUG: Looking for color of {joint_name}")
        #print(f"DEBUG: Joint color: {color.getRgb()}")
        
        painter.setPen(QPen(color, 2))
        painter.setBrush(color)
        painter.drawEllipse(x - 6, y - 6, 12, 12)
        
        # Draw joint label
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(x + 10, y + 5, joint_name)
        
    def draw_legend(self, painter):
        """Draw the color legend"""
        y_start = 20
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(10, y_start, "Joint Similarity:")
        
        colors = [
            (QColor(76, 175, 80), "Good (>90%)"),
            (QColor(255, 193, 7), "Average (70-90%)"),
            (QColor(244, 67, 54), "Poor (<70%)")
        ]
        
        for i, (color, text) in enumerate(colors):
            y = y_start + 25 * (i + 1)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(10, y - 8, 12, 12)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(30, y, text)

class SimilarityGraph(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=4):
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        
    def plot_similarity(self, temporal_scores, current_frame=None):
        """Plot similarity scores over time"""
        # Debug printing
        print("5Plotting similarity graph")
        print("5Temporal scores type:", type(temporal_scores))
        print("5Temporal scores:", temporal_scores)
        
        try:
            self.axes.clear()
            
            # Defensive programming
            if not temporal_scores:
                print("No temporal scores to plot")
                self.draw()
                return
            
            frames = list(temporal_scores.keys())
            scores = list(temporal_scores.values())
            
            print("Frames:", frames)
            print("Scores:", scores)
            
            self.axes.plot(frames, scores, 'b-', label='Similarity')
            
            if current_frame is not None:
                self.axes.axvline(x=current_frame, color='r', linestyle='--')
                
            self.axes.set_xlabel('Frame')
            self.axes.set_ylabel('Similarity Score')
            self.axes.set_title('Movement Similarity Over Time')
            self.axes.grid(True)
            
            # Add some safety for axis scaling
            if scores:
                self.axes.set_ylim(0, max(scores) * 1.1)
            
            self.draw()
        except Exception as e:
            print(f"Error plotting similarity graph: {e}")