"""
Real-time Deep Learning Acupressure Point Detection Application
FIXED VERSION - Points constrained within hand boundaries
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'acupressure_model.keras'
WINDOW_TITLE = "🧠 Deep Learning Acupressure Point Detector"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS = 30

# Acupressure point metadata
POINT_INFO = {
    'point1': {
        'name': 'LI-4 (Hegu)',
        'condition': 'Cold Relief',
        'description': 'Relieves headaches, cold symptoms',
        'color': (0, 120, 255)  # Orange (BGR)
    },
    'point2': {
        'name': 'PC-8 (Laogong)',
        'condition': 'Stress Relief',
        'description': 'Calms mind, reduces stress',
        'color': (255, 100, 100)  # Light blue (BGR)
    },
    'point3': {
        'name': 'HT-7 (Shen Men)',
        'condition': 'Anxiety Relief',
        'description': 'Eases anxiety, quiets mind',
        'color': (100, 255, 150)  # Green (BGR)
    }
}


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class DeepLearningAcupressureApp:
    def __init__(self, window):
        self.window = window
        self.window.title(WINDOW_TITLE)
        self.window.geometry("1400x850")
        self.window.configure(bg='#0a0a0a')
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model Not Found",
                f"Error: '{MODEL_PATH}' not found!\n\n"
                "Please run 'train_model.py' first to train the model."
            )
            self.window.destroy()
            sys.exit(1)
        
        # Load trained model
        print("Loading trained model...")
        try:
            self.model = keras.models.load_model(MODEL_PATH)
            print(f"✓ Model loaded from '{MODEL_PATH}'")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model:\n{str(e)}")
            self.window.destroy()
            sys.exit(1)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Failed to open webcam!")
            self.window.destroy()
            sys.exit(1)
        
        # UI State
        self.selected_mode = tk.StringVar(value="all")
        self.fps_counter = 0
        self.fps_value = 0
        self.frame_count = 0
        
        # IMPORTANT: Keep reference to PhotoImage to prevent garbage collection
        self.current_image = None
        
        # Build UI
        self.create_ui()
        
        # Start video loop
        self.is_running = True
        self.window.after(100, self.update_frame)  # Delayed start
        
        print("✓ Application started successfully!")
    
    # ========================================================================
    # UI CREATION
    # ========================================================================
    
    def create_ui(self):
        """Create the user interface."""
        
        # Top Control Panel
        control_frame = tk.Frame(self.window, bg='#1a1a2e', height=100)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        control_frame.pack_propagate(False)
        
        # Title
        title = tk.Label(
            control_frame,
            text="🧠 Deep Learning Acupressure Point Detector",
            bg='#1a1a2e',
            fg='#00ff88',
            font=('Arial', 18, 'bold')
        )
        title.pack(side=tk.TOP, pady=10)
        
        # Mode selection
        mode_frame = tk.Frame(control_frame, bg='#1a1a2e')
        mode_frame.pack(side=tk.TOP, pady=5)
        
        tk.Label(
            mode_frame,
            text="Detection Mode:",
            bg='#1a1a2e',
            fg='white',
            font=('Arial', 11, 'bold')
        ).pack(side=tk.LEFT, padx=10)
        
        modes = [
            ("Show All Points", "all", "#00ff88"),
            ("Cold Relief (LI-4)", "cold", "#ff6b6b"),
            ("Stress Relief (PC-8)", "stress", "#4ecdc4"),
            ("Anxiety Relief (HT-7)", "anxiety", "#ffd93d")
        ]
        
        for text, value, color in modes:
            btn = tk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.selected_mode,
                value=value,
                bg='#1a1a2e',
                fg=color,
                selectcolor='#16213e',
                font=('Arial', 10, 'bold'),
                activebackground='#1a1a2e',
                activeforeground=color,
                indicatoron=0,
                width=18,
                bd=3,
                relief=tk.RAISED,
                cursor='hand2'
            )
            btn.pack(side=tk.LEFT, padx=5)
        
        # Video Display Frame
        self.video_frame = tk.Label(self.window, bg='black', bd=5, relief=tk.SUNKEN)
        self.video_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        
        # Bottom Info Panel
        info_frame = tk.Frame(self.window, bg='#16213e', height=80)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        info_frame.pack_propagate(False)
        
        # Status label
        self.status_label = tk.Label(
            info_frame,
            text="👋 Position your hand in front of the camera",
            bg='#16213e',
            fg='#00ff88',
            font=('Arial', 12, 'bold'),
            anchor='w'
        )
        self.status_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Instructions label
        self.instructions_label = tk.Label(
            info_frame,
            text="🎯 Neural network will predict acupressure points in real-time",
            bg='#16213e',
            fg='#cccccc',
            font=('Arial', 10),
            anchor='w'
        )
        self.instructions_label.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        # FPS label
        self.fps_label = tk.Label(
            info_frame,
            text="FPS: --",
            bg='#16213e',
            fg='#888888',
            font=('Courier', 9),
            anchor='e'
        )
        self.fps_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
    
    # ========================================================================
    # BOUNDARY CHECKING & CORRECTION
    # ========================================================================
    
    def get_hand_boundary(self, hand_landmarks, frame_width, frame_height):
        """
        Create a convex hull around the hand to define its boundary.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Convex hull contour as numpy array
        """
        # Extract landmark coordinates in pixel space
        points = []
        for landmark in hand_landmarks.landmark:
            px = int(landmark.x * frame_width)
            py = int(landmark.y * frame_height)
            points.append([px, py])
        
        points = np.array(points, dtype=np.int32)
        
        # Create convex hull
        hull = cv2.convexHull(points)
        
        return hull
    
    def is_point_inside_hand(self, point, hand_contour):
        """
        Check if a point is inside the hand boundary.
        
        Args:
            point: (x, y) tuple
            hand_contour: Convex hull contour
            
        Returns:
            True if inside, False otherwise
        """
        result = cv2.pointPolygonTest(hand_contour, point, False)
        return result >= 0
    
    def constrain_point_to_hand(self, point, hand_contour, hand_landmarks, frame_width, frame_height):
        """
        If a point is outside the hand, move it to the nearest valid position inside.
        
        Args:
            point: (x, y) predicted point
            hand_contour: Convex hull of hand
            hand_landmarks: MediaPipe landmarks
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Corrected (x, y) point inside hand boundary
        """
        point_int = (int(point[0]), int(point[1]))
        
        # Check if already inside
        if self.is_point_inside_hand(point_int, hand_contour):
            return point
        
        # Find nearest point on hand contour
        min_distance = float('inf')
        nearest_point = point
        
        # Check against contour points
        for contour_point in hand_contour:
            cp = tuple(contour_point[0])
            dist = np.sqrt((point[0] - cp[0])**2 + (point[1] - cp[1])**2)
            if dist < min_distance:
                min_distance = dist
                nearest_point = cp
        
        # Also check against hand landmarks (more accurate for interior)
        for landmark in hand_landmarks.landmark:
            lx = landmark.x * frame_width
            ly = landmark.y * frame_height
            dist = np.sqrt((point[0] - lx)**2 + (point[1] - ly)**2)
            if dist < min_distance:
                min_distance = dist
                nearest_point = (lx, ly)
        
        # Move point slightly inward from boundary
        center = hand_contour.mean(axis=0)[0]
        direction = np.array(center) - np.array(nearest_point)
        direction_norm = direction / (np.linalg.norm(direction) + 1e-6)
        
        # Move 10 pixels inward
        corrected_point = np.array(nearest_point) + direction_norm * 10
        
        return tuple(corrected_point.astype(int))
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    
    def preprocess_landmarks(self, hand_landmarks, frame_width, frame_height):
        """
        Convert MediaPipe landmarks to normalized model input.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            frame_width: Width of video frame
            frame_height: Height of video frame
        
        Returns:
            Normalized landmarks array of shape (1, 42) ready for model input
        """
        # Extract all 21 landmarks
        landmarks = []
        for landmark in hand_landmarks.landmark:
            # MediaPipe gives normalized coordinates [0, 1]
            landmarks.append([landmark.x, landmark.y])
        
        # Convert to numpy array and flatten
        landmarks_array = np.array(landmarks, dtype=np.float32).flatten()
        
        # Add batch dimension: (42,) -> (1, 42)
        return landmarks_array.reshape(1, -1)
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def draw_neon_point(self, frame, point, color, label, description, index):
        """
        Draw a glowing neon acupressure point with label.
        
        Args:
            frame: OpenCV frame
            point: (x, y) coordinate tuple
            color: BGR color tuple
            label: Point name
            description: Point description
            index: Point index (1, 2, or 3)
        """
        x, y = int(point[0]), int(point[1])
        
        # Ensure point is within frame boundaries
        h, w = frame.shape[:2]
        x = max(30, min(x, w - 30))
        y = max(30, min(y, h - 30))
        
        # Multi-layered glow effect
        # Outer glow (largest, most transparent)
        cv2.circle(frame, (x, y), 28, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 22, color, 3, cv2.LINE_AA)
        
        # Middle glow
        cv2.circle(frame, (x, y), 16, color, 4, cv2.LINE_AA)
        
        # Main circle (solid)
        cv2.circle(frame, (x, y), 12, color, -1, cv2.LINE_AA)
        
        # Inner white glow
        cv2.circle(frame, (x, y), 8, (255, 255, 255), -1, cv2.LINE_AA)
        
        # Center dot
        cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
        
        # Pulsing outer ring (optional animated effect)
        pulse_radius = 35 + int(5 * np.sin(self.fps_counter * 0.2))
        cv2.circle(frame, (x, y), pulse_radius, color, 1, cv2.LINE_AA)
        
        # Label positioning
        label_x = x + 40
        label_y = y - 20
        
        # Adjust if too close to edge
        if label_x + 250 > w:
            label_x = x - 270
        if label_x < 10:
            label_x = 10
        if label_y < 50:
            label_y = y + 50
        
        # Draw label background with glow
        label_text = f"{index}. {label}"
        desc_text = description
        
        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        (desc_w, desc_h), _ = cv2.getTextSize(desc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        max_width = max(label_w, desc_w)
        
        # Glowing background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (label_x - 10, label_y - label_h - 15),
                     (label_x + max_width + 15, label_y + desc_h + 15),
                     (0, 0, 0), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border with point color
        cv2.rectangle(frame,
                     (label_x - 10, label_y - label_h - 15),
                     (label_x + max_width + 15, label_y + desc_h + 15),
                     color, 2, cv2.LINE_AA)
        
        # Text
        cv2.putText(frame, label_text, (label_x, label_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, desc_text, (label_x, label_y + desc_h + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Connection line
        cv2.line(frame, (x, y), (label_x - 10, label_y - 10), color, 2, cv2.LINE_AA)
    
    def should_show_point(self, point_index):
        """Determine if a point should be shown based on selected mode."""
        mode = self.selected_mode.get()
        
        if mode == "all":
            return True
        elif mode == "cold" and point_index == 0:
            return True
        elif mode == "stress" and point_index == 1:
            return True
        elif mode == "anxiety" and point_index == 2:
            return True
        
        return False
    
    # ========================================================================
    # MAIN VIDEO LOOP
    # ========================================================================
    
    def update_frame(self):
        """Main video processing loop."""
        if not self.is_running:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.window.after(33, self.update_frame)
                return
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Default status
            status = "👋 Position your hand in front of the camera"
            instructions = "🎯 Neural network ready for prediction"
            points_detected = 0
            
            # Process detected hands
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks,
                                                     results.multi_handedness):
                    
                    # GET HAND BOUNDARY (CRITICAL for constraining points)
                    hand_contour = self.get_hand_boundary(hand_landmarks, w, h)
                    
                    # Draw semi-transparent hand boundary overlay
                    overlay = frame.copy()
                    cv2.drawContours(overlay, [hand_contour], 0, (0, 255, 100), 2, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    
                    # Draw hand skeleton
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    # Preprocess landmarks for model
                    input_data = self.preprocess_landmarks(hand_landmarks, w, h)
                    
                    # 🧠 NEURAL NETWORK PREDICTION
                    predictions = self.model.predict(input_data, verbose=0)[0]
                    
                    # Extract 3 predicted points: [x1, y1, x2, y2, x3, y3]
                    raw_predicted_points = [
                        (predictions[0] * w, predictions[1] * h),  # Point 1
                        (predictions[2] * w, predictions[3] * h),  # Point 2
                        (predictions[4] * w, predictions[5] * h),  # Point 3
                    ]
                    
                    # ⭐ CONSTRAIN POINTS TO HAND BOUNDARY ⭐
                    constrained_points = []
                    for raw_point in raw_predicted_points:
                        corrected_point = self.constrain_point_to_hand(
                            raw_point, hand_contour, hand_landmarks, w, h
                        )
                        constrained_points.append(corrected_point)
                    
                    # Draw predicted points based on selected mode
                    point_keys = ['point1', 'point2', 'point3']
                    for idx, (point, key) in enumerate(zip(constrained_points, point_keys)):
                        if self.should_show_point(idx):
                            info = POINT_INFO[key]
                            self.draw_neon_point(
                                frame, point,
                                info['color'],
                                info['name'],
                                info['description'],
                                idx + 1
                            )
                            points_detected += 1
                    
                    # Update status
                    hand_type = hand_info.classification[0].label.upper()
                    mode_name = self.selected_mode.get().upper()
                    status = f"✅ {hand_type} hand detected | Mode: {mode_name} | {points_detected} points shown"
                    instructions = f"🧠 Points Auto-Constrained to Hand | Press for 30-60s"
            
            # Draw professional header overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 30), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            cv2.putText(frame, "Deep Learning Acupressure Point Detector",
                       (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 136), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Powered by TensorFlow + MediaPipe | FPS: {self.fps_value}",
                       (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
            
            # Update UI labels
            self.status_label.config(text=status)
            self.instructions_label.config(text=instructions)
            
            # Update FPS counter
            self.fps_counter += 1
            if self.fps_counter % 30 == 0:
                self.fps_value = 30  # Approximate
            self.fps_label.config(text=f"FPS: {self.fps_value}")
            
            # Convert to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # CRITICAL FIX: Store PhotoImage reference to prevent garbage collection
            self.current_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update the label with the new image
            self.video_frame.configure(image=self.current_image)
            
            self.frame_count += 1
            
        except Exception as e:
            print(f"Error in update_frame: {e}")
            import traceback
            traceback.print_exc()
        
        # Schedule next frame
        if self.is_running:
            self.window.after(33, self.update_frame)
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    def cleanup(self):
        """Cleanup resources."""
        self.is_running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.hands is not None:
            self.hands.close()
        print("✓ Resources cleaned up")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main application entry point."""
    print("=" * 70)
    print("DEEP LEARNING ACUPRESSURE POINT DETECTOR")
    print("=" * 70)
    print()
    
    root = tk.Tk()
    app = DeepLearningAcupressureApp(root)
    
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            app.cleanup()
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected. Cleaning up...")
        app.cleanup()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        app.cleanup()


if __name__ == "__main__":
    main()
