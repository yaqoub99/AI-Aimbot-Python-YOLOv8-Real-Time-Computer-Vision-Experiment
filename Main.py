import os
import sys
import time
import logging
import cv2
import math
import random
import numpy as np
from mss import mss
from screeninfo import get_monitors
from ultralytics import YOLO
import torch
import torch.backends.cudnn as cudnn
from pynput import keyboard, mouse as pynput_mouse

# --- STRICT WINDOWS ONLY KERNEL IMPORTS ---
try:
    import win32api
    import win32con
    HAS_WIN32 = True
except ImportError:
    raise ImportError("Critical System Error: This software architecture requires a native Windows OS environment.")

import ctypes
HAS_WINDLL = hasattr(ctypes, "windll")

# --- BACKEND ENGINE LIBRARIES FOR HARDWARE EMULATION ---
try:
    import pydirectinput
    pydirectinput.FAILSAFE = False
except ImportError:
    pydirectinput = None

try:
    import mouse as mouse_lib
except ImportError:
    mouse_lib = None

# Universal Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniversalDetectorSystem")

from PySide6.QtCore import QThread, Signal, Slot, Qt, QMutex, QMutexLocker 
from PySide6.QtGui import QPixmap, QColor, QPainter, QPen, QFont, QDragEnterEvent, QDropEvent , QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea,
    QCheckBox, QLabel, QSlider, QHBoxLayout, QComboBox, QFrame, QPushButton 
)

# Premium Cyber-Studio CSS Styling Sheet with Ultra-Accessible Scrollbar
DARK_STYLE = """
    QWidget { 
        background-color: #0A0A0C; 
        color: #E2E8F0; 
        font-family: 'Segoe UI', -apple-system, sans-serif; 
        font-size: 13px;
    }
    QFrame { 
        background-color: #13131A; 
        border-radius: 10px; 
        padding: 16px; 
        margin-bottom: 4px; 
        border: 1px solid #222230; 
    }
    QLabel { 
        color: #94A3B8; 
        font-weight: 500;
    }
    
    /* Modern Slider Design */
    QSlider::groove:horizontal { 
        border: none; 
        height: 6px; 
        background: #1E1E2F; 
        border-radius: 3px; 
    }
    QSlider::sub-page:horizontal { 
        background: QLinearGradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #0066FF, stop: 1 #00A0FF); 
        border-radius: 3px; 
    }
    QSlider::handle:horizontal { 
        background: #FFFFFF; 
        border: 2px solid #00A0FF;
        width: 16px; 
        height: 16px; 
        margin: -5px 0; 
        border-radius: 8px; 
    }
    QSlider::handle:horizontal:hover {
        background: #00A0FF;
        border: 2px solid #FFFFFF;
    }
    
    /* Clean Cyber Checkboxes */
    QCheckBox { 
        font-weight: 600; 
        color: #E2E8F0;
        spacing: 8px; 
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 2px solid #334155;
        background: #13131A;
    }
    QCheckBox::indicator:checked {
        background: #00A0FF;
        border-color: #00A0FF;
    }
    
    /* Action Buttons */
    QPushButton { 
        background-color: #1E1E2F; 
        border: 1px solid #334155; 
        border-radius: 6px; 
        padding: 10px; 
        font-weight: bold; 
        color: #E2E8F0;
    }
    QPushButton:hover {
        background-color: #2D2D44;
        border-color: #475569;
    }
    QPushButton:checked { 
        background-color: #00A0FF; 
        color: #0A0A0C; 
        border: 1px solid #00A0FF; 
    }
    
    /* Professional Dropdowns */
    QComboBox { 
        background-color: #1E1E2F; 
        border: 1px solid #334155; 
        padding: 8px 12px; 
        border-radius: 6px; 
        color: #FFFFFF; 
        min-width: 180px; 
    }
    QComboBox::drop-down { 
        border: none; 
        width: 24px;
    }
    QComboBox QAbstractItemView {
        background-color: #13131A;
        border: 1px solid #334155;
        selection-background-color: #00A0FF;
        selection-color: #0A0A0C;
    }
    
    /* WIDE & HIGH-CONTRAST ACCESSIBLE SCROLLBAR */
    QScrollArea { 
        border: none; 
        background-color: #0A0A0C; 
    }
    QScrollBar:vertical { 
        border: none; 
        background: #111116; 
        width: 18px; 
        margin: 0px; 
    }
    QScrollBar::handle:vertical { 
        background: #3B4252; 
        min-height: 50px; 
        border-radius: 6px; 
        margin: 3px; 
    }
    QScrollBar::handle:vertical:hover { 
        background: #00A0FF; 
    }
    QScrollBar::handle:vertical:pressed { 
        background: #0066FF; 
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { 
        border: none; 
        background: none; 
        height: 0px; 
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }
"""

class InferenceWorker(QThread):
    frame_ready = Signal(list, int, bool, QColor)
    fps_signal = Signal(float)
    model_loaded_signal = Signal(str, str) 
    hardware_status_signal = Signal(str)

    def __init__(self, monitor_width, monitor_height):
        super().__init__()
        self.running = False
        self.show_boxes = True
        self.confidence_threshold = 0.25
        self.fov_radius = 150
        self.circle_color = QColor(0, 160, 255)
        self.screen_w, self.screen_h = monitor_width, monitor_height
        
        # Core Controls
        self.tracker_enabled = False
        self.tracking_key_pressed = False
        self.selected_key_type = "keyboard"
        self.selected_key = keyboard.Key.shift
        
        # High-Speed Reactive Tuning Scalars
        self.tracking_speed = 0.45      
        self.close_speed_floor = 0.18   
        self.vertical_offset = 0.50     
        self.human_arcs_enabled = True
        self.tremor_intensity = 0.12
        
        # Shared State Thread Protection Mutex
        self.model_mutex = QMutex()
        self.current_model_path = "yolov8n.pt"
        self.active_classes = [0] 
        self.pending_model_swap = False
        
        self.device = 'cpu'
        self.half_precision = False
        # HARDCODED EXCLUSIVELY TO PYWIN32
        self.active_backend = "pywin32"
        
        self.kb_listener = None
        self.m_listener = None
        
        self.update_monitor_zone()
        self.start_listeners()

    def update_fov(self, r): self.fov_radius = r; self.update_monitor_zone()
    def update_confidence(self, c): self.confidence_threshold = c / 100.0
    def update_color(self, color): self.circle_color = color

    def trigger_model_swap(self, model_path):
        with QMutexLocker(self.model_mutex):
            self.current_model_path = model_path
            filename = os.path.basename(model_path).lower()
            if filename.startswith("yolov8") and len(filename) <= 11:
                self.active_classes = [0]
            else:
                self.active_classes = None
            self.pending_model_swap = True

    def update_monitor_zone(self):
        d = self.fov_radius * 2
        self.monitor = {
            "top": (self.screen_h // 2) - self.fov_radius, 
            "left": (self.screen_w // 2) - self.fov_radius, 
            "width": d, "height": d
        }

    def start_listeners(self):
        def on_press(key):
            if self.selected_key_type == "keyboard":
                if key == self.selected_key or (hasattr(key, 'char') and key.char == self.selected_key):
                    self.tracking_key_pressed = True

        def on_release(key):
            if self.selected_key_type == "keyboard":
                if key == self.selected_key or (hasattr(key, 'char') and key.char == self.selected_key):
                    self.tracking_key_pressed = False

        def on_click(x, y, button, pressed):
            if self.selected_key_type == "mouse":
                if button == self.selected_key:
                    self.tracking_key_pressed = pressed

        self.kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.kb_listener.start()
        
        self.m_listener = pynput_mouse.Listener(on_click=on_click)
        self.m_listener.start()

    def update_tracking_key(self, key_text):
        self.tracking_key_pressed = False
        mappings = {
            "Shift": ("keyboard", keyboard.Key.shift), "Ctrl": ("keyboard", keyboard.Key.ctrl),
            "Alt": ("keyboard", keyboard.Key.alt), "Caps Lock": ("keyboard", keyboard.Key.caps_lock),
            "E": ("keyboard", "e"), "Q": ("keyboard", "q"), "F": ("keyboard", "f"),
            "V": ("keyboard", "v"), "X": ("keyboard", "x"),
            "Mouse Left": ("mouse", pynput_mouse.Button.left), "Mouse Right": ("mouse", pynput_mouse.Button.right),
            "Mouse X1": ("mouse", pynput_mouse.Button.x1), "Mouse X2": ("mouse", pynput_mouse.Button.x2)
        }
        key_type, key_val = mappings.get(key_text, ("keyboard", keyboard.Key.shift))
        self.selected_key_type = key_type
        self.selected_key = key_val

    def dispatch_mouse_movement(self, dx, dy):
        int_x, int_y = int(dx), int(dy)
        if int_x == 0 and int_y == 0:
            return

        if self.active_backend == "pywin32":
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int_x, int_y, 0, 0)
        elif self.active_backend == "pydirectinput" and pydirectinput is not None:
            pydirectinput.moveRel(int_x, int_y, relative=True, _pause=False)
        elif self.active_backend == "ctypes" and HAS_WINDLL:
            ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_MOVE, int_x, int_y, 0, 0)
        elif self.active_backend == "mouse" and mouse_lib is not None:
            mouse_lib.move(int_x, int_y, absolute=False, duration=0)

    @torch.inference_mode()
    def run(self):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            self.half_precision = True  
            cudnn.benchmark = True       
            hw_status_text = f"Hardware Core: NVIDIA CUDA Active ({torch.cuda.get_device_name(0)})"
        else:
            self.device = 'cpu'
            self.half_precision = False
            hw_status_text = "Hardware Core: CPU Mode Only (CUDA Engine Not Found)"
        
        self.hardware_status_signal.emit(hw_status_text)

        try:
            model = YOLO(self.current_model_path)
            model.to(self.device)
            self.model_loaded_signal.emit(os.path.basename(self.current_model_path), "Human Core (Class 0)")
        except Exception:
            self.current_model_path = "yolov8n.pt"
            model = YOLO("yolov8n.pt")
            model.to(self.device)
            self.model_loaded_signal.emit("yolov8n.pt", "Human Core (Class 0)")

        self.running = True
        
        target_fps = 60.0
        frame_duration = 1.0 / target_fps
        
        with mss() as sct:
            while self.running:
                loop_start = time.perf_counter()
                
                with QMutexLocker(self.model_mutex):
                    if self.pending_model_swap:
                        try:
                            del model
                            if self.device.startswith('cuda'): torch.cuda.empty_cache()
                            model = YOLO(self.current_model_path)
                            model.to(self.device)
                            mode_desc = "Human Core (Class 0)" if self.active_classes == [0] else "All Classes Active"
                            self.model_loaded_signal.emit(os.path.basename(self.current_model_path), mode_desc)
                        except Exception:
                            model = YOLO("yolov8n.pt")
                            model.to(self.device)
                            self.active_classes = [0]
                            self.model_loaded_signal.emit("yolov8n.pt", "Human Core (Class 0)")
                        self.pending_model_swap = False

                try:
                    screenshot = sct.grab(self.monitor)
                    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
                    
                    results = model.predict(
                        frame, 
                        conf=self.confidence_threshold, 
                        verbose=False, 
                        device=self.device, 
                        classes=self.active_classes,
                        half=self.half_precision,
                        imgsz=320
                    )
                    
                    boxes_to_draw = []
                    closest_target = None
                    min_dist = float('inf')

                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            w, h = x2 - x1, y2 - y1
                            
                            x_center = x1 + (w / 2.0)
                            y_center = y1 + (h * self.vertical_offset)
                            
                            distance = ((x_center - self.fov_radius)**2 + (y_center - self.fov_radius)**2)**0.5
                            
                            if distance <= self.fov_radius:
                                boxes_to_draw.append((x1, y1, w, h, float(box.conf[0])))
                                if distance < min_dist:
                                    min_dist = distance
                                    closest_target = (x_center - self.fov_radius, y_center - self.fov_radius, distance)

                    if self.tracker_enabled and self.tracking_key_pressed and closest_target is not None:
                        dx, dy, dist = closest_target
                        
                        if dist >= 2.0:
                            if dist > 20.0:
                                scalar = np.clip(self.tracking_speed + (dist / 800.0), 0.25, 0.85)
                                move_x = dx * scalar
                                move_y = dy * scalar
                                
                                if self.human_arcs_enabled:
                                    arc = math.sin(time.time() * 4.0) * 0.02
                                    move_x += -dy * arc
                                    move_y += dx * arc
                            else:
                                move_x = dx * max((self.tracking_speed * 0.6) * (dist / 20.0), self.close_speed_floor)
                                move_y = dy * max((self.tracking_speed * 0.6) * (dist / 20.0), self.close_speed_floor)
                            
                            if self.tremor_intensity > 0:
                                move_x += random.uniform(-self.tremor_intensity, self.tremor_intensity)
                                move_y += random.uniform(-self.tremor_intensity, self.tremor_intensity)
                                
                            self.dispatch_mouse_movement(move_x, move_y)

                    self.frame_ready.emit(boxes_to_draw, self.fov_radius, self.show_boxes, self.circle_color)
                    
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = frame_duration - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                    total_elapsed = time.perf_counter() - loop_start
                    self.fps_signal.emit(1.0 / (total_elapsed + 0.0001))
                    
                except Exception as e:
                    logger.warning(f"Inference Loop Error: {e}")
                    time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.kb_listener: self.kb_listener.stop()
        if self.m_listener: self.m_listener.stop()
        self.wait()


class OverlayWindow(QMainWindow):
    def __init__(self, w, h):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.WindowTransparentForInput | 
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setGeometry(0, 0, w, h)
        self.display = QLabel(self)
        self.display.setGeometry(0, 0, w, h)
        self.w, self.h = w, h

    @Slot(list, int, bool, QColor)
    def update_overlay(self, boxes, radius, show_boxes, color):
        pixmap = QPixmap(self.w, self.h)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(QPen(color, 2))
        painter.drawEllipse(self.w//2 - radius, self.h//2 - radius, radius * 2, radius * 2)
        
        if show_boxes:
            painter.setPen(QPen(Qt.GlobalColor.green, 2))
            for (x, y, w, h, _) in boxes:
                painter.drawRect(self.w//2 - radius + x, self.h//2 - radius + y, w, h)
        painter.end()
        self.display.setPixmap(pixmap)


class ControlPanel(QWidget):
    def __init__(self, worker, overlay):
        super().__init__()
        self.worker = worker
        self.overlay = overlay
        self.setWindowTitle("Detector Studio Pro")
        self.setStyleSheet(DARK_STYLE)
        if os.path.exists("8911115.png"):
            self.setWindowIcon(QIcon("8911115.png"))
        
        self.setMinimumSize(685, 850)
        self.resize(685, 850)
        self.setAcceptDrops(True)
        
        window_layout = QVBoxLayout(self)
        window_layout.setContentsMargins(8, 8, 8, 8)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(16, 16, 32, 16) 
        main_layout.setSpacing(12)
        
        title = QLabel("DETECTOR STUDIO CONTROL ENGINE")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF; letter-spacing: 1px; margin-bottom: 8px;")
        main_layout.addWidget(title)
        
        # --- CORE ENGINE STATUS MONITOR ---
        status_frame = QFrame()
        status_layout = QVBoxLayout(status_frame)
        
        self.hw_lbl = QLabel("Hardware Core: Scanning...")
        self.hw_lbl.setStyleSheet("color: #E2E8F0; font-size: 12px; font-weight: bold;")
        status_layout.addWidget(self.hw_lbl)
        
        self.fps_label = QLabel("Engine State: Idle | Diagnostic Performance: 0.0 FPS")
        self.fps_label.setStyleSheet("color: #94A3B8; font-size: 11px;")
        status_layout.addWidget(self.fps_label)
        main_layout.addWidget(status_frame)

        # --- NETWORK MODEL PIPELINE CONTROLLER ---
        model_frame = QFrame()
        model_layout = QVBoxLayout(model_frame)
        m_title = QLabel("NEURAL NETWORK LAYER WEIGHTS")
        m_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        m_title.setStyleSheet("color: #00A0FF;")
        model_layout.addWidget(m_title)
        
        m_row = QHBoxLayout()
        lbl_w = QLabel("Weights File:")
        m_row.addWidget(lbl_w)
        self.model_box = QComboBox()
        m_row.addWidget(self.model_box)
        model_layout.addLayout(m_row)
        
        self.model_status_lbl = QLabel("Active Weight Vector: yolov8n.pt")
        self.model_status_lbl.setStyleSheet("color: #E2E8F0; font-size: 12px;")
        model_layout.addWidget(self.model_status_lbl)
        
        self.filter_mode_lbl = QLabel("Target Ruleset: Human Core (Class 0)")
        self.filter_mode_lbl.setStyleSheet("color: #00FFAA; font-size: 11px; font-weight: bold;")
        model_layout.addWidget(self.filter_mode_lbl)
        
        drop_instruction = QLabel("💡 Drag & Drop custom weights (.pt or .onnx) anywhere inside the studio workspace to load immediately.")
        drop_instruction.setStyleSheet("color: #64748B; font-size: 11px; font-style: italic; margin-top: 4px;")
        model_layout.addWidget(drop_instruction)
        main_layout.addWidget(model_frame)
        
        # --- TRACKING ENGINE OPERATIONAL CENTER ---
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        c_title = QLabel("TRACKING RUNTIME AUTOMATION")
        c_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        c_title.setStyleSheet("color: #00A0FF;")
        controls_layout.addWidget(c_title)
        
        self.track_btn = QPushButton("Enable Operational Tracker")
        self.track_btn.setCheckable(True)
        self.track_btn.toggled.connect(self.toggle_tracker)
        controls_layout.addWidget(self.track_btn)
        
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("Activation Binder:"))
        self.key_box = QComboBox()
        self.key_box.addItems([
            "Shift", "Ctrl", "Alt", "Caps Lock", "E", "Q", "F", "V", "X", 
            "Mouse Left", "Mouse Right", "Mouse X1", "Mouse X2"
        ])
        self.key_box.currentTextChanged.connect(self.worker.update_tracking_key)
        key_layout.addWidget(self.key_box)
        controls_layout.addLayout(key_layout)
        
        self.toggle = QCheckBox("Render Target Boxes")
        self.toggle.setChecked(True)
        self.toggle.stateChanged.connect(lambda s: setattr(self.worker, 'show_boxes', s == 2))
        controls_layout.addWidget(self.toggle)
        main_layout.addWidget(controls_frame)
        
        # --- ADVANCED DRIVER API BACKEND INTERFACE ---
        backend_frame = QFrame()
        backend_layout = QVBoxLayout(backend_frame)
        b_title = QLabel("SYSTEM KERNEL INPUT DRIVER")
        b_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        b_title.setStyleSheet("color: #00A0FF;")
        backend_layout.addWidget(b_title)
        
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Active I/O Driver API:"))
        self.backend_box = QComboBox()
        
        backends = ["pywin32"]
        if pydirectinput is not None: backends.append("pydirectinput")
        if HAS_WINDLL: backends.append("ctypes")
        if mouse_lib is not None: backends.append("mouse")
        
        self.backend_box.addItems(backends)
        self.backend_box.currentTextChanged.connect(self.change_backend)
        combo_row.addWidget(self.backend_box)
        backend_layout.addLayout(combo_row)
        
        self.backend_desc = QLabel("")
        self.backend_desc.setStyleSheet("color: #64748B; font-size: 11px;")
        backend_layout.addWidget(self.backend_desc)
        self.change_backend("pywin32")
        main_layout.addWidget(backend_frame)

        # --- ADVANCED KINEMATICS & VELOCITY MODULATION PANEL ---
        tuning_frame = QFrame()
        tuning_layout = QVBoxLayout(tuning_frame)
        
        tuning_title = QLabel("VELOCITY KINEMATICS & MODULATION SMOOTHING")
        tuning_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        tuning_title.setStyleSheet("color: #00A0FF;")
        tuning_layout.addWidget(tuning_title)
        
        s_row = QHBoxLayout()
        s_row.addWidget(QLabel("Max Tracking Speed (Velocity Gain):"))
        self.speed_lbl = QLabel("0.45")
        self.speed_lbl.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        s_row.addStretch()
        s_row.addWidget(self.speed_lbl)
        tuning_layout.addLayout(s_row)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 95)
        self.speed_slider.setValue(45)
        self.speed_slider.valueChanged.connect(self.change_speed)
        tuning_layout.addWidget(self.speed_slider)
        
        f_row = QHBoxLayout()
        f_row.addWidget(QLabel("Proximity Speed Floor (Clamping Velocity):"))
        self.floor_lbl = QLabel("0.18")
        self.floor_lbl.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        f_row.addStretch()
        f_row.addWidget(self.floor_lbl)
        tuning_layout.addLayout(f_row)
        self.floor_slider = QSlider(Qt.Orientation.Horizontal)
        self.floor_slider.setRange(5, 50)
        self.floor_slider.setValue(18)
        self.floor_slider.valueChanged.connect(self.change_floor)
        tuning_layout.addWidget(self.floor_slider)
        
        h_row = QHBoxLayout()
        h_row.addWidget(QLabel("Target Axis Alignment (Vertical Scaling Offset):"))
        self.height_lbl = QLabel("Center (0.50)")
        self.height_lbl.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        h_row.addStretch()
        h_row.addWidget(self.height_lbl)
        tuning_layout.addLayout(h_row)
        self.height_slider = QSlider(Qt.Orientation.Horizontal)
        self.height_slider.setRange(5, 95)
        self.height_slider.setValue(50)
        self.height_slider.valueChanged.connect(self.change_height)
        tuning_layout.addWidget(self.height_slider)
        
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Micro Tremor Mitigation Value:"))
        self.tremor_lbl = QLabel("0.12 px")
        self.tremor_lbl.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        n_row.addStretch()
        n_row.addWidget(self.tremor_lbl)
        tuning_layout.addLayout(n_row)
        self.tremor_slider = QSlider(Qt.Orientation.Horizontal)
        self.tremor_slider.setRange(0, 50)
        self.tremor_slider.setValue(12)
        self.tremor_slider.valueChanged.connect(self.change_tremor)
        tuning_layout.addWidget(self.tremor_slider)

        self.arc_toggle = QCheckBox("Simulate Ergonomic Wrist Deviations (Arc Profiles)")
        self.arc_toggle.setChecked(True)
        self.arc_toggle.stateChanged.connect(lambda s: setattr(self.worker, 'human_arcs_enabled', s == 2))
        tuning_layout.addWidget(self.arc_toggle)
        main_layout.addWidget(tuning_frame)
        
        # --- FIELD OF VISION DIMENSIONAL ARCHITECTURE ---
        bounds_frame = QFrame()
        bounds_layout = QVBoxLayout(bounds_frame)
        
        b_title_zone = QLabel("DIMENSIONAL RADAR BOUNDS")
        b_title_zone.setFont(QFont("Segoe UI", 9, QFont.Bold))
        b_title_zone.setStyleSheet("color: #00A0FF;")
        bounds_layout.addWidget(b_title_zone)
        
        fov_row = QHBoxLayout()
        fov_row.addWidget(QLabel("Active Canvas FOV Boundary Diameter:"))
        self.fov_val_lbl = QLabel("150 px")
        self.fov_val_lbl.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        fov_row.addStretch()
        fov_row.addWidget(self.fov_val_lbl)
        bounds_layout.addLayout(fov_row)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(40, 900)
        self.slider.setValue(150)
        self.slider.valueChanged.connect(lambda v: (self.worker.update_fov(v), self.fov_val_lbl.setText(f"{v} px")))
        bounds_layout.addWidget(self.slider)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Minimum Threshold Verification Bound:"))
        self.conf_label = QLabel("25%")
        self.conf_label.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        conf_row.addStretch()
        conf_row.addWidget(self.conf_label)
        bounds_layout.addLayout(conf_row)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(lambda v: (self.worker.update_confidence(v), self.conf_label.setText(f"{v}%")))
        bounds_layout.addWidget(self.conf_slider)
        
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Radar Canvas Matrix Color Scheme:"))
        self.color_box = QComboBox()
        self.color_box.addItems(["Cyan", "Red", "White", "Yellow"])
        self.color_box.currentTextChanged.connect(self.change_color)
        color_row.addWidget(self.color_box)
        bounds_layout.addLayout(color_row)
        main_layout.addWidget(bounds_frame)
        
        scroll.setWidget(scroll_content)
        window_layout.addWidget(scroll)
        
        self.model_cache = {}
        self.scan_workspace_models()
        self.model_box.currentTextChanged.connect(self.on_model_selection_changed)
        
        self.worker.frame_ready.connect(self.overlay.update_overlay)
        self.worker.fps_signal.connect(lambda f: self.fps_label.setText(f"Engine State: Processing Stream | Diagnostic Performance: {f:.1f} FPS"))
        self.worker.model_loaded_signal.connect(self.update_ui_model_labels)
        self.worker.hardware_status_signal.connect(self.update_hardware_label)
        
        self.worker.start()
        self.overlay.show()

    def scan_workspace_models(self):
        self.model_box.blockSignals(True)
        self.model_box.clear()
        local_files = [f for f in os.listdir('.') if f.endswith('.pt') or f.endswith('.onnx')]
        for file in local_files:
            self.model_cache[file] = file
            self.model_box.addItem(file)
        if "yolov8n.pt" not in self.model_cache:
            self.model_cache["yolov8n.pt"] = "yolov8n.pt"
            self.model_box.addItem("yolov8n.pt")
        self.model_box.blockSignals(False)

    def on_model_selection_changed(self, model_name):
        if model_name in self.model_cache:
            full_path = self.model_cache[model_name]
            self.worker.trigger_model_swap(full_path)

    def update_ui_model_labels(self, name, mode_text):
        self.model_status_lbl.setText(f"Active Weight Vector: <b>{name}</b>")
        self.filter_mode_lbl.setText(f"Target Ruleset: {mode_text}")
        if "All Classes" in mode_text:
            self.filter_mode_lbl.setStyleSheet("color: #FFB000; font-size: 11px; font-weight: bold;")
        else:
            self.filter_mode_lbl.setStyleSheet("color: #00FFAA; font-size: 11px; font-weight: bold;")

    def update_hardware_label(self, status_text):
        self.hw_lbl.setText(status_text)
        if "CPU" in status_text:
            self.hw_lbl.setStyleSheet("color: #FF5555; font-size: 12px; font-weight: bold;")
        else:
            self.hw_lbl.setStyleSheet("color: #00FFAA; font-size: 12px; font-weight: bold;")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.pt') or file_path.endswith('.onnx'):
                model_name = os.path.basename(file_path)
                self.model_cache[model_name] = file_path
                if self.model_box.findText(model_name) == -1:
                    self.model_box.addItem(model_name)
                self.model_box.setCurrentText(model_name)
                logger.info(f"Integrated model path via Drag & Drop interface: {file_path}")

    def toggle_tracker(self, checked):
        self.worker.tracker_enabled = checked
        if checked:
            self.track_btn.setText("Operational Tracker: ENGAGED")
        else:
            self.track_btn.setText("Enable Operational Tracker")
            self.worker.tracking_key_pressed = False

    def change_backend(self, text):
        self.worker.active_backend = text
        descriptions = {
            "pywin32": "Status: Active • Low-level User32 structural runtime layer. Highest performance.",
            "pydirectinput": "Status: Standby • Hardware driver virtualization mode for WebGL/3D environments.",
            "ctypes": "Status: Standby • Windows core kernel memory pointer mapping execution.",
            "mouse": "Status: Standby • Direct system hook targeting absolute coordinate streams."
        }
        self.backend_desc.setText(descriptions.get(text, "Windows input subsystem architecture active."))

    def change_speed(self, val):
        float_val = val / 100.0
        self.worker.tracking_speed = float_val
        self.speed_lbl.setText(f"{float_val:.2f}")

    def change_floor(self, val):
        float_val = val / 100.0
        self.worker.close_speed_floor = float_val
        self.floor_lbl.setText(f"{float_val:.2f}")

    def change_tremor(self, val):
        float_val = val / 100.0
        self.worker.tremor_intensity = float_val
        self.tremor_lbl.setText(f"{float_val:.2f} px")

    def change_height(self, val):
        float_val = val / 100.0
        self.worker.vertical_offset = float_val
        if val == 50:
            self.height_lbl.setText("Center (0.50)")
        elif val < 50:
            self.height_lbl.setText(f"Upper Frame Bound ({float_val:.2f})")
        else:
            self.height_lbl.setText(f"Lower Frame Bound ({float_val:.2f})")

    def change_color(self, text):
        colors = {
            "Cyan": QColor(0, 160, 255), "Red": QColor(255, 0, 0), 
            "White": QColor(255, 255, 255), "Yellow": QColor(255, 255, 0)
        }
        self.worker.update_color(colors.get(text))

    def closeEvent(self, event):
        self.worker.stop()
        self.overlay.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    primary_monitor = get_monitors()[0]
    W, H = primary_monitor.width, primary_monitor.height
    
    logger.info(f"System Canvas Resolution configured to: {W}x{H}")
    
    processor = InferenceWorker(W, H)
    hud_overlay = OverlayWindow(W, H)
    dashboard = ControlPanel(processor, hud_overlay)
    
    dashboard.show()
    sys.exit(app.exec())