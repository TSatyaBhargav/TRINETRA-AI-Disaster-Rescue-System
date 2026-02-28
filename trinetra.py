"""
===============================================================================
🚁 TRINETRA - MULTI-MODE RESCUE SYSTEM WITH THERMAL CAM
===============================================================================
✅ Added THERMAL CAM mode (Mode 9)
✅ Simulates infrared/thermal imaging
✅ Multiple thermal palettes (Hot, Cold, Rainbow, Ironbow)
✅ Heat glow around detected objects
✅ Temperature scale overlay
===============================================================================
"""

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time
import threading
from flask import Flask, Response, render_template_string, jsonify
import os
from pathlib import Path
from datetime import datetime
import torch

# ===============================
# CONFIGURATION
# ===============================
CONFIG = {
    "screen_region": {"top": 100, "left": 400, "width": 800, "height": 600},
    "web_host": "127.0.0.1",
    "web_port": 5000,
    "confidence": 0.35,
    "auto_capture": True,
    "capture_cooldown": 2,
    "thermal_palette": "hot",  # hot, cold, rainbow, ironbow
}

# ===============================
# SIZE CATEGORIES FOR DISTANCE ESTIMATION - FIXED RANGES
# ===============================
SIZE_CATEGORIES = {
    "tiny": {"range": (5, 20), "color": (0, 0, 255), "distance": "200m", "label": "🔴 TINY"},
    "small": {"range": (20, 40), "color": (0, 165, 255), "distance": "150m", "label": "🟠 SMALL"},
    "medium": {"range": (40, 80), "color": (0, 255, 255), "distance": "100m", "label": "🟡 MEDIUM"},
    "large": {"range": (80, 150), "color": (0, 255, 0), "distance": "50m", "label": "🟢 LARGE"},
    "xlarge": {"range": (150, 2000), "color": (255, 0, 255), "distance": "<20m", "label": "🟣 CLOSE"}
}

# ===============================
# THERMAL CAMERA SIMULATOR
# ===============================
class ThermalSimulator:
    def __init__(self):
        self.palettes = {
            "hot": self.apply_hot_palette,
            "cold": self.apply_cold_palette,
            "rainbow": self.apply_rainbow_palette,
            "ironbow": self.apply_ironbow_palette
        }
        self.current_palette = "hot"
    
    def apply_hot_palette(self, frame):
        """Apply HOT thermal palette (black → red → yellow → white)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    
    def apply_cold_palette(self, frame):
        """Apply COLD palette (blue tones)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_COOL)
    
    def apply_rainbow_palette(self, frame):
        """Apply RAINBOW palette for temperature variation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    def apply_ironbow_palette(self, frame):
        """Apply IRONBOW palette (metallic look)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_RAINBOW)
    
    def add_temperature_scale(self, frame):
        """Add temperature scale legend"""
        h, w = frame.shape[:2]
        scale_height = 20
        scale_y = h - scale_height - 10
        
        # Draw gradient scale
        for i in range(w - 200, w - 50):
            # Calculate color based on position (simulate temperature gradient)
            val = int(255 * (i - (w - 200)) / 150)
            if self.current_palette == "hot":
                color = (0, 0, val) if val < 128 else (0, val-128, 255-val)
            elif self.current_palette == "cold":
                color = (val, 255-val, 255)
            elif self.current_palette == "rainbow":
                color = (val, 255-val, 255-val)
            else:
                color = (val, val//2, 255-val)
            
            cv2.line(frame, (i, scale_y), (i, scale_y + scale_height), color, 1)
        
        # Add labels
        cv2.putText(frame, "COLD", (w - 200, scale_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "HOT", (w - 50, scale_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def add_heat_glow(self, frame, detections):
        """Add heat glow around detected objects"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            center = ((x1 + x2)//2, (y1 + y2)//2)
            radius = max(x2 - x1, y2 - y1) // 2
            
            # Draw heat glow rings
            for r in range(radius, radius + 25, 5):
                alpha = 0.4 if r == radius else 0.2
                color = (0, 0, 255)  # Red heat glow
                cv2.circle(frame, center, r, color, 2)
        
        return frame
    
    def apply_thermal_effect(self, frame, detections=None):
        """Apply complete thermal camera simulation"""
        # Apply selected palette
        thermal_frame = self.palettes[self.current_palette](frame)
        
        # Add heat glow around detected objects
        if detections:
            thermal_frame = self.add_heat_glow(thermal_frame, detections)
        
        # Add temperature scale
        thermal_frame = self.add_temperature_scale(thermal_frame)
        
        return thermal_frame
    
    def toggle_palette(self):
        """Cycle through available palettes"""
        palettes = list(self.palettes.keys())
        current_idx = palettes.index(self.current_palette)
        next_idx = (current_idx + 1) % len(palettes)
        self.current_palette = palettes[next_idx]
        return self.current_palette

# ===============================
# MODE CONFIGURATIONS - ADDED THERMAL MODE
# ===============================
MODES = {
    "1": {
        "name": "🏠 BASIC",
        "description": "Human detection only",
        "model": "models/01_basic_human.pt",
        "classes": [0],
        "class_names": {0: "Human"},
        "color": (0, 255, 0),
        "hex_color": "#00ff00",
        "thermal": False
    },
    "2": {
        "name": "🚨 DISASTER",
        "description": "Human detection in disaster zones",
        "model": "models/02_disaster_real.pt",
        "classes": [0, 1, 2],
        "class_names": {0: "🔥 Fire", 1: "💨 Smoke", 2: "👤 Human"},
        "color": (0, 0, 255),
        "hex_color": "#ff0000",
        "thermal": False
    },
    "3": {
        "name": "🌲 FOREST",
        "description": "Wildlife detection",
        "model": "models/03_forest_animals.pt",
        "classes": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "class_names": {
            14: "Bird", 15: "Cat", 16: "Dog", 17: "Horse", 18: "Sheep",
            19: "Cow", 20: "Elephant", 21: "Bear", 22: "Zebra", 23: "Giraffe"
        },
        "color": (0, 255, 255),
        "hex_color": "#00ffff",
        "thermal": False
    },
    "4": {
        "name": "🌊 MARINE",
        "description": "Marine vessel detection",
        "model": "models/04_marine.pt",
        "classes": [0, 8, 9],
        "class_names": {0: "Human", 8: "🚤 Boat", 9: "🚢 Ship"},
        "color": (0, 165, 255),
        "hex_color": "#00a5ff",
        "thermal": False
    },
    "5": {
        "name": "🚗 VEHICLE",
        "description": "Vehicle detection",
        "model": "models/05_vehicle.pt",
        "classes": [1, 2, 3, 5, 7],
        "class_names": {1: "🚲 Bicycle", 2: "🚗 Car", 3: "🏍️ Motorcycle", 5: "🚌 Bus", 7: "🚛 Truck"},
        "color": (255, 255, 0),
        "hex_color": "#ffff00",
        "thermal": False
    },
    "6": {
        "name": "⚔️ ARMY",
        "description": "Military personnel detection",
        "model": "models/06_army.pt",
        "classes": [0],
        "class_names": {0: "⚔️ Personnel"},
        "color": (128, 0, 128),
        "hex_color": "#800080",
        "thermal": False
    },
    "7": {
        "name": "🚢 SHIP",
        "description": "Ship detection",
        "model": "models/07_ship.pt",
        "classes": [8, 9],
        "class_names": {8: "🚤 Boat", 9: "🚢 Ship"},
        "color": (255, 0, 255),
        "hex_color": "#ff00ff",
        "thermal": False
    },
    "8": {
        "name": "⛏️ MINING",
        "description": "Mining safety monitoring",
        "model": "models/08_mining.pt",
        "classes": [0],
        "class_names": {0: "⛏️ Miner"},
        "color": (255, 165, 0),
        "hex_color": "#ffa500",
        "thermal": False
    },
    "9": {  # NEW THERMAL MODE
        "name": "🔥 THERMAL",
        "description": "Heat signature detection (simulated)",
        "model": "models/02_disaster_real.pt",  # Reuse disaster model
        "classes": [0, 1, 2],
        "class_names": {0: "🔥 Heat Source", 1: "💨 Hot Smoke", 2: "👤 Warm Body"},
        "color": (255, 69, 0),
        "hex_color": "#ff4500",
        "thermal": True  # Flag to apply thermal effect
    }
}

# ===============================
# HTML TEMPLATE - FIXED STATS DISPLAY WITH THERMAL SUPPORT
# ===============================
DETECTION_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ mode_name }} - TRINETRA</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }
        body { background: #0a0a0a; color: white; padding: 15px; }
        .container { max-width: 100%; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; }
        .mode-badge { background: {{ mode_color }}; color: black; padding: 8px 20px; border-radius: 25px; font-weight: bold; }
        .stats-badge { background: #333; padding: 8px 20px; border-radius: 25px; }
        .video-container { width: 100%; background: #000; border-radius: 12px; overflow: hidden; 
                          margin: 15px 0; border: 3px solid {{ mode_color }}; }
        .video-container img { width: 100%; height: auto; display: block; }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 20px 0; }
        .stat-card { background: #1a1a1a; border-radius: 10px; padding: 15px; border-left: 4px solid; }
        .stat-card.tiny { border-left-color: #ff0000; }
        .stat-card.small { border-left-color: #ffa500; }
        .stat-card.medium { border-left-color: #ffff00; }
        .stat-card.large { border-left-color: #00ff00; }
        .stat-card.xlarge { border-left-color: #ff00ff; }
        .stat-number { font-size: 2rem; font-weight: bold; }
        .stat-label { color: #888; font-size: 0.8rem; }
        .controls { display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; 
               font-weight: bold; text-decoration: none; display: inline-block; }
        .btn-primary { background: {{ mode_color }}; color: black; }
        .btn-secondary { background: #333; color: white; }
        .recent-panel { background: #1a1a1a; border-radius: 10px; padding: 15px; margin-top: 20px; max-height: 300px; overflow-y: auto; }
        .detection-item { display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #333; }
        .serial { color: {{ mode_color }}; font-weight: bold; }
        .footer { margin-top: 20px; color: #666; font-size: 0.8rem; text-align: center; }
        .refresh-btn { background: #444; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer; }
        .thermal-badge { background: #ff4500; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.7rem; margin-left: 5px; }
        .palette-indicator { background: #333; padding: 2px 8px; border-radius: 3px; font-size: 0.7rem; margin-left: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="mode-badge">{{ mode_name }}{% if thermal %} <span class="thermal-badge">🔥 THERMAL</span>{% endif %}</span>
            <span class="stats-badge">⚡ {{ fps }} FPS | 🎯 {{ stats.total }} total</span>
        </div>
        
        <div class="video-container">
            <img src="/video_feed" alt="Live Feed" id="videoFeed">
        </div>
        
        <div class="controls">
            <a href="/" class="btn btn-secondary">🔙 Change Mode</a>
            <button class="btn btn-primary" onclick="captureNow()">📸 Capture Now</button>
            <button class="btn btn-secondary" onclick="refreshStats()">🔄 Refresh</button>
            {% if thermal %}
            <button class="btn btn-secondary" onclick="togglePalette()">🎨 Change Palette</button>
            {% endif %}
        </div>
        
        {% if thermal %}
        <div class="palette-indicator" style="text-align: center; margin: 10px 0;">
            Current Palette: <span id="palette">{{ palette }}</span>
        </div>
        {% endif %}
        
        <div class="dashboard">
            <div class="stat-card tiny">
                <div class="stat-label">🔴 TINY (200m)</div>
                <div class="stat-number" id="tinyCount">{{ stats.tiny }}</div>
            </div>
            <div class="stat-card small">
                <div class="stat-label">🟠 SMALL (150m)</div>
                <div class="stat-number" id="smallCount">{{ stats.small }}</div>
            </div>
            <div class="stat-card medium">
                <div class="stat-label">🟡 MEDIUM (100m)</div>
                <div class="stat-number" id="mediumCount">{{ stats.medium }}</div>
            </div>
            <div class="stat-card large">
                <div class="stat-label">🟢 LARGE (50m)</div>
                <div class="stat-number" id="largeCount">{{ stats.large }}</div>
            </div>
            <div class="stat-card xlarge">
                <div class="stat-label">🟣 CLOSE (<20m)</div>
                <div class="stat-number" id="xlargeCount">{{ stats.xlarge }}</div>
            </div>
        </div>
        
        <div class="recent-panel">
            <h3>📋 Recent Detections</h3>
            <div id="recent-list">
                {% for det in recent_detections %}
                <div class="detection-item">
                    <span><span class="serial">#{{ det.serial }}</span> {{ det.label }}</span>
                    <span>{{ det.time }}</span>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="footer">
            Serial Counter: <strong id="serial">{{ serial }}</strong> | Auto-capture: {{ "ON" if auto_capture else "OFF" }}
        </div>
    </div>
    
    <script>
        function captureNow() {
            fetch('/capture')
                .then(response => response.json())
                .then(data => {
                    alert('📸 Captured: ' + data.filename);
                    refreshStats();
                });
        }
        
        function refreshStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('tinyCount').innerText = data.tiny || 0;
                    document.getElementById('smallCount').innerText = data.small || 0;
                    document.getElementById('mediumCount').innerText = data.medium || 0;
                    document.getElementById('largeCount').innerText = data.large || 0;
                    document.getElementById('xlargeCount').innerText = data.xlarge || 0;
                    document.getElementById('serial').innerText = 'S' + String(data.serial).padStart(4, '0');
                });
        }
        
        {% if thermal %}
        function togglePalette() {
            fetch('/toggle_palette', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    document.getElementById('palette').innerText = data.palette;
                    alert('Palette changed to: ' + data.palette);
                });
        }
        {% endif %}
        
        // Auto-refresh every 2 seconds
        setInterval(refreshStats, 2000);
        
        // Also refresh when page loads
        window.onload = refreshStats;
    </script>
</body>
</html>
"""

MODE_SELECTION_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🚁 TRINETRA - Rescue System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }
        body { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: white; text-align: center; font-size: 2.5rem; margin-bottom: 10px; }
        .subtitle { color: #00ff88; text-align: center; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .mode-card { background: white; border-radius: 15px; padding: 20px; cursor: pointer; 
                    transition: transform 0.3s, box-shadow 0.3s; box-shadow: 0 10px 20px rgba(0,0,0,0.2); }
        .mode-card:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(0,255,136,0.3); }
        .mode-name { font-size: 1.3rem; font-weight: bold; margin: 10px 0; }
        .mode-desc { color: #666; font-size: 0.9rem; }
        .model-status { font-size: 0.8rem; color: #00ff88; margin-top: 10px; }
        .local-info { background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; 
                     margin-top: 30px; text-align: center; color: white; }
        .new-badge { background: #ff4500; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; margin-left: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚁 TRINETRA</h1>
        <div class="subtitle">AI-Powered Disaster Rescue System</div>
        
        <div class="grid">
            {% for num, mode in modes.items() %}
            <div class="mode-card" onclick="selectMode('{{ num }}')" style="border-left: 5px solid {{ mode.hex_color }};">
                <div class="mode-name">{{ mode.name }}{% if num == '9' %} <span class="new-badge">NEW</span>{% endif %}</div>
                <div class="mode-desc">{{ mode.description }}</div>
                <div class="model-status">{{ "✅ Model Ready" if mode.model_exists else "❌ Not Trained" }}</div>
            </div>
            {% endfor %}
        </div>
        
        <div class="local-info">
            <p>📱 Open on this computer: <strong>http://localhost:5000</strong></p>
            <p>🔥 New: Thermal Camera Mode (Mode 9) - Simulated heat signature detection</p>
        </div>
    </div>
    
    <script>
        function selectMode(mode) {
            window.location.href = '/start/' + mode;
        }
    </script>
</body>
</html>
"""

# ===============================
# CAPTURE MANAGER
# ===============================
class CaptureManager:
    def __init__(self):
        self.serial_counter = 1
        self.recent_detections = []
        os.makedirs("captured_images", exist_ok=True)
    
    def capture(self, frame, detections, mode_name, stats):
        """Capture image with serial number"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        serial = f"S{self.serial_counter:04d}"
        filename = f"captured_images/{serial}_{timestamp}.jpg"
        
        # Annotate frame
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Add header
        cv2.rectangle(annotated, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.putText(annotated, f"TRINETRA - {mode_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, f"Serial: {serial}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Stats: T{stats['tiny']} S{stats['small']} M{stats['medium']} L{stats['large']} C{stats['xlarge']}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw detections
        for det in detections[:10]:
            x1, y1, x2, y2 = det["bbox"]
            color = det.get("color", (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            label = f"{det['label']}"
            cv2.putText(annotated, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imwrite(filename, annotated)
        
        # Add to recent detections
        self.recent_detections.insert(0, {
            'serial': serial,
            'label': f"{len(detections)} detections",
            'time': datetime.now().strftime("%H:%M:%S")
        })
        if len(self.recent_detections) > 10:
            self.recent_detections.pop()
        
        self.serial_counter += 1
        return filename, serial

# ===============================
# DETECTOR CLASS (UPDATED WITH THERMAL SUPPORT)
# ===============================
class ModeDetector:
    def __init__(self, mode_num):
        self.mode_num = mode_num
        self.mode = MODES[mode_num]
        self.recent_detections = []
        self.stats = {'total': 0, 'tiny': 0, 'small': 0, 'medium': 0, 'large': 0, 'xlarge': 0}
        
        # Initialize thermal simulator if in thermal mode
        self.thermal_sim = ThermalSimulator() if self.mode.get('thermal', False) else None
        
        # Load model
        model_path = Path(self.mode["model"])
        if model_path.exists():
            self.model = YOLO(str(model_path))
            size = model_path.stat().st_size / (1024*1024)
            print(f"✅ Loaded {self.mode['name']} model ({size:.1f} MB)")
            if self.thermal_sim:
                print(f"   🔥 Thermal mode active with {self.thermal_sim.current_palette} palette")
        else:
            print(f"❌ Model not found: {model_path}")
            self.model = None
    
    def get_size_category(self, size):
        """Categorize by pixel size for distance estimation"""
        for cat, info in SIZE_CATEGORIES.items():
            if info["range"][0] <= size <= info["range"][1]:
                return cat, info
        return "large", SIZE_CATEGORIES["large"]
    
    def detect(self, frame, apply_thermal=True):
        """Run detection with optional thermal effect"""
        if not self.model:
            return []
        
        results = self.model(frame, conf=CONFIG["confidence"], verbose=False)
        
        detections = []
        
        # Reset stats
        for key in self.stats:
            self.stats[key] = 0
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Mode-specific filtering
                if cls_id in self.mode["classes"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    size = max(x2 - x1, y2 - y1)
                    
                    # Get category and update stats
                    cat, info = self.get_size_category(size)
                    self.stats[cat] += 1
                    
                    # Get class name
                    class_name = self.mode["class_names"].get(cls_id, f"Class {cls_id}")
                    
                    # Special formatting for disaster mode
                    if self.mode_num == "2":
                        if cls_id == 0:
                            label = f"🔥 Fire"
                        elif cls_id == 1:
                            label = f"💨 Smoke"
                        else:
                            label = f"👤 Human"
                    else:
                        label = f"{class_name}"
                    
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "conf": conf,
                        "size": size,
                        "category": cat,
                        "color": info["color"],
                        "class_id": cls_id
                    })
        
        self.stats['total'] = len(detections)
        return detections

# ===============================
# FLASK SERVER (UPDATED WITH THERMAL SUPPORT)
# ===============================
class RescueServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.detector = None
        self.frame = None
        self.fps = 0
        self.capture_manager = CaptureManager()
        self.last_capture = 0
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            # Check which models exist
            for mode in MODES.values():
                mode["model_exists"] = Path(mode["model"]).exists()
            
            return render_template_string(
                MODE_SELECTION_HTML,
                modes=MODES
            )
        
        @self.app.route('/start/<mode>')
        def start_mode(mode):
            if mode in MODES:
                self.detector = ModeDetector(mode)
                return render_template_string(
                    DETECTION_HTML,
                    mode_name=MODES[mode]['name'],
                    mode_color=MODES[mode]['hex_color'],
                    thermal=MODES[mode].get('thermal', False),
                    palette=self.detector.thermal_sim.current_palette if self.detector and self.detector.thermal_sim else "",
                    stats=self.detector.stats if self.detector else {'total':0, 'tiny':0, 'small':0, 'medium':0, 'large':0, 'xlarge':0},
                    recent_detections=self.capture_manager.recent_detections,
                    fps="0.0",
                    serial=f"S{self.capture_manager.serial_counter:04d}",
                    auto_capture=CONFIG["auto_capture"]
                )
            return "Invalid mode", 404
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/stats')
        def get_stats():
            if self.detector:
                stats = self.detector.stats.copy()
                stats['serial'] = self.capture_manager.serial_counter
                return jsonify(stats)
            return jsonify({'total':0, 'tiny':0, 'small':0, 'medium':0, 'large':0, 'xlarge':0, 'serial':1})
        
        @self.app.route('/capture')
        def capture():
            if self.frame is not None and self.detector:
                filename, serial = self.capture_manager.capture(
                    self.frame, 
                    self.detector.recent_detections[-10:] if self.detector.recent_detections else [],
                    self.detector.mode['name'],
                    self.detector.stats
                )
                return jsonify({"success": True, "filename": filename, "serial": serial})
            return jsonify({"success": False})
        
        @self.app.route('/toggle_palette', methods=['POST'])
        def toggle_palette():
            if self.detector and self.detector.thermal_sim:
                new_palette = self.detector.thermal_sim.toggle_palette()
                return jsonify({"success": True, "palette": new_palette})
            return jsonify({"success": False})
    
    def generate_frames(self):
        while True:
            if self.frame is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except:
                    pass
            time.sleep(0.03)
    
    def update_frame(self, frame, fps):
        self.frame = frame
        self.fps = fps
        
        # Auto-capture if enabled and humans detected
        if (CONFIG["auto_capture"] and self.detector and 
            self.detector.stats['total'] > 0 and 
            time.time() - self.last_capture > CONFIG["capture_cooldown"]):
            
            filename, serial = self.capture_manager.capture(
                frame,
                self.detector.recent_detections[-10:] if self.detector.recent_detections else [],
                self.detector.mode['name'],
                self.detector.stats
            )
            self.last_capture = time.time()
            print(f"📸 Auto-captured {serial} with {self.detector.stats['total']} detections")
    
    def run(self):
        threading.Thread(target=self.app.run, 
                        kwargs={'host': CONFIG["web_host"], 
                               'port': CONFIG["web_port"], 
                               'debug': False, 
                               'threaded': True},
                        daemon=True).start()

# ===============================
# MAIN
# ===============================
def main():
    print("=" * 80)
    print("🚁 TRINETRA - MULTI-MODE RESCUE SYSTEM WITH THERMAL CAM")
    print("=" * 80)
    
    # Check models
    print("\n📦 Checking trained models:")
    for mode_num, mode in MODES.items():
        model_path = Path(mode["model"])
        if model_path.exists():
            size = model_path.stat().st_size / (1024*1024)
            print(f"   ✅ {mode['name']}: {mode['model']} ({size:.1f} MB)")
        else:
            print(f"   ❌ {mode['name']}: NOT FOUND - {mode['model']}")
    
    print("\n🔥 THERMAL MODE FEATURES:")
    print("   • Hot palette: Black→Red→Yellow→White")
    print("   • Cold palette: Blue tones")
    print("   • Rainbow palette: Full spectrum")
    print("   • Ironbow palette: Metallic look")
    print("   • Heat glow around detected objects")
    print("   • Temperature scale overlay")
    
    # Start server
    server = RescueServer()
    server.run()
    
    # Screen capture
    sct = mss()
    screen = CONFIG["screen_region"]
    
    print("\n" + "=" * 80)
    print("📱 OPEN IN BROWSER: http://localhost:5000")
    print("=" * 80)
    print("\n✅ Server running. Press Ctrl+C to stop\n")
    
    try:
        while True:
            if server.detector and server.detector.model:
                frame_start = time.time()
                
                # Capture screen
                img = sct.grab(screen)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Detect
                detections = server.detector.detect(frame)
                server.detector.recent_detections = detections
                
                # Apply thermal effect if in thermal mode
                display = frame.copy()
                if server.detector.thermal_sim:
                    display = server.detector.thermal_sim.apply_thermal_effect(frame, detections)
                
                # Draw detection boxes on display (thermal mode already has them)
                if not server.detector.thermal_sim:
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        color = det["color"]
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display, det['label'], (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Add overlay with stats
                h, w = display.shape[:2]
                overlay = display.copy()
                cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
                
                # Add stats
                y_offset = 35
                mode_display = f"{server.detector.mode['name']} {'[THERMAL]' if server.detector.thermal_sim else ''}"
                cv2.putText(display, f"TRINETRA - {mode_display}", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Total: {server.detector.stats['total']}", 
                           (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display, f"T:{server.detector.stats['tiny']} S:{server.detector.stats['small']} M:{server.detector.stats['medium']} L:{server.detector.stats['large']} C:{server.detector.stats['xlarge']}", 
                           (20, y_offset+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if server.detector.thermal_sim:
                    cv2.putText(display, f"Palette: {server.detector.thermal_sim.current_palette}", 
                               (20, y_offset+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                
                # FPS
                fps = 1.0 / (time.time() - frame_start)
                server.update_frame(display, fps)
                
                # Show status
                if len(detections) > 0:
                    thermal_tag = " [THERMAL]" if server.detector.thermal_sim else ""
                    print(f"\r⚡ {server.detector.mode['name']}{thermal_tag} | "
                          f"Detected: {len(detections)} | "
                          f"T:{server.detector.stats['tiny']} S:{server.detector.stats['small']} "
                          f"M:{server.detector.stats['medium']} L:{server.detector.stats['large']} "
                          f"C:{server.detector.stats['xlarge']}", end='')
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()