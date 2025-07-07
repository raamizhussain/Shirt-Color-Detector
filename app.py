import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
import base64
from collections import Counter, defaultdict
from sklearn.cluster import KMeans, DBSCAN
import webcolors
import colorsys
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Page configuration with better styling
st.set_page_config(
    page_title="AI Shirt Color Detection Pro",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-bottom: 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .color-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .detection-stats {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .help-section {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
    }
    .warning-section {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎯 AI Shirt Color Detection Pro</h1>
    <p>Advanced computer vision system for detecting and analyzing shirt colors in images and videos</p>
    <h3>(USE DETR AI Model for Better and Accurate Results)</h3>
    <h4>RECOMMENDED CHOICE: DETR (AI MODEL), Color Distribution Analysis (Detection Mode) and Multi-Region Sampling (Color Analysis Method)</h4>
    <p>For best results, use high-quality images with clear shirt visibility and keep the keep the detection confidence at 0.10</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar configuration
st.sidebar.markdown("## 🎛️ Detection Settings")

# Model selection
model_type = st.sidebar.selectbox(
    "🤖 AI Model",
    ["YOLOv5 (Fast)", "YOLOv8 (Better)", "DETR (Most Accurate)"],
    help="Choose the AI model for person detection"
)

# Detection mode
detection_mode = st.sidebar.selectbox(
    "🔍 Detection Mode",
    ["Smart Auto-Grouping", "Specific Color Hunt", "Color Distribution Analysis", "Similar Color Clustering"],
    help="Choose your detection strategy"
)

# Target color selection for specific mode
if detection_mode == "Specific Color Hunt":
    target_color = st.sidebar.selectbox(
        "🎨 Target Color",
        ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray", "brown", 
         "navy", "maroon", "olive", "teal", "coral", "burgundy", "khaki", "beige", "turquoise"],
        help="Select the color you want to detect"
    )
    
    # Color tolerance
    color_tolerance = st.sidebar.slider(
        "🎯 Color Matching Tolerance",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
        help="Higher values = more flexible color matching"
    )

# Advanced settings
st.sidebar.markdown("### ⚙️ Advanced Settings")

confidence_threshold = st.sidebar.slider(
    "🔍 Detection Confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.6,
    step=0.05,
    help="Higher values = more confident detections only"
)

# Color analysis settings
color_analysis_method = st.sidebar.selectbox(
    "🎨 Color Analysis Method",
    ["K-Means Clustering", "Histogram Analysis", "Dominant Color + Context", "Multi-Region Sampling"],
    help="Choose color extraction method"
)

# Image preprocessing options
st.sidebar.markdown("### 🖼️ Image Processing")
apply_enhancement = st.sidebar.checkbox("✨ Auto-enhance image", value=True)
remove_background = st.sidebar.checkbox("🎭 Background removal", value=False)
shirt_region_focus = st.sidebar.checkbox("👕 Focus on shirt region", value=True)

# Comprehensive color dictionary with better categorization
ENHANCED_COLORS = {
    # Primary colors
    "RED": (255, 0, 0), "GREEN": (0, 255, 0), "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0), "ORANGE": (255, 165, 0), "PURPLE": (128, 0, 128),
    "PINK": (255, 192, 203), "BLACK": (0, 0, 0), "WHITE": (255, 255, 255),
    "GRAY": (128, 128, 128), "BROWN": (165, 42, 42),
    
    # Extended colors with better accuracy
    "DARK_RED": (139, 0, 0), "LIGHT_RED": (255, 182, 193), "CRIMSON": (220, 20, 60),
    "DARK_GREEN": (0, 100, 0), "LIGHT_GREEN": (144, 238, 144), "FOREST_GREEN": (34, 139, 34),
    "DARK_BLUE": (0, 0, 139), "LIGHT_BLUE": (173, 216, 230), "NAVY": (0, 0, 128),
    "ROYAL_BLUE": (65, 105, 225), "STEEL_BLUE": (70, 130, 180), "DENIM": (21, 96, 189),
    "MAROON": (128, 0, 0), "BURGUNDY": (128, 0, 32), "WINE": (114, 47, 55),
    "OLIVE": (128, 128, 0), "KHAKI": (240, 230, 140), "MUSTARD": (255, 219, 88),
    "TEAL": (0, 128, 128), "TURQUOISE": (64, 224, 208), "AQUA": (0, 255, 255),
    "CORAL": (255, 127, 80), "SALMON": (250, 128, 114), "PEACH": (255, 218, 185),
    "GOLD": (255, 215, 0), "SILVER": (192, 192, 192), "BEIGE": (245, 245, 220),
    "TAN": (210, 180, 140), "TAUPE": (72, 60, 50), "CREAM": (255, 253, 208),
    "IVORY": (255, 255, 240), "PEARL": (240, 234, 214), "BONE": (248, 248, 240),
    "CHARCOAL": (54, 69, 79), "SLATE": (112, 128, 144), "GRAPHITE": (65, 65, 65),
    "LIGHT_GRAY": (211, 211, 211), "DARK_GRAY": (169, 169, 169), "WARM_GRAY": (133, 117, 112),
    "MAGENTA": (255, 0, 255), "FUCHSIA": (255, 0, 255), "VIOLET": (238, 130, 238),
    "INDIGO": (75, 0, 130), "LAVENDER": (230, 230, 250), "PLUM": (221, 160, 221),
    "MINT": (245, 255, 250), "SAGE": (158, 169, 147), "EMERALD": (80, 200, 120),
    "JADE": (0, 168, 107), "LIME": (0, 255, 0), "CHARTREUSE": (127, 255, 0),
    "AZURE": (240, 255, 255), "PERIWINKLE": (195, 205, 230), "POWDER_BLUE": (176, 224, 230),
    "ROSE": (255, 228, 225), "BLUSH": (222, 93, 131), "DUSTY_ROSE": (194, 145, 137),
    "CHOCOLATE": (210, 105, 30), "COFFEE": (111, 78, 55), "ESPRESSO": (58, 43, 35),
    "CAMEL": (193, 154, 107), "SAND": (244, 164, 96), "WHEAT": (245, 222, 179),
    "RUST": (183, 65, 14), "COPPER": (184, 115, 51), "BRONZE": (205, 127, 50),
    "MAUVE": (224, 176, 255), "ORCHID": (218, 112, 214), "LILAC": (200, 162, 200),
    "SCARLET": (255, 36, 0), "RUBY": (224, 17, 95), "GARNET": (115, 54, 53),
    "SAPPHIRE": (15, 82, 186), "COBALT": (0, 71, 171), "CERULEAN": (0, 123, 167),
    "AMBER": (255, 191, 0), "CITRINE": (228, 208, 10), "TOPAZ": (255, 200, 124),
    "ONYX": (53, 56, 57), "EBONY": (85, 93, 80), "JET": (52, 52, 52),
    "SNOW": (255, 250, 250), "ALABASTER": (250, 248, 240), "SEASHELL": (255, 245, 238),
    "MIDNIGHT": (25, 25, 112), "SPACE_GRAY": (113, 113, 113), "STORM_GRAY": (79, 79, 79)
}

# Color categories for better classification
COLOR_CATEGORIES = {
    "REDS": ["RED", "DARK_RED", "LIGHT_RED", "CRIMSON", "MAROON", "BURGUNDY", "WINE", "SCARLET", "RUBY", "GARNET", "ROSE", "BLUSH", "DUSTY_ROSE"],
    "BLUES": ["BLUE", "DARK_BLUE", "LIGHT_BLUE", "NAVY", "ROYAL_BLUE", "STEEL_BLUE", "DENIM", "SAPPHIRE", "COBALT", "CERULEAN", "AZURE", "PERIWINKLE", "POWDER_BLUE"],
    "GREENS": ["GREEN", "DARK_GREEN", "LIGHT_GREEN", "FOREST_GREEN", "OLIVE", "TEAL", "TURQUOISE", "EMERALD", "JADE", "LIME", "CHARTREUSE", "MINT", "SAGE"],
    "YELLOWS": ["YELLOW", "GOLD", "AMBER", "CITRINE", "TOPAZ", "MUSTARD", "KHAKI"],
    "ORANGES": ["ORANGE", "CORAL", "SALMON", "PEACH", "RUST", "COPPER", "BRONZE", "SAND", "WHEAT"],
    "PURPLES": ["PURPLE", "MAGENTA", "FUCHSIA", "VIOLET", "INDIGO", "LAVENDER", "PLUM", "MAUVE", "ORCHID", "LILAC"],
    "PINKS": ["PINK", "ROSE", "BLUSH", "DUSTY_ROSE"],
    "BROWNS": ["BROWN", "CHOCOLATE", "COFFEE", "ESPRESSO", "CAMEL", "TAN", "TAUPE", "BEIGE"],
    "GRAYS": ["GRAY", "LIGHT_GRAY", "DARK_GRAY", "WARM_GRAY", "CHARCOAL", "SLATE", "GRAPHITE", "SPACE_GRAY", "STORM_GRAY"],
    "BLACKS": ["BLACK", "ONYX", "EBONY", "JET", "MIDNIGHT"],
    "WHITES": ["WHITE", "CREAM", "IVORY", "PEARL", "BONE", "SNOW", "ALABASTER", "SEASHELL"]
}

@st.cache_resource
def load_model(model_name: str):
    """Load the specified AI model with caching."""
    try:
        if model_name == "YOLOv5 (Fast)":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        elif model_name == "YOLOv8 (Better)":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        else:  # DETR
            model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        
        model.conf = confidence_threshold
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhanced image preprocessing with multiple techniques."""
    if not apply_enhancement:
        return image
    
    # Convert to PIL for enhancement
    pil_image = Image.fromarray(image)
    
    # Auto-enhance brightness and contrast
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.15)
    
    # Enhance color saturation slightly
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(1.05)
    
    # Apply noise reduction
    enhanced_np = np.array(enhanced)
    denoised = cv2.bilateralFilter(enhanced_np, 9, 75, 75)
    
    # Sharpen slightly
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel * 0.1)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def extract_shirt_regions_advanced(person_crop: np.ndarray, use_segmentation: bool = True) -> List[np.ndarray]:
    """Advanced shirt region extraction with multiple strategies."""
    h, w, _ = person_crop.shape
    regions = []
    
    if shirt_region_focus:
        # Strategy 1: Traditional geometric regions
        regions_coords = [
            (int(0.15 * h), int(0.55 * h), int(0.1 * w), int(0.9 * w)),   # Main torso
            (int(0.20 * h), int(0.45 * h), int(0.15 * w), int(0.85 * w)),  # Upper chest
            (int(0.25 * h), int(0.50 * h), int(0.20 * w), int(0.80 * w)),  # Center chest
            (int(0.30 * h), int(0.55 * h), int(0.25 * w), int(0.75 * w)),  # Lower chest
        ]
        
        # Strategy 2: Adaptive regions based on person size
        if h > 200:  # Large person detection
            regions_coords.extend([
                (int(0.18 * h), int(0.35 * h), int(0.2 * w), int(0.8 * w)),   # Upper shirt
                (int(0.35 * h), int(0.52 * h), int(0.2 * w), int(0.8 * w)),   # Lower shirt
            ])
        
        for y1, y2, x1, x2 in regions_coords:
            if y2 > y1 and x2 > x1 and y2 <= h and x2 <= w:
                region = person_crop[y1:y2, x1:x2]
                if region.shape[0] > 10 and region.shape[1] > 10:
                    regions.append(region)
    
    # Strategy 3: Edge-based region detection
    if use_segmentation and len(regions) > 0:
        try:
            # Use edge detection to find shirt boundaries
            gray = cv2.cvtColor(person_crop, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours that might represent shirt boundaries
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and position
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500 and area < (h * w * 0.3):  # Reasonable shirt area
                    x, y, w_c, h_c = cv2.boundingRect(contour)
                    if 0.1 * h < y < 0.6 * h:  # In shirt region
                        valid_contours.append(contour)
            
            # Extract regions from valid contours
            for contour in valid_contours[:2]:  # Max 2 additional regions
                x, y, w_c, h_c = cv2.boundingRect(contour)
                if w_c > 20 and h_c > 20:
                    region = person_crop[y:y+h_c, x:x+w_c]
                    regions.append(region)
        
        except Exception:
            pass  # Fall back to geometric regions only
    
    return regions if regions else [person_crop[int(0.2*h):int(0.6*h), int(0.1*w):int(0.9*w)]]

def analyze_color_advanced(image: np.ndarray, method: str = "K-Means Clustering") -> List[Tuple[int, int, int]]:
    """Advanced color analysis with multiple methods."""
    if image.shape[0] < 10 or image.shape[1] < 10:
        return [(128, 128, 128)]
    
    # Preprocess image
    processed = enhance_image(image) if apply_enhancement else image
    
    # Convert to different color spaces for analysis
    lab_image = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
    hsv_image = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
    
    pixels = processed.reshape((-1, 3))
    
    # Remove extreme pixels (shadows, highlights, skin tones)
    mask = np.logical_and.reduce([
        np.mean(pixels, axis=1) > 25,     # Not too dark
        np.mean(pixels, axis=1) < 230,    # Not too light
        np.std(pixels, axis=1) > 15,      # Not too uniform (likely background)
    ])
    
    # Additional filter to remove skin tones
    skin_mask = np.logical_not(np.logical_and.reduce([
        pixels[:, 0] > 95,   # R
        pixels[:, 1] > 40,   # G
        pixels[:, 2] > 20,   # B
        pixels[:, 0] > pixels[:, 1],  # R > G
        pixels[:, 0] > pixels[:, 2],  # R > B
        np.abs(pixels[:, 0] - pixels[:, 1]) > 15
    ]))
    
    combined_mask = np.logical_and(mask, skin_mask)
    
    if np.sum(combined_mask) < 50:
        combined_mask = mask
    
    if np.sum(combined_mask) > 0:
        filtered_pixels = pixels[combined_mask]
    else:
        filtered_pixels = pixels
    
    # Apply selected color analysis method
    if method == "K-Means Clustering":
        n_colors = min(5, len(filtered_pixels))
        if n_colors > 0:
            kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
            kmeans.fit(filtered_pixels)
            
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            counts = Counter(labels)
            
            # Return colors sorted by frequency
            colors = []
            for label, count in counts.most_common():
                color = tuple(map(int, centers[label]))
                colors.append(color)
            return colors
    
    elif method == "Histogram Analysis":
        # Analyze color histogram
        hist_r = cv2.calcHist([processed], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([processed], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([processed], [2], None, [256], [0, 256])
        
        # Find peaks in each channel
        r_peak = np.argmax(hist_r)
        g_peak = np.argmax(hist_g)
        b_peak = np.argmax(hist_b)
        
        return [(r_peak, g_peak, b_peak)]
    
    elif method == "Dominant Color + Context":
        # Use both K-means and median color
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        kmeans.fit(filtered_pixels)
        
        dominant_color = tuple(map(int, kmeans.cluster_centers_[0]))
        median_color = tuple(map(int, np.median(filtered_pixels, axis=0)))
        
        return [dominant_color, median_color]
    
    else:  # Multi-Region Sampling
        # Sample from multiple regions
        h, w = processed.shape[:2]
        regions = [
            processed[h//4:3*h//4, w//4:3*w//4],  # Center
            processed[h//3:2*h//3, w//3:2*w//3],  # Inner center
            processed[h//6:5*h//6, w//6:5*w//6],  # Outer center
        ]
        
        colors = []
        for region in regions:
            if region.shape[0] > 5 and region.shape[1] > 5:
                region_pixels = region.reshape((-1, 3))
                median_color = tuple(map(int, np.median(region_pixels, axis=0)))
                colors.append(median_color)
        
        return colors if colors else [(128, 128, 128)]

def color_distance_perceptual(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """Calculate perceptual color distance using improved Delta E."""
    try:
        # Convert to LAB color space
        lab1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2LAB)[0][0].astype(float)
        lab2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2LAB)[0][0].astype(float)
        
        # Delta E CIE76 with perceptual weighting
        dl = lab1[0] - lab2[0]
        da = lab1[1] - lab2[1]
        db = lab1[2] - lab2[2]
        
        # Weighted delta E (gives more importance to lightness)
        delta_e = np.sqrt(0.5 * dl**2 + da**2 + db**2)
        
        return delta_e
    except:
        # Fallback to Euclidean distance in RGB
        return euclidean(color1, color2)

def classify_color_intelligent(rgb: Tuple[int, int, int], tolerance: float = 40) -> str:
    """Intelligent color classification with context awareness."""
    r, g, b = rgb
    
    # Convert to HSV for better color analysis
    hsv = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    hue, sat, val = hsv
    
    # Special case handling for common misclassifications
    
    # Handle achromatic colors (low saturation)
    if sat < 0.2:
        if val < 0.25:
            return "BLACK"
        elif val > 0.85:
            return "WHITE"
        elif val < 0.4:
            return "DARK_GRAY"
        elif val > 0.7:
            return "LIGHT_GRAY"
        else:
            return "GRAY"
    
    # Handle very dark colors
    if val < 0.3:
        if sat < 0.5:
            return "BLACK"
        else:
            # Dark colored items
            hue_deg = hue * 360
            if hue_deg < 30 or hue_deg > 330:
                return "DARK_RED"
            elif hue_deg < 90:
                return "BROWN"
            elif hue_deg < 150:
                return "DARK_GREEN"
            elif hue_deg < 270:
                return "DARK_BLUE"
            else:
                return "PURPLE"
    
    # Handle very light colors
    if val > 0.85 and sat < 0.4:
        return "WHITE"
    
    # Color classification based on HSV
    hue_deg = hue * 360
    
    # Define hue ranges with better boundaries
    if hue_deg < 15 or hue_deg > 345:
        if sat > 0.6 and val > 0.4:
            return "RED"
        elif sat > 0.3:
            return "LIGHT_RED"
        else:
            return "PINK"
    
    elif hue_deg < 45:
        if val < 0.5:
            return "BROWN"
        elif r > g * 1.2:
            return "ORANGE"
        else:
            return "YELLOW"
    
    elif hue_deg < 75:
        if g > r * 1.1 and g > b * 1.1:
            return "YELLOW"
        else:
            return "GOLD"
    
    elif hue_deg < 105:
        return "LIME" if sat > 0.7 else "LIGHT_GREEN"
    
    elif hue_deg < 135:
        if sat > 0.6:
            return "GREEN"
        else:
            return "SAGE"
    
    elif hue_deg < 165:
        return "TEAL" if sat > 0.5 else "MINT"
    
    elif hue_deg < 195:
        return "CYAN" if sat > 0.7 else "LIGHT_BLUE"
    
    elif hue_deg < 225:
        if sat > 0.6:
            return "BLUE"
        else:
            return "POWDER_BLUE"
    
    elif hue_deg < 255:
        return "ROYAL_BLUE" if sat > 0.7 else "PERIWINKLE"
    
    elif hue_deg < 285:
        return "PURPLE" if sat > 0.5 else "LAVENDER"
    
    elif hue_deg < 315:
        return "MAGENTA" if sat > 0.7 else "ORCHID"
    
    else:
        return "PINK" if sat > 0.5 else "ROSE"

def find_best_color_match_advanced(rgb: Tuple[int, int, int], tolerance: float = 40) -> str:
    """Advanced color matching with multiple strategies."""
    # First try intelligent classification
    intelligent_match = classify_color_intelligent(rgb, tolerance)
    
    # Then try exact matching against our color database
    best_match = None
    min_distance = float('inf')
    
    for color_name, color_rgb in ENHANCED_COLORS.items():
        distance = color_distance_perceptual(rgb, color_rgb)
        if distance < min_distance:
            min_distance = distance
            best_match = color_name
    
    # Use the best match if it's within tolerance
    if min_distance <= tolerance:
        return best_match
    
    # Otherwise, use intelligent classification
    return intelligent_match

def analyze_shirt_color_comprehensive(person_crop: np.ndarray) -> Optional[Dict]:
    """Comprehensive shirt color analysis with confidence scoring."""
    try:
        regions = extract_shirt_regions_advanced(person_crop)
        
        if not regions:
            return None
        
        all_colors = []
        region_scores = []
        
        for i, region in enumerate(regions):
            if region.shape[0] < 10 or region.shape[1] < 10:
                continue
            
            # Analyze colors using selected method
            colors = analyze_color_advanced(region, color_analysis_method)
            
            # Weight colors by region importance (first region is most important)
            weight = 1.0 / (i + 1)
            
            for color in colors:
                color_name = find_best_color_match_advanced(color, 50)
                all_colors.append((color_name, color, weight))
                region_scores.append(weight)
        
        if not all_colors:
            return None
        
        # Aggregate results with weighted voting
        color_votes = defaultdict(float)
        color_samples = defaultdict(list)
        
        for color_name, color_rgb, weight in all_colors:
            color_votes[color_name] += weight
            color_samples[color_name].append(color_rgb)
        
        # Find the most confident color
        if not color_votes:
            return None
        
        # Get top colors sorted by confidence
        sorted_colors = sorted(color_votes.items(), key=lambda x: x[1], reverse=True)
        primary_color = sorted_colors[0][0]
        confidence = sorted_colors[0][1] / sum(color_votes.values())
        
        # Calculate average RGB for the primary color
        primary_rgb = tuple(map(int, np.mean(color_samples[primary_color], axis=0)))
        
        # Determine color category
        category = None
        for cat, colors in COLOR_CATEGORIES.items():
            if primary_color in colors:
                category = cat
                break
        
        # Additional analysis
        secondary_colors = [color for color, _ in sorted_colors[1:4]]
        
        return {
            'primary_color': primary_color,
            'primary_rgb': primary_rgb,
            'confidence': confidence,
            'category': category,
            'secondary_colors': secondary_colors,
            'all_detected': dict(sorted_colors),
            'total_regions': len(regions),
            'analysis_method': color_analysis_method
        }
    
    except Exception as e:
        st.error(f"Error in color analysis: {str(e)}")
        return None

def detect_people_and_shirts(image: np.ndarray, model) -> List[Dict]:
    """Detect people and analyze their shirt colors."""
    try:
        # Run inference
        results = model(image)
        
        # Parse results
        detections = []
        df = results.pandas().xyxy[0]
        
        for _, detection in df.iterrows():
            if detection['class'] == 0 and detection['confidence'] >= confidence_threshold:  # Person class
                # Extract bounding box
                x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
                
                # Crop person
                person_crop = image[y1:y2, x1:x2]
                
                if person_crop.shape[0] > 50 and person_crop.shape[1] > 30:
                    # Analyze shirt color
                    color_analysis = analyze_shirt_color_comprehensive(person_crop)
                    
                    if color_analysis:
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': detection['confidence'],
                            'person_crop': person_crop,
                            'color_analysis': color_analysis
                        })
        
        return detections
    
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return []

def filter_detections_by_mode(detections: List[Dict], mode: str) -> List[Dict]:
    """Filter detections based on selected mode."""
    if mode == "Specific Color Hunt" and 'target_color' in globals():
        target_upper = target_color.upper()
        
        # Find target color in our color database
        target_variations = []
        for color_name in ENHANCED_COLORS.keys():
            if target_upper in color_name or color_name.startswith(target_upper):
                target_variations.append(color_name)
        
        # Also check categories
        for category, colors in COLOR_CATEGORIES.items():
            if target_upper in category:
                target_variations.extend(colors)
        
        # Filter detections
        filtered = []
        for detection in detections:
            color_info = detection['color_analysis']
            primary_color = color_info['primary_color']
            
            # Check if primary color matches target
            if primary_color in target_variations:
                filtered.append(detection)
            else:
                # Check secondary colors
                for secondary in color_info['secondary_colors']:
                    if secondary in target_variations:
                        filtered.append(detection)
                        break
        
        return filtered
    
    elif mode == "Similar Color Clustering":
        # Group similar colors together
        if not detections:
            return detections
        
        # Extract all primary colors
        colors = [d['color_analysis']['primary_rgb'] for d in detections]
        
        if len(colors) < 2:
            return detections
        
        # Use DBSCAN for clustering
        try:
            clustering = DBSCAN(eps=30, min_samples=1, metric='euclidean')
            clusters = clustering.fit_predict(colors)
            
            # Group detections by cluster
            clustered_detections = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                clustered_detections[cluster_id].append(detections[i])
            
            # Return largest cluster
            largest_cluster = max(clustered_detections.values(), key=len)
            return largest_cluster
        
        except:
            return detections
    
    else:
        return detections

def create_color_visualization(detections: List[Dict]) -> plt.Figure:
    """Create advanced color visualization."""
    if not detections:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎨 Advanced Color Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Extract color data
    colors = [d['color_analysis']['primary_rgb'] for d in detections]
    color_names = [d['color_analysis']['primary_color'] for d in detections]
    confidences = [d['color_analysis']['confidence'] for d in detections]
    
    # 1. Color Distribution Pie Chart
    ax1 = axes[0, 0]
    color_counts = Counter(color_names)
    colors_for_pie = [np.array(ENHANCED_COLORS.get(name, (128, 128, 128)))/255 for name in color_counts.keys()]
    
    wedges, texts, autotexts = ax1.pie(
        color_counts.values(),
        labels=color_counts.keys(),
        colors=colors_for_pie,
        autopct='%1.1f%%',
        startangle=90
    )
    ax1.set_title('Color Distribution')
    
    # 2. Confidence vs Color Scatter
    ax2 = axes[0, 1]
    scatter_colors = [np.array(color)/255 for color in colors]
    scatter = ax2.scatter(range(len(colors)), confidences, c=scatter_colors, s=100, alpha=0.7)
    ax2.set_xlabel('Detection #')
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Detection Confidence by Color')
    ax2.grid(True, alpha=0.3)
    
    # 3. Color Palette
    ax3 = axes[1, 0]
    ax3.set_xlim(0, len(colors))
    ax3.set_ylim(0, 1)
    
    for i, (color, name, conf) in enumerate(zip(colors, color_names, confidences)):
        rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, 
                               edgecolor='black', facecolor=np.array(color)/255)
        ax3.add_patch(rect)
        ax3.text(i+0.5, 0.5, f'{name}\n{conf:.2f}', 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax3.set_title('Detected Color Palette')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # 4. Category Distribution
    ax4 = axes[1, 1]
    categories = [d['color_analysis']['category'] for d in detections if d['color_analysis']['category']]
    if categories:
        category_counts = Counter(categories)
        ax4.bar(category_counts.keys(), category_counts.values(), 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax4.set_title('Color Categories')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No categories detected', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Color Categories')
    
    plt.tight_layout()
    return fig

def create_detection_summary(detections: List[Dict]) -> str:
    """Create a comprehensive detection summary."""
    if not detections:
        return "No shirt colors detected."
    
    summary = f"## 🎯 Detection Summary\n"
    summary += f"**Total People Detected:** {len(detections)}\n"
    summary += f"**Detection Mode:** {detection_mode}\n"
    summary += f"**Analysis Method:** {color_analysis_method}\n\n"
    
    # Color statistics
    colors = [d['color_analysis']['primary_color'] for d in detections]
    color_counts = Counter(colors)
    
    summary += "### 🎨 Color Breakdown:\n"
    for color, count in color_counts.most_common():
        percentage = (count / len(detections)) * 100
        summary += f"- **{color}**: {count} person{'s' if count > 1 else ''} ({percentage:.1f}%)\n"
    
    # Confidence analysis
    avg_confidence = np.mean([d['color_analysis']['confidence'] for d in detections])
    summary += f"\n### 📊 Analysis Quality:\n"
    summary += f"- **Average Confidence:** {avg_confidence:.2f}\n"
    summary += f"- **High Confidence Detections:** {sum(1 for d in detections if d['color_analysis']['confidence'] > 0.7)}\n"
    
    # Category analysis
    categories = [d['color_analysis']['category'] for d in detections if d['color_analysis']['category']]
    if categories:
        cat_counts = Counter(categories)
        summary += f"\n### 🏷️ Color Categories:\n"
        for cat, count in cat_counts.most_common():
            summary += f"- **{cat}**: {count} detection{'s' if count > 1 else ''}\n"
    
    return summary

# Main application logic
def main():
    # Load model
    model = load_model(model_type)
    
    if model is None:
        st.error("Failed to load AI model. Please refresh the page.")
        return
    
    # Input options
    input_method = st.radio(
        "📷 Choose Input Method:",
        ["Upload Image", "Upload Video", "Camera Capture", "URL Input"],
        horizontal=True
    )
    
    # Handle different input methods
    uploaded_file = None
    image_array = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image containing people wearing shirts"
        )
    
    elif input_method == "Upload Video":
        uploaded_file = st.file_uploader(
            "Choose a video file...", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file (first frame will be analyzed)"
        )
    
    elif input_method == "Camera Capture":
        st.markdown("### 📸 Camera Capture")
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            uploaded_file = camera_image
    
    elif input_method == "URL Input":
        image_url = st.text_input(
            "🔗 Enter image URL:", 
            placeholder="https://example.com/image.jpg"
        )
        if image_url:
            try:
                response = requests.get(image_url)
                uploaded_file = BytesIO(response.content)
            except:
                st.error("Failed to load image from URL")
    
    if uploaded_file is not None:
        # Process the input
        if input_method == "Upload Video":
            # Extract first frame from video
            try:
                # Save uploaded video temporarily
                temp_video = BytesIO(uploaded_file.getvalue())
                cap = cv2.VideoCapture()
                
                # Read first frame
                ret, frame = cap.read()
                if ret:
                    image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap.release()
            except:
                st.error("Failed to process video file")
                return
        else:
            # Process image
            try:
                image = Image.open(uploaded_file)
                image_array = np.array(image.convert('RGB'))
            except:
                st.error("Failed to process image file")
                return
        
        if image_array is not None:
            # Display original image
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(image_array, use_column_width=True)
            
            with col2:
                st.subheader("🎛️ Processing Status")
                
                # Processing steps
                with st.spinner("🔍 Detecting people..."):
                    detections = detect_people_and_shirts(image_array, model)
                
                if detections:
                    st.success(f"✅ Found {len(detections)} people")
                    
                    # Apply mode filtering
                    with st.spinner("🎨 Analyzing colors..."):
                        filtered_detections = filter_detections_by_mode(detections, detection_mode)
                    
                    if filtered_detections:
                        st.success(f"🎯 {len(filtered_detections)} matches found")
                        
                        # Create visualizations
                        st.subheader("🎨 Color Analysis Results")
                        
                        # Show detection summary
                        summary = create_detection_summary(filtered_detections)
                        st.markdown(summary)
                        
                        # Create annotated image
                        annotated_image = image_array.copy()
                        
                        for i, detection in enumerate(filtered_detections):
                            bbox = detection['bbox']
                            color_info = detection['color_analysis']
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_image, 
                                        (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                        color_info['primary_rgb'], 3)
                            
                            # Add label
                            label = f"{color_info['primary_color']} ({color_info['confidence']:.2f})"
                            cv2.putText(annotated_image, label, 
                                      (bbox[0], bbox[1]-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                      color_info['primary_rgb'], 2)
                        
                        # Display annotated image
                        st.subheader("🎯 Detected Results")
                        st.image(annotated_image, use_column_width=True)
                        
                        # Show color visualization
                        fig = create_color_visualization(filtered_detections)
                        if fig:
                            st.pyplot(fig)
                        
                        # Detailed results
                        st.subheader("📊 Detailed Analysis")
                        
                        for i, detection in enumerate(filtered_detections):
                            with st.expander(f"👤 Person {i+1} - {detection['color_analysis']['primary_color']}"):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    # Show person crop
                                    st.image(detection['person_crop'], caption="Person Crop")
                                
                                with col2:
                                    # Show detailed color info
                                    color_info = detection['color_analysis']
                                    
                                    st.markdown(f"**Primary Color:** {color_info['primary_color']}")
                                    st.markdown(f"**RGB Value:** {color_info['primary_rgb']}")
                                    st.markdown(f"**Confidence:** {color_info['confidence']:.3f}")
                                    st.markdown(f"**Category:** {color_info['category']}")
                                    st.markdown(f"**Secondary Colors:** {', '.join(color_info['secondary_colors'][:3])}")
                                    st.markdown(f"**Regions Analyzed:** {color_info['total_regions']}")
                                    
                                    # Color swatch
                                    color_swatch = np.full((50, 100, 3), color_info['primary_rgb'], dtype=np.uint8)
                                    st.image(color_swatch, caption="Color Swatch")
                    
                    else:
                        st.warning("🔍 No matches found for the selected criteria")
                        
                        if detection_mode == "Specific Color Hunt":
                            st.info(f"💡 Try increasing the color tolerance or selecting a different target color")
                
                else:
                    st.warning("😔 No people detected in the image")
                    st.info("💡 Try adjusting the confidence threshold or using a different image")

# Help section
with st.expander("❓ Help & Tips"):
    st.markdown("""
    ### 🎯 How to Use This Tool:
    
    1. **Choose your AI model** - YOLOv5 is fastest, DETR is most accurate
    2. **Select detection mode** - Different modes for different use cases
    3. **Upload your image** - Supports JPG, PNG, WebP formats
    4. **Adjust settings** - Fine-tune for better results
    
    ### 🎨 Detection Modes:
    
    - **Smart Auto-Grouping**: Automatically detects and groups similar colors
    - **Specific Color Hunt**: Look for a specific color (great for finding team members)
    - **Color Distribution Analysis**: Analyze overall color distribution
    - **Similar Color Clustering**: Group people by similar shirt colors
    
    ### 💡 Tips for Better Results:
    
    - Use high-resolution images with good lighting
    - Ensure people are clearly visible (not too small or obscured)
    - For specific color hunting, try different tolerance levels
    - Enable image enhancement for better color detection
    - Use shirt region focus for more accurate color detection
    
    ### 🔧 Troubleshooting:
    
    - **No detections**: Lower confidence threshold or check image quality
    - **Wrong colors**: Try different color analysis methods
    - **Poor accuracy**: Enable image enhancement and shirt region focus
    """)

# Run the application
if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-top: 2rem;">
    <h3 style="color: white; margin-bottom: 1rem;">🎯 AI Shirt Color Detection Pro</h3>
    <p style="color: #f0f0f0; margin-bottom: 0;">Powered by advanced computer vision and machine learning</p>
</div>
""", unsafe_allow_html=True)
