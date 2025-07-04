import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from sklearn.cluster import KMeans

st.set_page_config(page_title="Shirt Color Detector", layout="centered")
st.title("👕 Shirt Color Detector")
st.write("Upload an image to detect shirt colors worn by people.")

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

COMMON_COLORS = {
    "ALICEBLUE": (240, 248, 255),
    "ANTIQUEWHITE": (250, 235, 215),
    "AQUA": (0, 255, 255),
    "AQUAMARINE": (127, 255, 212),
    "AZURE": (240, 255, 255),
    "BEIGE": (245, 245, 220),
    "BISQUE": (255, 228, 196),
    "BLACK": (0, 0, 0),
    "BLANCHEDALMOND": (255, 235, 205),
    "BLUE": (0, 0, 255),
    "BLUEVIOLET": (138, 43, 226),
    "BROWN": (165, 42, 42),
    "BURLYWOOD": (222, 184, 135),
    "CADETBLUE": (95, 158, 160),
    "CHARTREUSE": (127, 255, 0),
    "CHOCOLATE": (210, 105, 30),
    "CORAL": (255, 127, 80),
    "CORNFLOWERBLUE": (100, 149, 237),
    "CORNSILK": (255, 248, 220),
    "CRIMSON": (220, 20, 60),
    "CYAN": (0, 255, 255),
    "DARKBLUE": (0, 0, 139),
    "DARKCYAN": (0, 139, 139),
    "DARKGOLDENROD": (184, 134, 11),
    "DARKGRAY": (169, 169, 169),
    "DARKGREEN": (0, 100, 0),
    "DARKKHAKI": (189, 183, 107),
    "DARKMAGENTA": (139, 0, 139),
    "DARKOLIVEGREEN": (85, 107, 47),
    "DARKORANGE": (255, 140, 0),
    "DARKORCHID": (153, 50, 204),
    "DARKRED": (139, 0, 0),
    "DARKSALMON": (233, 150, 122),
    "DARKSEAGREEN": (143, 188, 143),
    "DARKSLATEBLUE": (72, 61, 139),
    "DARKSLATEGRAY": (47, 79, 79),
    "DARKTURQUOISE": (0, 206, 209),
    "DARKVIOLET": (148, 0, 211),
    "DEEPPINK": (255, 20, 147),
    "DEEPSKYBLUE": (0, 191, 255),
    "DIMGRAY": (105, 105, 105),
    "DODGERBLUE": (30, 144, 255),
    "FIREBRICK": (178, 34, 34),
    "FLORALWHITE": (255, 250, 240),
    "FORESTGREEN": (34, 139, 34),
    "FUCHSIA": (255, 0, 255),
    "GAINSBORO": (220, 220, 220),
    "GHOSTWHITE": (248, 248, 255),
    "GOLD": (255, 215, 0),
    "GOLDENROD": (218, 165, 32),
    "GRAY": (128, 128, 128),
    "GREEN": (0, 128, 0),
    "GREENYELLOW": (173, 255, 47),
    "HONEYDEW": (240, 255, 240),
    "HOTPINK": (255, 105, 180),
    "INDIANRED": (205, 92, 92),
    "INDIGO": (75, 0, 130),
    "IVORY": (255, 255, 240),
    "KHAKI": (240, 230, 140),
    "LAVENDER": (230, 230, 250),
    "LAVENDERBLUSH": (255, 240, 245),
    "LAWNGREEN": (124, 252, 0),
    "LEMONCHIFFON": (255, 250, 205),
    "LIGHTBLUE": (173, 216, 230),
    "LIGHTCORAL": (240, 128, 128),
    "LIGHTCYAN": (224, 255, 255),
    "LIGHTGOLDENRODYELLOW": (250, 250, 210),
    "LIGHTGRAY": (211, 211, 211),
    "LIGHTGREEN": (144, 238, 144),
    "LIGHTPINK": (255, 182, 193),
    "LIGHTSALMON": (255, 160, 122),
    "LIGHTSEAGREEN": (32, 178, 170),
    "LIGHTSKYBLUE": (135, 206, 250),
    "LIGHTSLATEGRAY": (119, 136, 153),
    "LIGHTSTEELBLUE": (176, 196, 222),
    "LIGHTYELLOW": (255, 255, 224),
    "LIME": (0, 255, 0),
    "LIMEGREEN": (50, 205, 50),
    "LINEN": (250, 240, 230),
    "MAGENTA": (255, 0, 255),
    "MAROON": (128, 0, 0),
    "MEDIUMAQUAMARINE": (102, 205, 170),
    "MEDIUMBLUE": (0, 0, 205),
    "MEDIUMORCHID": (186, 85, 211),
    "MEDIUMPURPLE": (147, 112, 219),
    "MEDIUMSEAGREEN": (60, 179, 113),
    "MEDIUMSLATEBLUE": (123, 104, 238),
    "MEDIUMSPRINGGREEN": (0, 250, 154),
    "MEDIUMTURQUOISE": (72, 209, 204),
    "MEDIUMVIOLETRED": (199, 21, 133),
    "MIDNIGHTBLUE": (25, 25, 112),
    "MINTCREAM": (245, 255, 250),
    "MISTYROSE": (255, 228, 225),
    "MOCCASIN": (255, 228, 181),
    "NAVAJOWHITE": (255, 222, 173),
    "NAVY": (0, 0, 128),
    "OLDLACE": (253, 245, 230),
    "OLIVE": (128, 128, 0),
    "OLIVEDRAB": (107, 142, 35),
    "ORANGE": (255, 165, 0),
    "ORANGERED": (255, 69, 0),
    "ORCHID": (218, 112, 214),
    "PALEGOLDENROD": (238, 232, 170),
    "PALEGREEN": (152, 251, 152),
    "PALETURQUOISE": (175, 238, 238),
    "PALEVIOLETRED": (219, 112, 147),
    "PAPAYAWHIP": (255, 239, 213),
    "PEACHPUFF": (255, 218, 185),
    "PERU": (205, 133, 63),
    "PINK": (255, 192, 203),
    "PLUM": (221, 160, 221),
    "POWDERBLUE": (176, 224, 230),
    "PURPLE": (128, 0, 128),
    "REBECCAPURPLE": (102, 51, 153),
    "RED": (255, 0, 0),
    "ROSYBROWN": (188, 143, 143),
    "ROYALBLUE": (65, 105, 225),
    "SADDLEBROWN": (139, 69, 19),
    "SALMON": (250, 128, 114),
    "SANDYBROWN": (244, 164, 96),
    "SEAGREEN": (46, 139, 87),
    "SEASHELL": (255, 245, 238),
    "SIENNA": (160, 82, 45),
    "SILVER": (192, 192, 192),
    "SKYBLUE": (135, 206, 235),
    "SLATEBLUE": (106, 90, 205),
    "SLATEGRAY": (112, 128, 144),
    "SNOW": (255, 250, 250),
    "SPRINGGREEN": (0, 255, 127),
    "STEELBLUE": (70, 130, 180),
    "TAN": (210, 180, 140),
    "TEAL": (0, 128, 128),
    "THISTLE": (216, 191, 216),
    "TOMATO": (255, 99, 71),
    "TURQUOISE": (64, 224, 208),
    "VIOLET": (238, 130, 238),
    "WHEAT": (245, 222, 179),
    "WHITE": (255, 255, 255),
    "WHITESMOKE": (245, 245, 245),
    "YELLOW": (255, 255, 0),
    "YELLOWGREEN": (154, 205, 50)
}


def get_dominant_color_kmeans(image, k=3):
    """Return the most prominent shirt fabric color using KMeans clustering."""
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    counts = Counter(kmeans.labels_)
    cluster_centers = kmeans.cluster_centers_

    sorted_clusters = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    for cluster_idx, _ in sorted_clusters:
        color = cluster_centers[cluster_idx]
        if np.mean(color) < 230:  
            return tuple(map(int, color))
    
    return tuple(map(int, cluster_centers[sorted_clusters[0][0]]))

def closest_color(rgb):
    min_dist = float("inf")
    closest_name = None
    for name, ref_rgb in COMMON_COLORS.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(ref_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

uploaded_file = st.file_uploader("📂 Upload an image", type=['jpg', 'jpeg', 'png', 'webp', 'avif'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))

    results = model(image_np)
    df = results.pandas().xyxy[0]
    people = df[df['name'] == 'person']

    if people.empty:
        st.warning("No people detected in the image.")
    else:
        image_copy = image_np.copy()
        color_summary = {}

        for idx, row in people.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            person_crop = image_np[y1:y2, x1:x2]
            h, w, _ = person_crop.shape

            y_top = int(0.25 * h)
            y_bottom = int(0.50 * h)
            shirt_crop = person_crop[y_top:y_bottom, :]

            if shirt_crop.shape[0] < 20 or shirt_crop.shape[1] < 20:
                continue

            dom_color = get_dominant_color_kmeans(shirt_crop)
            color_name = closest_color(dom_color)
            color_summary.setdefault(color_name, []).append(idx + 1)

            label = f"{color_name.title()} Shirt"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_copy, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(image_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(image_copy, caption="🧍 Detected People + Shirt Colors", width=300)

        st.subheader("📊 Summary")
        for color, ids in color_summary.items():
            st.markdown(f"👕 **{color.title()}** — {len(ids)} person(s) → IDs: {ids}")
