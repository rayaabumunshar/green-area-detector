# app.py - Green Area Detector
# Run with: streamlit run app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenScope — Satellite Analysis",
    page_icon="🌿",
    layout="wide"
)

# ── CSS Injection ─────────────────────────────────────────────────────────────
# No extra libraries needed — st.markdown with unsafe_allow_html=True
st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --ink:       #0e1a0f;
    --paper:     #f4f1ea;
    --moss:      #2d5a27;
    --sage:      #6b9e63;
    --pale:      #c8dfc4;
    --accent:    #a3c96d;
    --warm:      #e8e0d0;
    --muted:     #7a8c78;
    --danger:    #c0392b;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    color: var(--ink);
}

/* ── App background ── */
.stApp {
    background-color: var(--paper);
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(107,158,99,0.10) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 100%, rgba(45,90,39,0.08) 0%, transparent 50%);
    min-height: 100vh;
}

/* ── Header band ── */
.gs-header {
    border-bottom: 2px solid var(--ink);
    padding: 2.5rem 0 1.5rem;
    margin-bottom: 2.5rem;
}
.gs-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 400;
    letter-spacing: -0.02em;
    color: var(--ink);
    line-height: 1;
    margin: 0 0 0.3rem;
}
.gs-header p {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: var(--pale) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--accent) !important;
    font-size: 1.1rem !important;
    font-weight: 400 !important;
    border-bottom: 1px solid rgba(163,201,109,0.3);
    padding-bottom: 0.5rem;
}
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(163,201,109,0.4) !important;
    border-radius: 2px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *  {
    color: var(--accent) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebar"] .stInfo {
    background: rgba(163,201,109,0.12) !important;
    border: 1px solid rgba(163,201,109,0.3) !important;
    border-radius: 2px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--moss) !important;
    border-radius: 4px !important;
    background: rgba(45,90,39,0.03) !important;
    padding: 1rem !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--sage) !important;
    background: rgba(45,90,39,0.06) !important;
}
[data-testid="stFileUploader"] label {
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stFileUploader"] button {
    background: var(--moss) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--sage), var(--accent)) !important;
    border-radius: 0 !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--warm) !important;
    border-radius: 0 !important;
    height: 3px !important;
}

/* ── Info / success / error messages ── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em !important;
}
div[data-testid="stAlert"][data-baseweb="notification"] {
    border-left: 3px solid var(--sage) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--warm);
    border: 1px solid rgba(14,26,15,0.12);
    padding: 1rem 1.2rem;
    border-radius: 2px;
}
[data-testid="stMetric"] label {
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
    color: var(--moss) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(14,26,15,0.15) !important;
    border-radius: 2px !important;
    background: var(--warm) !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── Tables (inside markdown) ── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin: 1rem 0;
}
th {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1.5px solid var(--ink);
    padding: 0.4rem 0.8rem;
    text-align: left;
}
td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid rgba(14,26,15,0.08);
}
tr:last-child td { border-bottom: none; }

/* ── Report card ── */
.gs-report {
    background: var(--warm);
    border: 1px solid rgba(14,26,15,0.12);
    border-radius: 2px;
    padding: 1.5rem;
    height: 100%;
}
.gs-report h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    font-weight: 400;
    color: var(--ink);
    margin: 0 0 1rem;
    border-bottom: 1px solid rgba(14,26,15,0.1);
    padding-bottom: 0.6rem;
}

/* ── Hide sidebar collapse button (renders as keyboard_double_arrow text) ── */
[data-testid="collapsedControl"],
button[kind="header"],
.st-emotion-cache-1lb4qcp,
section[data-testid="stSidebar"] > div > div:first-child button {
    display: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--paper); }
::-webkit-scrollbar-thumb { background: var(--pale); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--sage); }

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── pyplot figures ── */
[data-testid="stImage"], .stPlotlyChart {
    border: 1px solid rgba(14,26,15,0.1);
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gs-header">
    <h1>Green Area Detector</h1>
    <p>Satellite green space analysis &nbsp;·&nbsp; EfficientNet classifier</p>
</div>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    model_path = "final_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model not found. Place '{model_path}' in the same directory as this app.")
        return None
    return load_model(model_path)


model = load_my_model()


# ── Class definitions ─────────────────────────────────────────────────────────
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

GREEN_CLASSES = [
    'Forest', 'HerbaceousVegetation', 'Pasture',
    'AnnualCrop', 'PermanentCrop', 'River', 'SeaLake'
]


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown('<p style="font-family:\'DM Serif Display\',serif;color:#a3c96d;font-size:1.1rem;font-weight:400;border-bottom:1px solid rgba(163,201,109,0.3);padding-bottom:0.5rem;margin-bottom:1rem;">Settings</p>', unsafe_allow_html=True)

grid_size = st.sidebar.selectbox(
    "Grid resolution",
    options=[(5, 5), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)],
    format_func=lambda x: f"{x[0]}×{x[1]}  —  {x[0] * x[1]} cells",
    index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Green-class labels**

Forest · HerbaceousVegetation  
Pasture · AnnualCrop  
PermanentCrop · River · SeaLake
""")


# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a satellite image  —  JPG or PNG",
    type=["jpg", "jpeg", "png"]
)


# ── Core functions ────────────────────────────────────────────────────────────
def predict_cell(cell_img, model, class_names):
    cell_img = cell_img.resize((128, 128))
    if cell_img.mode != "RGB":
        cell_img = cell_img.convert("RGB")

    img_array = image.img_to_array(cell_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)
    sorted_indices = np.argsort(predictions[0])[::-1]

    predicted_idx = sorted_indices[0]
    predicted_class = class_names[predicted_idx]

    if predicted_class in ("Highway", "River"):
        predicted_idx = sorted_indices[1]
        predicted_class = class_names[predicted_idx]

    confidence = predictions[0][predicted_idx]
    return predicted_class, confidence


def analyze_green_area_grid(img, model, class_names, green_classes, grid_size=(8, 8)):
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_width, img_height = img.size
    rows, cols = grid_size
    cell_width  = img_width  // cols
    cell_height = img_height // rows

    grid_results      = []
    cell_predictions  = []
    cell_confidences  = []

    progress_bar = st.progress(0)
    status_text  = st.empty()

    for row in range(rows):
        row_results     = []
        row_predictions = []
        row_confidences = []

        for col in range(cols):
            left, top     = col * cell_width,  row * cell_height
            right, bottom = left + cell_width, top + cell_height

            cell_img = img.crop((left, top, right, bottom))
            predicted_class, confidence = predict_cell(cell_img, model, class_names)
            is_green = predicted_class in green_classes

            row_results.append(is_green)
            row_predictions.append(predicted_class)
            row_confidences.append(confidence)

            progress = (row * cols + col + 1) / (rows * cols)
            progress_bar.progress(progress)
            status_text.text(f"Scanning cell {row * cols + col + 1} of {rows * cols}")

        grid_results.append(row_results)
        cell_predictions.append(row_predictions)
        cell_confidences.append(row_confidences)

    status_text.empty()
    progress_bar.empty()

    total_cells      = rows * cols
    green_cells      = sum(sum(r) for r in grid_results)
    green_percentage = (green_cells / total_cells) * 100

    return grid_results, cell_predictions, cell_confidences, green_percentage, img.size


def create_visualization(img, grid_results, green_percentage, img_size, grid_size=(8, 8)):
    if img.mode != "RGB":
        img = img.convert("RGB")

    rows, cols = grid_size
    img_width, img_height = img_size

    # Refined matplotlib style
    plt.rcParams.update({
        'font.family':      'monospace',
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'figure.facecolor': '#f4f1ea',
        'axes.facecolor':   '#f4f1ea',
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.06)

    # Left: satellite image + overlay
    axes[0].imshow(img)
    axes[0].set_title(
        f"Coverage  ·  {green_percentage:.1f}% green",
        fontsize=10, color='#0e1a0f', loc='left', pad=10,
        fontfamily='monospace'
    )
    axes[0].axis("off")

    cell_width  = img_width  // cols
    cell_height = img_height // rows

    for row in range(rows):
        for col in range(cols):
            color = (0.18, 0.62, 0.15, 0.28) if grid_results[row][col] else (0.75, 0.22, 0.17, 0.18)
            rect = patches.Rectangle(
                (col * cell_width, row * cell_height),
                cell_width, cell_height,
                linewidth=0.4,
                edgecolor='white',
                facecolor=color
            )
            axes[0].add_patch(rect)

    # Right: heatmap
    heatmap_data = np.array(grid_results, dtype=float)
    im = axes[1].imshow(heatmap_data, cmap="RdYlGn", vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(
        "Density map",
        fontsize=10, color='#0e1a0f', loc='left', pad=10,
        fontfamily='monospace'
    )
    green_cells = int(sum(sum(r) for r in grid_results))
    axes[1].set_xlabel(
        f"{green_cells} / {rows * cols} green cells",
        fontsize=8, color='#7a8c78', fontfamily='monospace'
    )
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)

    cb = plt.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Non-green', 'Green'], fontsize=7, fontfamily='monospace')
    cb.outline.set_visible(False)

    return fig


def generate_report(green_percentage, grid_results, cell_predictions, grid_size):
    rows, cols   = grid_size
    total_cells  = rows * cols
    green_cells  = sum(sum(r) for r in grid_results)
    non_green    = total_cells - green_cells

    report = f"""
### Analysis summary

| | |
|---|---|
| Grid | {rows}×{cols} &nbsp; ({total_cells} cells) |
| Green cells | {green_cells} |
| Non-green | {non_green} |
| **Coverage** | **{green_percentage:.1f}%** |
"""
    return report


# ── Main flow ─────────────────────────────────────────────────────────────────
if model is not None and uploaded_file is not None:
    img = Image.open(uploaded_file)
    if img.mode != "RGB":
        img = img.convert("RGB")

    st.info(f"Resolution: {grid_size[0]}×{grid_size[1]} grid — {grid_size[0] * grid_size[1]} cells")

    with st.spinner("Running classifier..."):
        grid_results, cell_predictions, cell_confidences, green_percentage, img_size = \
            analyze_green_area_grid(img, model, class_names, GREEN_CLASSES, grid_size)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = create_visualization(img, grid_results, green_percentage, img_size, grid_size)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        report = generate_report(green_percentage, grid_results, cell_predictions, grid_size)
        st.markdown(report)

    with st.expander("Cell-level predictions"):
        st.write(cell_predictions)

    # st.success("Analysis complete.")

elif model is None:
    st.error("Model not loaded. Place 'final_model.keras' in the app directory.")

# else:
    # st.info("Upload a satellite image above to begin.")
