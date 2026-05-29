# Green Area Detector

A satellite image classification system that detects and calculates 
the percentage of green areas in a given satellite image using 
deep learning.

## How It Works
- Divides the satellite image into a grid of cells
- Classifies each cell as green or non-green using a CNN-based model
- Displays the percentage of green coverage in the image

## Tech Stack
- **Model:** EfficientNet (Transfer Learning)
- **Framework:** TensorFlow / Keras
- **Dataset:** EuroSAT (satellite imagery dataset)
- **Interface:** Streamlit web app

## Files
- `app.py` — Streamlit web application
- `final_model_fixed.h5` — trained EfficientNet model
- `training/` — Jupyter notebook with model training code
