# 💊 VialVision AI
**AI-Powered Pharmaceutical Vial Detection, Counting & Quality Control**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vialvision-ai.streamlit.app)
![YOLOv11](https://img.shields.io/badge/YOLOv11s-96.9%25_mAP50-green)

> M.Tech Final Year Project — Geeta Siddramayya Math | 1MS24RAI03  
> Ramaiah Institute of Technology, Bengaluru

---

## Features
- Real-time vial detection & quality classification (Sealed / Unsealed / Damaged)
- BotSort tracking + virtual line counting (no double counting)
- Image, Video & Webcam inference
- Downloadable HTML + CSV inspection reports

## Model
| Metric | Value |
|---|---|
| mAP@50 | 96.9% |
| mAP@50-95 | 94.1% |
| Speed | ~106 FPS |
| Dataset | 25,608 images |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requirements
```txt
streamlit>=1.28.0
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
Pillow>=9.0.0
numpy>=1.24.0
```

## Files
```
app.py          # Streamlit app
best.pt         # YOLOv11s weights (18MB)
requirements.txt
```

"A fully functional prototype is publicly accessible at https://vialvision-ai-pharma.streamlit.app for real-time demonstration and evaluation."

## NOTE
# ⚠ Live webcam only works when running locally.<br>
For cloud use — run: <b>streamlit run app.py</b> on your machine.
