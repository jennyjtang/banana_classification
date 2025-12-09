# Integrated Banana Ripeness Classification

Automated banana ripeness detection combining CNN and color analysis for real-time assessment.

## Overview

Determines banana ripeness objectively to reduce food waste and improve purchasing decisions.

**Key Features:**
- CNN binary classification (Ripe/Unripe)
- HSV color analysis for ripeness percentage (0-100%)
- Five-stage maturity assessment
- Real-time webcam processing
- Integrated predictions using confidence-weighted logic

## Installation
```bash
git clone https://github.com/jennyjtang/banana_classification
cd banana_classification
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

Download the Ripe–Unripe Banana Dataset from [this link](https://data.mendeley.com/datasets/y3649cmgg6/1)

Place images in:
```
data_banana/
├── train/images/
├── val/images/
└── test/images/
```

## Usage

**Train the model:**
```bash
python main.py
```

**Run the live classification:**
```bash
python live_classify.py
```

## Project Structure
```
banana_classification/
├── src/
│   ├── cnn.py
│   ├── dataset.py
│   └── train.py
├── main.py
├── live_classify.py
├── color_analyzer.py
└── data_banana/
```

## Results

- CNN Accuracy:
  [Nora insert here]
- Integrated system combines CNN + color analysis for fine-grained ripeness assessment

### Live Classification Demo (+ Controls)
<img width="797" height="591" alt="Screenshot 2025-12-08 224144" src="https://github.com/user-attachments/assets/73c38565-b06a-40a7-ac69-4a9decf7c2b2" />

*System detecting a banana with ripeness assessment showing ripeness classification (ripe), ripeness bucket (30-50%), and color distribution.*

- I to toggle modes
- Q to quit

## Authors

- Nora Amer
- Jen (Jenny) Tang

## Citation

Rahman, M. M., & Al Faisal, S. M. (2021). Ripe–unripe banana dataset [Data set](https://data.mendeley.com/datasets/y3649cmgg6/1).

## AI Usage Statement

AI tools (Claude) were used for conceptual understanding, debugging assistance, and implementation feedback. All core design decisions and experimental work were conducted by the project team.
