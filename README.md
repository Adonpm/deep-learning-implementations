# Deep Learning Implementations

Implementations of deep learning models (MLP, CNN, RNN) from scratch and using PyTorch, applied to real-world datasets as part of MTech AIML coursework at BITS Pilani.

---

## Assignments

### 1. ANN — Ride Fare Prediction (`uber-data-analysis-ann`)

**Task:** Regression — predicting Uber/Lyft ride fares in USD  
**Dataset:** [Uber & Lyft Rideshare Dataset, Boston MA](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma) — 637,628 rows, 14 features  
**Framework:** NumPy (implemented from scratch — no sklearn for the model)

| Model | RMSE | R² |
|---|---|---|
| Linear Regression (baseline) | $6.52 | 0.513 |
| MLP (3-layer, mini-batch GD) | **$1.82** | **0.962** |

**Key result:** MLP achieved a **3.6x reduction in RMSE** over the linear baseline, explaining 96.2% of fare variance.

---

### 2. CNN — Chest X-Ray Pneumonia Detection (`chest-xray-analysis-cnn`)

**Task:** Binary image classification — NORMAL vs PNEUMONIA  
**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,856 images  
**Framework:** PyTorch  
**Primary metric:** Recall (medical imaging — false negatives carry higher clinical cost than false alarms)

| Model | Accuracy | Recall (macro) | F1 (macro) |
|---|---|---|---|
| Custom CNN (trained end-to-end) | **94.7%** | **91.6%** | **93.2%** |
| ResNet18 (Transfer Learning, frozen) | 92.7% | 90.8% | 90.9% |

**Key result:** Custom CNN outperformed frozen ResNet18 transfer learning — end-to-end training on domain-specific grayscale X-ray images learned more task-relevant features than frozen ImageNet weights.

---

### 3. RNN — Climate Temperature Forecasting (`jena-climate-analysis-rnn`)

**Task:** Univariate time series forecasting — predicting next-hour temperature (°C)  
**Dataset:** [Jena Climate Dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip) (Max Planck Institute) — 70,092 hourly records, 2009–2016  
**Framework:** PyTorch  
**Primary metric:** MAE (interpretable in °C — directly meaningful for forecasting)

| Model | MAE (°C) | RMSE (°C) | R² | Training Time |
|---|---|---|---|---|
| LSTM (2 stacked layers) | **0.49** | **0.70** | **0.992** | 141s |
| Transformer (2-layer encoder) | 0.53 | 0.77 | 0.990 | 423s |

**Key result:** LSTM outperformed the Transformer across all metrics and trained **3x faster**, making it the better choice for this univariate sequence length (30 steps). The Transformer's self-attention mechanism offers less advantage at shorter sequence lengths.

---

## Repository Structure

```
deep-learning-implementations/
├── uber-data-analysis-ann/
│   ├── 2025ag05644_assignment1.ipynb
│   └── 2025ag05644_assignment1.html
├── chest-xray-analysis-cnn/
│   ├── 2025ag05644_cnn_assignment.ipynb
│   └── 2025ag05644_cnn_assignment.html
└── jena-climate-analysis-rnn/
    ├── 2025ag05644_rnn_assignment.ipynb
    └── 2025ag05644_rnn_assignment.html
```

---

## Tech Stack

- **NumPy** — ANN implemented from scratch (forward pass, backprop, mini-batch gradient descent)
- **PyTorch** — CNN and RNN/Transformer implementations
- **Matplotlib / Seaborn** — training curves and result visualisations
- **Scikit-learn** — preprocessing and evaluation metrics

---

## Context

These assignments are part of the **MTech in Artificial Intelligence & Machine Learning** program at **BITS Pilani** (2026–2028).
