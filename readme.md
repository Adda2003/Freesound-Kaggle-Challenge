# Freesound Audio Tagging 2019 — PyTorch Pipelines

This repository provides end-to-end PyTorch implementations for the Kaggle/DCASE Freesound Audio Tagging 2019 (curated subset) competition. It includes:

- **Exploratory analysis** of the curated dataset
- **Baseline 2D‑CNN** on MFCC features (following the Kaggle "Beginner’s Guide to Audio Data" notebook)
- **1D‑CNN** on raw waveform with Discrete Wavelet Transform (DWT) features
- **Variable‑length spectrogram CNN** (log‑mel + delta + spectral contrast)
- **Inference scripts** for each model type
- **Checkpointing** and resume support

---

## 📁 Repository Structure

```
freesound-tagging/
├── train_curated/            # .wav files (curated subset)
├── train_curated.csv         # metadata: fname, labels
├── data_analysis.py          # EDA: class distribution, waveforms, spectrograms, durations
├── train_model.py            # 2D‑CNN on mel+delta+contrast
├── wavelet_train.py          # 1D‑CNN on DWT features
├── train_cnn2d.py            # 2D‑CNN on MFCC features
|-- train.py                  # final script for multi-class and multi-label
├── predict_and_eval.py       # inference for train_model.py
├── wavelet_predict.py        # inference for wavelet_train.py
├── predict_cnn2d.py          # inference for train_cnn2d.py
├── requirements.txt          # Python dependencies
└── README.md                 # this file
```

---

## 📊 1. Data Analysis (`data_analysis.py`)

Performs exploratory analysis on the curated subset:

1. **Class distribution** — bar chart of tag frequencies
2. **Sample waveforms** — random clips plotted in time domain
3. **Sample spectrograms** — mel‑spectrogram visualizations
4. **Duration stats** — min, max, mean, median clip lengths (CSV output)

**Usage**:

```bash\npython
  --data-dir ./train_curated/ \
  --csv-path ./train_curated.csv \
  --output-dir analysis/
```

Outputs are saved under `analysis/`.

---

## 🛠 2. Training Pipelines

All training scripts share a common pattern:

- **Fixed preprocessing** of audio (padding/truncation)
- **Feature extraction** (spectrograms, MFCCs, or DWT)
- **PyTorch **``** + **`` with batching
- **Model architecture** ([Conv2D, Conv1D] + pooling + dense)
- **Training loop** with:
  - Cross‑entropy (single‑label) or BCEWithLogits (multi‑label)
  - Adam optimizer + weight decay
  - Per‑epoch checkpointing (`epoch_{n}.pth`)
  - Best‑model saving (`best_model.pth`)
  - `--resume` support to continue from the latest epoch

### 2.1 2D‑CNN on Mel+Delta+Contrast (`train_model.py`)

- **Input**: 3‑channel tensor: log‑mel spectrogram, delta, spectral contrast
- **Model**: 3×2×2 conv blocks, adaptive avg pool, linear classifier
- **Metric**: label‑weighted label ranking AP (LWLRAP)

```bash
python train_model.py \
  --data-dir ./train_curated/ \
  --csv-path ./train_curated.csv \
  --checkpoint-dir checkpoints/ \
  [--resume]
```

### 2.2 1D‑CNN on DWT Features (`wavelet_train.py`)

- **Input**: 5‑channel DWT coefficients (Daubechies‑4, 4 levels)
- **Model**: 1D‑CNN (Conv1D + pooling) matching Kaggle notebook flow
- **Metric**: LWLRAP

```bash
python wavelet_train.py \
  --data-dir ./train_curated/ \
  --csv-path ./train_curated.csv \
  --checkpoint-dir wavelet_checkpoints/ \
  [--resume]
```

### 2.3 2D‑CNN on MFCC (`train_cnn2d.py`)

- **Input**: 40‑dim MFCC matrix (40×\~87 frames)
- **Model**: 2 Conv2D blocks + dense layers (as in Kaggle Notebook)
- **Metric**: LWLRAP / accuracy

```bash
python train_cnn2d.py \
  --data-dir ./train_curated/ \
  --csv-path ./train_curated.csv \
  --checkpoint-dir cnn2d_checkpoints/ \
  [--resume]
```

---

## 🎯 3. Inference and Evaluation

Each pipeline has a matching predict script:

- **predict\_and\_eval.py** for `train_model.py`
- **wavelet\_predict.py** for `wavelet_train.py`
- **predict\_cnn2d.py** for `train_cnn2d.py`

They:

1. Load `best_model.pth` and class mapping (`train_curated.csv`)
2. Preprocess audio identically to training
3. Batch through DataLoader
4. Output `submission.csv` with header:
   ```
   ```

fname,Accelerating\_and\_revving\_and\_vroom,...,Zipper\_(clothing) 000ccb97.wav,0.1,...,0.3 ...\`\`\`\
5\. Optionally compute LWLRAP if an eval CSV is provided

**Example**:

```bash
python predict_cnn2d.py \
  --data-dir ./test_curated/ \
  --model-path cnn2d_checkpoints/best_model.pth \
  --csv-path ./train_curated.csv \
  --output-csv submission_cnn2d.csv
```

---

## 📋 Dependencies

See `requirements.txt` for exact versions:

```text
pandas
numpy
librosa
soundfile
matplotlib
scikit-learn
tqdm
torch
torchaudio
pywt
pyyaml  # optional
```

---

## 🔄 Resume & Checkpoints

- All training scripts use `--resume` to continue from the latest `epoch_{n}.pth`.
- Checkpoints are **not** committed—add `*/checkpoints/` to `.gitignore`.
- To keep disk usage low, consider:
  - Pruning older epochs (only keep `best_model.pth`).
  - Using rolling deletes in `save_checkpoint()`.

---

## 🛠 Next Steps

- Experiment with **data augmentation** (SpecAugment, time/frequency masking).
- Try **ensemble** of the three pipelines.
- Fine‑tune hyperparameters in `configs/` using a scheduler.

Happy tagging! 🚀

