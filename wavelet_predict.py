import os
import argparse
import pandas as pd
import numpy as np
import librosa
import pywt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import label_ranking_average_precision_score
from tqdm import tqdm

# Define the 1D-CNN architecture from wavelet_train.py
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, dropout=False, pool_size=None):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size=ksize, padding=ksize//2), nn.ReLU(inplace=True)]
        if pool_size:
            layers.append(nn.MaxPool1d(pool_size))
        if dropout:
            layers.append(nn.Dropout(0.1))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class WaveletCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 16, 9),
            ConvBlock(16, 16, 9, pool_size=16, dropout=True),
            ConvBlock(16, 32, 3),
            ConvBlock(32, 32, 3, pool_size=4, dropout=True),
            ConvBlock(32, 32, 3),
            ConvBlock(32, 32, 3, pool_size=4, dropout=True),
            ConvBlock(32, 256, 3),
            ConvBlock(256, 256, 3),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class WaveletInferenceDataset(Dataset):
    def __init__(self, files, data_dir, label2idx=None, csv_path=None, resample_rate=16000, duration=2.0):
        self.files = files
        self.data_dir = data_dir
        self.rate = resample_rate
        self.samples = int(resample_rate * duration)
        self.label2idx = label2idx
        self.do_eval = csv_path is not None
        if self.do_eval:
            df = pd.read_csv(csv_path)
            self.targets = df['labels'].map(label2idx).values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.data_dir, fname)
        y, sr = librosa.load(path, sr=self.rate)
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[:self.samples]
        y = (y - y.mean()) / (y.std() + 1e-8)
        coeffs = pywt.wavedec(y, wavelet='db4', level=4)
        chans = []
        for c in coeffs:
            c = np.asarray(c)
            if c.shape[0] < self.samples:
                c = np.pad(c, (0, self.samples - c.shape[0]))
            else:
                c = c[:self.samples]
            chans.append(c)
        feat = np.stack(chans, axis=0)
        x = torch.from_numpy(feat).float()
        if self.do_eval:
            return x, self.targets[idx]
        else:
            return x, fname

def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), ys

def compute_lwlrap(y_true, y_score):
    # one-hot convert
    y_onehot = np.zeros_like(y_score)
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1
    return label_ranking_average_precision_score(y_onehot, y_score)

def parse_args():
    parser = argparse.ArgumentParser(description='Wavelet model inference')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--csv-path', type=str, required=True, help='train_curated.csv for classes and eval')
    parser.add_argument('--eval-csv', type=str, default=None)
    parser.add_argument('--output-csv', type=str, default='wavelet_submission.csv')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    unique = sorted(df['labels'].unique())
    label2idx = {l:i for i,l in enumerate(unique)}
    idx2label = {i:l for l,i in label2idx.items()}
    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.wav')])
    dataset = WaveletInferenceDataset(files, args.data_dir, label2idx, args.eval_csv,)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaveletCNN(in_channels=len(pywt.wavedec(np.zeros(dataset.samples), 'db4', level=4)),
                       num_classes=len(unique)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    all_scores, all_true, all_fnames = [], [], []
    with torch.no_grad():
        for x, y_or_f in tqdm(loader, desc='Inference'):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_scores.append(probs)
            if args.eval_csv:
                all_true.extend(y_or_f)
            else:
                all_fnames.extend(y_or_f)
    scores = np.vstack(all_scores)
    if args.eval_csv:
        print('LWLRAP:', compute_lwlrap(np.array(all_true), scores))
    out_df = pd.DataFrame(scores, columns=unique)
    out_df.insert(0, 'fname', all_fnames)
    out_df.to_csv(args.output_csv, index=False)
    print(f'Saved to {args.output_csv}')

if __name__ == '__main__':
    main()
