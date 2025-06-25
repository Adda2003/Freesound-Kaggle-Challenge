import os
import argparse
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import label_ranking_average_precision_score
from tqdm import tqdm

# Constants matching train_cnn2d.py
SR = 22050
DURATION = 2.0
SAMPLES = int(SR * DURATION)
N_MFCC = 40
HOP_LENGTH = 512
MAX_FRAMES = int(np.ceil(SAMPLES / HOP_LENGTH))

class CNN2D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64 * (N_MFCC//4) * (MAX_FRAMES//4), 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class InferenceDataset(Dataset):
    def __init__(self, file_list, data_dir, label2idx=None, eval_csv=None):
        self.files = file_list
        self.data_dir = data_dir
        self.eval = eval_csv is not None
        if self.eval:
            df = pd.read_csv(eval_csv)
            # map labels to indices
            self.targets = df['labels'].map(label2idx).values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.data_dir, fname)
        y, _ = librosa.load(path, sr=SR)
        # pad/truncate
        if len(y) < SAMPLES:
            y = np.pad(y, (0, SAMPLES - len(y)))
        else:
            y = y[:SAMPLES]
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        if mfcc.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_FRAMES]
        x = torch.from_numpy(mfcc).unsqueeze(0).float()
        if self.eval:
            return x, self.targets[idx]
        else:
            return x, fname


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for CNN2D MFCC model')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to wav files')
    parser.add_argument('--model-path', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--csv-path', type=str, required=True, help='Path to train_curated.csv')
    parser.add_argument('--eval-csv', type=str, default=None, help='CSV for evaluation mode')
    parser.add_argument('--output-csv', type=str, default='submission_cnn2d.csv')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def compute_lwlrap(y_true, y_score):
    y_onehot = np.zeros_like(y_score)
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1
    return label_ranking_average_precision_score(y_onehot, y_score)


def main():
    args = parse_args()
    # load class mapping
    df = pd.read_csv(args.csv_path)
    labels = sorted(df['labels'].unique())
    label2idx = {l:i for i,l in enumerate(labels)}

    # list wav files
    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.wav')])
    dataset = InferenceDataset(files, args.data_dir, label2idx, eval_csv=args.eval_csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN2D(num_classes=len(labels)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_scores, all_true, all_fnames = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            x, y_or_f = batch
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
    # build submission
    out = pd.DataFrame(scores, columns=labels)
    out.insert(0, 'fname', all_fnames)
    out.to_csv(args.output_csv, index=False)
    print(f'Saved to {args.output_csv}')

if __name__ == '__main__':
    main()
