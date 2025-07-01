import os
import argparse
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, label_ranking_average_precision_score
from tqdm import tqdm

# Constants for MFCC
SR = 22050
DURATION = 2.0  # seconds
SAMPLES = int(SR * DURATION)
N_MFCC = 40
HOP_LENGTH = 512
MAX_FRAMES = int(np.ceil(SAMPLES / HOP_LENGTH))  # ~87


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 2D-CNN on MFCC features for audio tagging"
    )
    parser.add_argument(
        '--data-dir', type=str, default='train_curated/',
        help='Path to train_curated/ directory (default: train_curated/)'
    )
    parser.add_argument(
        '--csv-path', type=str, default='train_curated.csv',
        help='Path to train_curated.csv (default: train_curated.csv)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Number of samples per batch (default: 32)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=1e-4,
        help='Weight decay (default: 1e-4)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='new_checkpoints',
        help='Directory for saving checkpoints (default: new_checkpoints)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of DataLoader workers (default: 4)'
    )
    return parser.parse_args()


class MFCCDataset(Dataset):
    def __init__(self, csv_path, data_dir, label2idx):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, _ = librosa.load(os.path.join(self.data_dir, row['fname']), sr=SR)
        # pad/truncate
        if len(y) < SAMPLES:
            y = np.pad(y, (0, SAMPLES - len(y)))
        else:
            y = y[:SAMPLES]
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        # normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        # pad frames
        if mfcc.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_FRAMES]
        # to tensor
        x = torch.from_numpy(mfcc).unsqueeze(0).float()  # (1, N_MFCC, MAX_FRAMES)
        y_lbl = self.label2idx[row['labels']]
        return x, y_lbl


def save_checkpoint(state, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))


def load_latest_checkpoint(checkpoint_dir, model, optimizer):
    if not os.path.isdir(checkpoint_dir):
        return 0
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_')]
    if not files:
        return 0
    epochs = [int(f.split('_')[1].split('.')[0]) for f in files]
    latest = max(epochs)
    ckpt = torch.load(os.path.join(checkpoint_dir, f'epoch_{latest}.pth'))
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    return latest


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


def compute_lwlrap(y_true, y_score):
    # one-hot encode
    y_onehot = np.zeros_like(y_score)
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1
    return label_ranking_average_precision_score(y_onehot, y_score)


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(args.csv_path)
    labels = sorted(df['labels'].unique())
    label2idx = {l:i for i,l in enumerate(labels)}

    full = MFCCDataset(args.csv_path, args.data_dir, label2idx)
    idxs = np.random.permutation(len(full))
    split = int(0.8 * len(full))
    train_ds = torch.utils.data.Subset(full, idxs[:split])
    val_ds = torch.utils.data.Subset(full, idxs[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CNN2D(num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start = load_latest_checkpoint(args.checkpoint_dir, model, optimizer) + 1 if args.resume else 1
    best_lwlrap = 0.0
    for epoch in range(start, args.epochs+1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Train {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        all_true, all_scores = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, 1).cpu().numpy()
                all_scores.append(probs)
                all_true.extend(y.numpy())
        scores = np.vstack(all_scores)
        lwlrap = compute_lwlrap(np.array(all_true), scores)
        print(f"Epoch {epoch}: LWLRAP={lwlrap:.4f}")

        save_checkpoint({'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()},
                        args.checkpoint_dir, epoch)
        if lwlrap > best_lwlrap:
            best_lwlrap = lwlrap
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))

    print(f"Done. Best LWLRAP: {best_lwlrap:.4f}")

if __name__ == '__main__':
    train()