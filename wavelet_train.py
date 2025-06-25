import os
import argparse
import pandas as pd
import numpy as np
import librosa
import pywt  # Wavelet transforms library; ensure dependency is installed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import label_ranking_average_precision_score
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 1D-CNN with DWT features for audio tagging"
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
        '--batch-size', type=int, default=16,
        help='Number of samples per batch (default: 16)'
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=1e-5,
        help='Weight decay (default: 1e-5)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='wavelet_checkpoints',
        help='Directory for saving checkpoints (default: wavelet_checkpoints)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from latest checkpoint'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of DataLoader workers (default: 4)'
    )
    return parser.parse_args()


class WaveletAudioDataset(Dataset):
    def __init__(self, csv_path, data_dir, label2idx, resample_rate=16000, duration=2.0):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.rate = resample_rate
        self.samples = int(resample_rate * duration)
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, sr = librosa.load(os.path.join(self.data_dir, row['fname']), sr=self.rate)
        # Random offset + pad/truncate to fixed length
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            max_offset = len(y) - self.samples
            offset = np.random.randint(0, max_offset + 1)  # random crop
            y = y[offset:offset + self.samples]
        # Normalize per-sample to avoid extreme values
        y = (y - y.mean()) / (y.std() + 1e-8)
        # Perform DWT; check that 'db4' and level=4 are appropriate for data length
        coeffs = pywt.wavedec(y, wavelet='db4', level=4)
        # Pad or truncate each coefficient array to uniform length
        chans = []
        for c in coeffs:
            c = np.asarray(c)
            if c.shape[0] < self.samples:
                c = np.pad(c, (0, self.samples - c.shape[0]))
            else:
                c = c[:self.samples]
            chans.append(c)
        feats = np.stack(chans, axis=0)
        x = torch.from_numpy(feats).float()
        y_lbl = self.label2idx[row['labels']]
        return x, y_lbl


def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


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
        # Ensure channel count matches number of DWT levels+1
        self.net = nn.Sequential(
            ConvBlock(in_channels, 16, 9),
            ConvBlock(16, 16, 9, pool_size=16, dropout=True),
            ConvBlock(16, 32, 3),
            ConvBlock(32, 32, 3, pool_size=4, dropout=True),
            ConvBlock(32, 32, 3),
            ConvBlock(32, 32, 3, pool_size=4, dropout=True),
            ConvBlock(32, 256, 3),
            ConvBlock(256, 256, 3),
            nn.AdaptiveMaxPool1d(1),  # global max over time
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def save_checkpoint(state, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))


def load_latest_checkpoint(checkpoint_dir, model, optimizer):
    # return 0 if no checkpoints
    if not os.path.isdir(checkpoint_dir):
        return 0
    # only consider files named epoch_<n>.pth
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('epoch_') and f.endswith('.pth')
    ]
    if not files:
        return 0
    # extract epoch numbers
    epochs = [int(f[len('epoch_'):].split('.')[0]) for f in files]
    latest_epoch = max(epochs)
    path = os.path.join(checkpoint_dir, f'epoch_{latest_epoch}.pth')
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    return latest_epoch


def compute_lwlrap(y_true, y_score):
    # Convert integer labels to one-hot for metric
    y_onehot = np.zeros_like(y_score)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    return label_ranking_average_precision_score(y_onehot, y_score)


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build label index mapping once
    df = pd.read_csv(args.csv_path)
    unique_labels = df['labels'].unique()
    label2idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    full = WaveletAudioDataset(args.csv_path, args.data_dir, label2idx)
    idxs = np.random.permutation(len(full))
    split = int(0.8 * len(full))
    train_ds = torch.utils.data.Subset(full, idxs[:split])
    val_ds = torch.utils.data.Subset(full, idxs[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    model = WaveletCNN(in_channels=len(pywt.wavedec(np.zeros(full.samples), 'db4', level=4)),
                       num_classes=len(label2idx)).to(device)  # dynamic channel count
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = load_latest_checkpoint(args.checkpoint_dir, model, optimizer) + 1 if args.resume else 1

    best_lwlrap = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_true, all_scores = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Val"):
                logits = model(x.to(device))
                all_scores.append(torch.softmax(logits, 1).cpu().numpy())
                all_true.extend(y.numpy())
        all_scores = np.vstack(all_scores)
        lwlrap = compute_lwlrap(np.array(all_true), all_scores)
        print(f"Epoch {epoch}: LWLRAP={lwlrap:.4f}")

        # Save
        save_checkpoint({'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()},
                        args.checkpoint_dir, epoch)
        if lwlrap > best_lwlrap:
            best_lwlrap = lwlrap
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))

    print(f"Finished. Best LWLRAP: {best_lwlrap:.4f}")


if __name__ == '__main__':
    train()
