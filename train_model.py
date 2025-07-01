import os
import argparse
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio tagging model")
    parser.add_argument('--data-dir', type=str, default='train_curated/',
                        help='Path to train_curated/ directory (default: train_curated/)')
    parser.add_argument('--csv-path', type=str, default='train_curated.csv',
                        help='Path to train_curated.csv (default: train_curated.csv)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_cnn2d')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


class AudioDataset(Dataset):
    def __init__(self, csv_path, data_dir, mlb=None):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.df['labels'] = self.df['labels'].apply(lambda x: x.split(','))
        if mlb is None:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit(self.df['labels'])
        else:
            self.mlb = mlb
        self.targets = self.mlb.transform(self.df['labels'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_dir, row['fname'])
        y, sr = librosa.load(path, sr=16000)

        # Mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                           n_fft=1024, hop_length=512)
        S_db = librosa.power_to_db(S, ref=np.max)

        # First and second deltas (both shape 128×T)
        delta1 = librosa.feature.delta(S_db, order=1)
        delta2 = librosa.feature.delta(S_db, order=2)

        # Stack into 3 channels
        features = np.stack([S_db, delta1, delta2], axis=0)

        # Convert to tensor
        features = torch.from_numpy(features).float()
        target = torch.from_numpy(self.targets[idx]).float()
        return features, target


def collate_fn(batch):
    # batch: list of (features, target)
    xs, ys = zip(*batch)
    # Find max time dimension
    max_t = max(x.shape[2] for x in xs)
    # Pad time dimension
    xs_pad = []
    for x in xs:
        c, f, t = x.shape
        if t < max_t:
            pad = torch.zeros(c, f, max_t - t)
            x = torch.cat([x, pad], dim=2)
        xs_pad.append(x)
    x_batch = torch.stack(xs_pad)
    y_batch = torch.stack(ys)
    return x_batch, y_batch


class ConvBnReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBnReLU(3, 32), nn.MaxPool2d(2),
            ConvBnReLU(32, 64), nn.MaxPool2d(2),
            ConvBnReLU(64, 128), nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.conv_blocks(x)
        g = self.global_pool(h).view(h.size(0), -1)
        return self.classifier(g)


def save_checkpoint(state, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    torch.save(state, filename)


def load_latest_checkpoint(checkpoint_dir, model, optimizer):
    # only consider files named "epoch_<n>.pth"
    files = [f for f in os.listdir(checkpoint_dir)
             if f.startswith('epoch_') and f.endswith('.pth')]
    if not files:
        return 0
    # extract epoch numbers
    epochs = [int(f[len('epoch_'):].split('.')[0]) for f in files]
    latest_epoch = max(epochs)
    path = os.path.join(checkpoint_dir, f'epoch_{latest_epoch}.pth')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    return latest_epoch


def compute_lwlrap(y_true, y_score):
    # y_true, y_score: numpy arrays, shape (n_samples, n_classes)
    return label_ranking_average_precision_score(y_true, y_score)


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    full = AudioDataset(args.csv_path, args.data_dir)
    mlb = full.mlb
    # Stratified split (80/20) using number of labels per sample
    # Simplest: random split
    idx = np.arange(len(full))
    np.random.shuffle(idx)
    split = int(0.8 * len(full))
    train_idx, val_idx = idx[:split], idx[split:]
    train_ds = torch.utils.data.Subset(full, train_idx)
    val_ds = torch.utils.data.Subset(full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=args.num_workers)

    model = AudioCNN(num_classes=len(mlb.classes_)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    start_epoch = 1
    if args.resume:
        start_epoch = load_latest_checkpoint(args.checkpoint_dir, model, optimizer) + 1
        print(f"Resuming from epoch {start_epoch}")

    best_lwlrap = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses, all_true, all_scores = [], [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x = x.to(device)
                logits = model(x)
                loss = criterion(logits, y.to(device))
                val_losses.append(loss.item())
                scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(scores)
                all_true.append(y.numpy())
        all_scores = np.vstack(all_scores)
        all_true = np.vstack(all_true)
        lwlrap = compute_lwlrap(all_true, all_scores)
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, LWLRAP = {lwlrap:.4f}")

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'lwlrap': lwlrap
        }, args.checkpoint_dir, epoch)

        # Track best
        if lwlrap > best_lwlrap:
            best_lwlrap = lwlrap
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))

    print(f"Training complete. Best LWLRAP: {best_lwlrap:.4f}")


if __name__ == '__main__':
    train()
