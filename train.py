import os
import argparse
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants for MFCC
SR = 22050
DURATION = 10.0  # seconds
SAMPLES = int(SR * DURATION)
N_MFCC = 40
HOP_LENGTH = 512
MAX_FRAMES = int(np.ceil(SAMPLES / HOP_LENGTH))  # ~87


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 2D-CNN on MFCC features for multi-label audio tagging"
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
        '--epochs', type=int, default=30,
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
        '--checkpoint-dir', type=str, default='cnn2d_checkpoints',
        help='Directory for saving checkpoints (default: cnn2d_checkpoints)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of DataLoader workers (default: 4)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Threshold for binary predictions (default: 0.5)'
    )
    return parser.parse_args()


def create_label_mappings(df):
    """Create mappings between labels and indices"""
    # Get all unique labels from the dataset
    all_labels = set()
    for labels_str in df['labels']:
        labels = labels_str.split(',')
        all_labels.update(labels)
    
    # Sort for consistent ordering
    all_labels = sorted(list(all_labels))
    
    # Create bidirectional mappings
    label2idx = {label: idx for idx, label in enumerate(all_labels)}
    idx2label = {idx: label for idx, label in enumerate(all_labels)}
    
    return label2idx, idx2label, len(all_labels)


class MFCCDataset(Dataset):
    def __init__(self, csv_path, data_dir, label2idx):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.label2idx = label2idx
        self.num_classes = len(label2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and preprocess audio
        y, _ = librosa.load(os.path.join(self.data_dir, row['fname']), sr=SR)
        
        # Pad/truncate to fixed length
        if len(y) < SAMPLES:
            y = np.pad(y, (0, SAMPLES - len(y)))
        else:
            y = y[:SAMPLES]
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        
        # Normalize features (per-sample normalization)
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # Pad/truncate frames to fixed length
        if mfcc.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_FRAMES]
        
        # Convert to tensor and add channel dimension
        x = torch.from_numpy(mfcc).unsqueeze(0).float()  # (1, N_MFCC, MAX_FRAMES)
        
        # Create multi-label target vector
        labels = row['labels'].split(',')
        y_multi = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in self.label2idx:  # Safety check
                y_multi[self.label2idx[label]] = 1.0
        
        return x, y_multi


def save_checkpoint(state, checkpoint_dir, epoch):
    """Save model and optimizer state"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))


def load_latest_checkpoint(checkpoint_dir, model, optimizer):
    """Load the latest checkpoint"""
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
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)), 
            nn.Dropout(0.3),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)), 
            nn.Dropout(0.3),
            
            # Classification head
            nn.Flatten(),
            nn.Linear(64 * (N_MFCC//4) * (MAX_FRAMES//4), 128), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  # No activation - BCEWithLogitsLoss handles sigmoid
        )

    def forward(self, x):
        return self.net(x)


def compute_lwlrap(y_true, y_scores):
    """Compute Label-Weighted Label-Ranking Average Precision"""
    return label_ranking_average_precision_score(y_true, y_scores)


def evaluate_model(model, data_loader, criterion, device, threshold=0.5):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_scores = []
    all_predictions = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            # Get probabilities using sigmoid
            probs = torch.sigmoid(logits)
            
            # Get binary predictions using threshold
            preds = (probs > threshold).float()
            
            # Store for metrics computation
            all_targets.append(y.cpu().numpy())
            all_scores.append(probs.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
    
    # Concatenate all batches
    targets = np.vstack(all_targets)
    scores = np.vstack(all_scores)
    predictions = np.vstack(all_predictions)
    
    # Compute metrics
    avg_loss = total_loss / len(data_loader)
    lwlrap = compute_lwlrap(targets, scores)
    
    # Compute per-class accuracy and overall metrics
    correct_predictions = (predictions == targets).sum()
    total_predictions = targets.size
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, lwlrap, accuracy, targets, scores, predictions


def stratified_split(df, test_size=0.2, random_state=42):
    """
    Create stratified split for multi-label data using iterative approach
    Falls back to random split if iterative stratification fails
    """
    try:
        # Try to use iterative stratification (requires skmultilearn)
        from skmultilearn.model_selection import iterative_train_test_split
        
        # Create multi-label matrix
        label2idx, _, num_classes = create_label_mappings(df)
        y_multilabel = np.zeros((len(df), num_classes))
        
        for idx, labels_str in enumerate(df['labels']):
            labels = labels_str.split(',')
            for label in labels:
                if label in label2idx:
                    y_multilabel[idx, label2idx[label]] = 1
        
        # Perform iterative stratification
        X_indices = np.arange(len(df)).reshape(-1, 1)
        X_train, y_train, X_test, y_test = iterative_train_test_split(
            X_indices, y_multilabel, test_size=test_size
        )
        
        train_indices = X_train.flatten()
        val_indices = X_test.flatten()
        
        print("Using iterative stratification for multi-label split")
        
    except ImportError:
        print("skmultilearn not available, using random split")
        # Fall back to random split
        indices = np.random.RandomState(random_state).permutation(len(df))
        split_point = int((1 - test_size) * len(df))
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
    
    return train_indices, val_indices


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data and create label mappings
    df = pd.read_csv(args.csv_path)
    label2idx, idx2label, num_classes = create_label_mappings(df)
    
    print(f"Dataset: {len(df)} samples")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(label2idx.keys())}")

    # Create datasets with proper multi-label handling
    full_dataset = MFCCDataset(args.csv_path, args.data_dir, label2idx)
    
    # Create stratified train/validation split
    train_indices, val_indices = stratified_split(df, test_size=0.2, random_state=42)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize model, loss, and optimizer
    model = CNN2D(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Multi-label loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Resume from checkpoint if requested
    start_epoch = 1
    if args.resume:
        start_epoch = load_latest_checkpoint(args.checkpoint_dir, model, optimizer) + 1
        print(f"Resumed training from epoch {start_epoch}")

    best_lwlrap = 0.0
    best_epoch = 0
    
    print("\nStarting training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs}")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{train_loss/train_batches:.4f}'
            })

        # Validation phase
        val_loss, lwlrap, accuracy, _, _, _ = evaluate_model(
            model, val_loader, criterion, device, args.threshold
        )
        
        # Update learning rate scheduler
        scheduler.step(lwlrap)
        
        # Print epoch results
        print(f"\nEpoch {epoch:2d}/{args.epochs}:")
        print(f"  Train Loss: {train_loss/train_batches:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LWLRAP:     {lwlrap:.4f}")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        checkpoint_state = {
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'epoch': epoch,
            'lwlrap': lwlrap,
            'label2idx': label2idx,
            'idx2label': idx2label
        }
        save_checkpoint(checkpoint_state, args.checkpoint_dir, epoch)
        
        # Save best model
        if lwlrap > best_lwlrap:
            best_lwlrap = lwlrap
            best_epoch = epoch
            torch.save(checkpoint_state, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"  ★ New best LWLRAP: {best_lwlrap:.4f}")
        
        print("-" * 50)

    print(f"\nTraining completed!")
    print(f"Best LWLRAP: {best_lwlrap:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: {args.checkpoint_dir}/best_model.pth")


if __name__ == '__main__':
    train()