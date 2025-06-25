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

# Reuse AudioCNN from train_model
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

class InferenceDataset(Dataset):
    def __init__(self, file_list, data_dir, mlb=None, csv_path=None):
        self.files = file_list
        self.data_dir = data_dir
        self.mlb = mlb
        self.do_eval = csv_path is not None
        if self.do_eval:
            df = pd.read_csv(csv_path)
            df['label_list'] = df['labels'].apply(lambda x: x.split(','))
            self.targets = self.mlb.transform(df['label_list'])
        else:
            self.targets = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.data_dir, fname)
        y, sr = librosa.load(path, sr=None)
        # feature extraction
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
        S_db = librosa.power_to_db(S, ref=np.max)
        delta = librosa.feature.delta(S_db)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        feat = np.stack([S_db, delta, contrast], axis=0)
        x = torch.from_numpy(feat).float()
        if self.do_eval:
            y = torch.from_numpy(self.targets[idx]).float()
            return x, y
        else:
            return x, fname

def collate_fn(batch):
    xs = [item[0] for item in batch]
    if len(batch[0]) == 2:
        ys = [item[1] for item in batch]
    else:
        ys = None
    # pad to max time
    max_t = max(x.shape[2] for x in xs)
    xs_pad = []
    for x in xs:
        c, f, t = x.shape
        if t < max_t:
            pad = torch.zeros(c, f, max_t - t)
            x = torch.cat([x, pad], dim=2)
        xs_pad.append(x)
    x_batch = torch.stack(xs_pad)
    if ys is not None:
        return x_batch, torch.stack(ys)
    else:
        fnames = [item[1] for item in batch]
        return x_batch, fnames


def compute_lwlrap(y_true, y_score):
    return label_ranking_average_precision_score(y_true, y_score)


def parse_args():
    parser = argparse.ArgumentParser(description='Predict and evaluate model')
    parser.add_argument('--data-dir',    type=str, required=True, help='Path to audio files (wav)')
    parser.add_argument('--model-path',  type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--csv-path',    type=str, required=True,
                        help='Path to metadata CSV (train_curated.csv)')      # renamed
    parser.add_argument('--eval-csv',    type=str, default=None,
                        help='If set, run in evaluation mode against this CSV')
    parser.add_argument('--output-csv',  type=str, default='submission.csv',
                        help='Where to save your predictions')
    parser.add_argument('--batch-size',  type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # load classes and label binarizer
    df = pd.read_csv(args.classes)
    df['label_list'] = df['labels'].apply(lambda x: x.split(','))
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(df['label_list'])
    class_names = mlb.classes_

    # list audio files
    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.wav')])

    dataset = InferenceDataset(files, args.data_dir, mlb, csv_path=args.eval_csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_scores = []
    all_true = []
    all_fnames = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predict'):  
            x, y_or_f = batch
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)
            if args.eval_csv:
                all_true.append(y_or_f.numpy())
            else:
                all_fnames.extend(y_or_f)

    all_scores = np.vstack(all_scores)
    if args.eval_csv:
        all_true = np.vstack(all_true)
        print('LWLRAP:', compute_lwlrap(all_true, all_scores))

    # build submission
    df_out = pd.DataFrame(all_scores, columns=class_names)
    df_out.insert(0, 'fname', all_fnames)
    df_out.to_csv(args.output_csv, index=False)
    print(f'Saved predictions to {args.output_csv}')

if __name__ == '__main__':
    main()
