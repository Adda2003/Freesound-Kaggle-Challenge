import os
import argparse
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for Freesound Audio Tagging 2019 curated subset"
    )
    parser.add_argument(
        "--data-dir", type=str, default="train_curated/",
        help="Path to the directory containing train_curated/ (default: train_curated/)"
    )
    parser.add_argument(
        "--csv-path", type=str, default="train_curated.csv",
        help="Path to train_curated.csv (default: train_curated.csv)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="analysis/",
        help="Directory to save analysis plots (default: analysis/)"
    )
    return parser.parse_args()


def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df['label_list'] = df['labels'].apply(lambda x: x.split(','))
    return df


def plot_class_distribution(df, output_dir):
    # Flatten list of labels
    all_labels = [lbl for sublist in df['label_list'] for lbl in sublist]
    label_counts = pd.Series(all_labels).value_counts()

    plt.figure(figsize=(12, 6))
    label_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Audio Tag')
    plt.ylabel('Number of Clips')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()


def plot_sample_waveform(df, data_dir, output_dir, n_samples=5):
    sample_files = df['fname'].sample(n_samples, random_state=42).values
    plt.figure(figsize=(12, 2 * n_samples))
    for idx, fname in enumerate(sample_files, 1):
        path = os.path.join(data_dir, fname)
        y, sr = librosa.load(path, sr=None)
        plt.subplot(n_samples, 1, idx)
        plt.plot(y)
        plt.title(f'Waveform of {fname} (labels: {df[df.fname == fname].labels.values[0]})')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_waveforms.png'))
    plt.close()


def plot_sample_spectrogram(df, data_dir, output_dir, n_samples=3):
    sample_files = df['fname'].sample(n_samples, random_state=24).values
    plt.figure(figsize=(12, 3 * n_samples))
    for idx, fname in enumerate(sample_files, 1):
        path = os.path.join(data_dir, fname)
        y, sr = librosa.load(path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.subplot(n_samples, 1, idx)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram of {fname}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_spectrograms.png'))
    plt.close()


def compute_duration_stats(df, data_dir):
    durations = []
    for fname in df['fname']:
        path = os.path.join(data_dir, fname)
        info = librosa.get_duration(filename=path)
        durations.append(info)
    durations = np.array(durations)
    stats = {
        'min_sec': float(durations.min()),
        'max_sec': float(durations.max()),
        'mean_sec': float(durations.mean()),
        'median_sec': float(np.median(durations))
    }
    return stats


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_metadata(args.csv_path)

    # Plot class distribution
    plot_class_distribution(df, args.output_dir)
    print("Saved class distribution plot to:", args.output_dir)

    # Plot sample waveforms
    plot_sample_waveform(df, args.data_dir, args.output_dir)
    print("Saved sample waveforms plot to:", args.output_dir)

    # Plot sample spectrograms
    plot_sample_spectrogram(df, args.data_dir, args.output_dir)
    print("Saved sample spectrograms plot to:", args.output_dir)

    # Compute and print duration statistics
    stats = compute_duration_stats(df, args.data_dir)
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(args.output_dir, 'duration_stats.csv'), index=False)
    print("Duration statistics saved to:", args.output_dir)
    print(stats_df.to_string(index=False))

if __name__ == '__main__':
    main()
