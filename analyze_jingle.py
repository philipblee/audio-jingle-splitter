import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy


def analyze_jingle_segments(file_path, segment_length=2.0, hop_length=0.1):
    """
    Analyze an 8-second jingle to find the best 2-second segment for detection.

    Args:
        file_path: Path to the WAV file
        segment_length: Length of segment to extract (seconds)
        hop_length: Step size for sliding window analysis (seconds)

    Returns:
        best_start_time: Start time of the optimal segment
        analysis_results: Detailed analysis data
    """

    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    duration = len(y) / sr

    print(f"Audio loaded: {duration:.2f} seconds, {sr} Hz sample rate")

    # Calculate segment parameters
    segment_samples = int(segment_length * sr)
    hop_samples = int(hop_length * sr)

    # Initialize scoring arrays
    num_segments = int((len(y) - segment_samples) / hop_samples) + 1
    scores = {
        'spectral_centroid': [],
        'spectral_rolloff': [],
        'mfcc_variance': [],
        'rms_energy': [],
        'zero_crossing_rate': [],
        'spectral_contrast': [],
        'combined_score': [],
        'start_times': []
    }

    # Analyze each possible 2-second segment
    for i in range(num_segments):
        start_sample = i * hop_samples
        end_sample = start_sample + segment_samples

        if end_sample > len(y):
            break

        segment = y[start_sample:end_sample]
        start_time = start_sample / sr
        scores['start_times'].append(start_time)

        # Feature extraction
        # 1. Spectral centroid (brightness)
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
        scores['spectral_centroid'].append(spec_cent)

        # 2. Spectral rolloff (frequency distribution)
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))
        scores['spectral_rolloff'].append(spec_rolloff)

        # 3. MFCC variance (timbral complexity)
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_var = np.var(mfccs)
        scores['mfcc_variance'].append(mfcc_var)

        # 4. RMS energy (loudness consistency)
        rms = np.mean(librosa.feature.rms(y=segment))
        scores['rms_energy'].append(rms)

        # 5. Zero crossing rate (texture)
        zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
        scores['zero_crossing_rate'].append(zcr)

        # 6. Spectral contrast (frequency distribution distinctiveness)
        spec_contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        scores['spectral_contrast'].append(spec_contrast)

    # Normalize scores (0-1 range)
    for key in scores:
        if key != 'start_times' and key != 'combined_score':
            arr = np.array(scores[key])
            scores[key] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

    # Calculate combined score (weighted combination)
    weights = {
        'spectral_centroid': 0.2,  # Brightness/distinctiveness
        'mfcc_variance': 0.25,  # Timbral complexity (high weight)
        'rms_energy': 0.2,  # Energy consistency
        'spectral_contrast': 0.25,  # Frequency distinctiveness (high weight)
        'zero_crossing_rate': 0.1  # Texture variation
    }

    combined = np.zeros(len(scores['start_times']))
    for key, weight in weights.items():
        combined += weight * np.array(scores[key])

    scores['combined_score'] = combined

    # Find the best segment
    best_idx = np.argmax(combined)
    best_start_time = scores['start_times'][best_idx]

    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Jingle Segment Analysis - Finding Best 2-Second Window', fontsize=16)

    # Plot waveform with segment highlights
    time_axis = np.linspace(0, duration, len(y))
    axes[0, 0].plot(time_axis, y, alpha=0.7, color='blue')
    axes[0, 0].axvspan(best_start_time, best_start_time + segment_length,
                       alpha=0.3, color='red', label=f'Best segment ({best_start_time:.1f}s)')
    axes[0, 0].set_title('Audio Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot spectrogram
    D = librosa.stft(y)
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz',
                                   ax=axes[0, 1], cmap='viridis')
    axes[0, 1].axvspan(best_start_time, best_start_time + segment_length,
                       alpha=0.3, color='red')
    axes[0, 1].set_title('Spectrogram')
    plt.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')

    # Plot individual feature scores
    start_times = scores['start_times']

    axes[1, 0].plot(start_times, scores['mfcc_variance'], 'o-', label='MFCC Variance')
    axes[1, 0].plot(start_times, scores['spectral_contrast'], 's-', label='Spectral Contrast')
    axes[1, 0].axvline(best_start_time, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Key Distinctiveness Features')
    axes[1, 0].set_xlabel('Start Time (s)')
    axes[1, 0].set_ylabel('Normalized Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(start_times, scores['rms_energy'], '^-', label='RMS Energy')
    axes[1, 1].plot(start_times, scores['spectral_centroid'], 'd-', label='Spectral Centroid')
    axes[1, 1].axvline(best_start_time, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Energy & Spectral Features')
    axes[1, 1].set_xlabel('Start Time (s)')
    axes[1, 1].set_ylabel('Normalized Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot combined score
    axes[2, 0].plot(start_times, scores['combined_score'], 'o-', color='purple', linewidth=2)
    axes[2, 0].axvline(best_start_time, color='red', linestyle='--',
                       label=f'Best: {best_start_time:.1f}s')
    axes[2, 0].set_title('Combined Distinctiveness Score')
    axes[2, 0].set_xlabel('Start Time (s)')
    axes[2, 0].set_ylabel('Score')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Show segment comparison table
    axes[2, 1].axis('off')

    # Create results table
    top_3_indices = np.argsort(combined)[-3:][::-1]
    table_data = []
    for i, idx in enumerate(top_3_indices):
        rank = i + 1
        start_time = scores['start_times'][idx]
        score = combined[idx]
        table_data.append([f"#{rank}", f"{start_time:.1f}s - {start_time + 2:.1f}s", f"{score:.3f}"])

    table = axes[2, 1].table(cellText=table_data,
                             colLabels=['Rank', 'Time Range', 'Score'],
                             cellLoc='center',
                             loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[2, 1].set_title('Top 3 Segment Candidates')

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Best 2-second segment: {best_start_time:.1f}s to {best_start_time + segment_length:.1f}s")
    print(f"Combined score: {combined[best_idx]:.3f}")
    print(f"\nTop 3 candidates:")
    for i, idx in enumerate(top_3_indices):
        start_time = scores['start_times'][idx]
        score = combined[idx]
        print(f"  {i + 1}. {start_time:.1f}s - {start_time + 2:.1f}s (score: {score:.3f})")

    return best_start_time, scores


def extract_best_segment(input_file, output_file, start_time, duration=2.0):
    """
    Extract the identified best 2-second segment to a new file.
    """
    y, sr = librosa.load(input_file, sr=None)
    start_sample = int(start_time * sr)
    end_sample = int((start_time + duration) * sr)
    segment = y[start_sample:end_sample]

    # Save the segment
    import soundfile as sf
    sf.write(output_file, segment, sr)
    print(f"Best segment saved to: {output_file}")


# Example usage
if __name__ == "__main__":
    # Replace 'jingle.wav' with your file path
    input_file = "jingle.wav"

    try:
        # Analyze and find best segment
        best_start, analysis = analyze_jingle_segments(input_file)

        # Extract the best segment to a new file
        output_file = "best_jingle_segment.wav"
        extract_best_segment(input_file, output_file, best_start)

        print(f"\nUse the segment from {best_start:.1f}s to {best_start + 2:.1f}s")
        print(f"This segment has been saved as '{output_file}' for your detection algorithm.")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: pip install librosa matplotlib scipy soundfile")