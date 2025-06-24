import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import correlate
import os
from pathlib import Path
import matplotlib.pyplot as plt


class JingleDetector:
    def __init__(self, template_path, threshold=0.7, min_segment_length=30):
        """
        Initialize the jingle detector using a saved template file.

        Args:
            template_path: Path to the saved jingle template WAV file
            threshold: Correlation threshold for detection
            min_segment_length: Minimum length of content segments in seconds
        """
        print(f"Loading jingle template from: {template_path}")

        self.threshold = threshold
        self.min_segment_length = min_segment_length
        self.source_file = None

        # Load the template
        self.template, self.template_sr = librosa.load(template_path, sr=22050)

        # Normalize template
        self.template = self.template / np.max(np.abs(self.template))

        template_duration = len(self.template) / self.template_sr
        print(f"Template loaded: {template_duration:.2f}s ({len(self.template)} samples)")

        # Create multiple representations for robust matching
        self.template_mfcc = librosa.feature.mfcc(y=self.template, sr=self.template_sr, n_mfcc=13)
        self.template_chroma = librosa.feature.chroma_stft(y=self.template, sr=self.template_sr)

        # Test self-correlation
        self_corr = np.corrcoef(self.template.flatten(), self.template.flatten())[0, 1]
        print(f"Template self-correlation: {self_corr:.3f} (should be 1.0)")

        if self_corr > 0.99:
            print("✓ Template loaded successfully!")
        else:
            print("⚠ Warning: Template may have loading issues")

    def detect_jingles(self, audio_file_path, method='correlation'):
        """
        Detect jingle occurrences in the audio file.

        Args:
            audio_file_path: Path to the M4A/audio file to analyze
            method: Detection method ('correlation', 'mfcc', 'chroma', 'combined')

        Returns:
            jingle_times: List of detected jingle start times in seconds
        """
        print(f"Loading audio file: {audio_file_path}")

        # Load the full audio file
        audio, sr = librosa.load(audio_file_path, sr=22050)  # Standardize sample rate
        audio = audio / np.max(np.abs(audio))  # Normalize

        print(f"Audio loaded: {len(audio) / sr / 60:.1f} minutes, {sr} Hz")

        if method == 'correlation':
            candidates = self._cross_correlation_detection(audio, sr)
        elif method == 'mfcc':
            candidates = self._mfcc_detection(audio, sr)
        elif method == 'chroma':
            candidates = self._chroma_detection(audio, sr)
        else:  # combined - keep for compatibility but not recommended
            correlations = self._cross_correlation_detection(audio, sr)
            candidates = correlations  # Just use correlation results

        # Filter and clean up detections
        jingle_times = self._filter_detections(candidates, audio, sr)

        print(f"Found {len(jingle_times)} jingle occurrences")
        return jingle_times

    def _cross_correlation_detection(self, audio, sr):
        """Cross-correlation based detection."""
        print("Running cross-correlation detection...")

        # Compute cross-correlation
        correlation = correlate(audio, self.template, mode='valid')
        correlation = correlation / (np.linalg.norm(audio[:len(self.template)]) * np.linalg.norm(self.template))

        # Find peaks above threshold
        peaks, properties = signal.find_peaks(correlation,
                                              height=self.threshold,
                                              distance=int(sr * 5))  # Min 5 seconds between detections

        # Convert to time
        times = peaks / sr
        scores = correlation[peaks]

        return list(zip(times, scores))

    def _mfcc_detection(self, audio, sr):
        """MFCC-based detection using sliding window."""
        print("Running MFCC-based detection...")

        template_len = len(self.template)
        hop_length = int(sr * 0.1)  # 0.1 second hops for better precision
        matches = []

        total_duration = len(audio) / sr
        progress_interval = 600  # 10 minutes in seconds
        next_progress = progress_interval

        for i in range(0, len(audio) - template_len, hop_length):
            current_time = i / sr

            # Print progress every 10 minutes
            if current_time >= next_progress:
                print(f"  MFCC progress: {current_time / 60:.1f} minutes / {total_duration / 60:.1f} minutes")
                next_progress += progress_interval

            segment = audio[i:i + template_len]

            # Extract MFCC features
            segment_mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)

            # Compute similarity (normalized cross-correlation of MFCC)
            similarity = np.corrcoef(self.template_mfcc.flatten(),
                                     segment_mfcc.flatten())[0, 1]

            if not np.isnan(similarity) and similarity > self.threshold:
                time_pos = i / sr
                matches.append((time_pos, similarity))

        print(f"  MFCC detection completed: {len(matches)} matches found")
        return matches

    def _chroma_detection(self, audio, sr):
        """Chroma-based detection for harmonic content."""
        print("Running chroma-based detection...")

        template_len = len(self.template)
        hop_length = int(sr * 0.1)  # 0.1 second hops for better precision
        matches = []

        total_duration = len(audio) / sr
        progress_interval = 600  # 10 minutes in seconds
        next_progress = progress_interval

        for i in range(0, len(audio) - template_len, hop_length):
            current_time = i / sr

            # Print progress every 10 minutes
            if current_time >= next_progress:
                print(f"  Chroma progress: {current_time / 60:.1f} minutes / {total_duration / 60:.1f} minutes")
                next_progress += progress_interval

            segment = audio[i:i + template_len]

            # Extract chroma features
            segment_chroma = librosa.feature.chroma_stft(y=segment, sr=sr)

            # Compute similarity
            similarity = np.corrcoef(self.template_chroma.flatten(),
                                     segment_chroma.flatten())[0, 1]

            if not np.isnan(similarity) and similarity > self.threshold - 0.1:  # Slightly lower threshold
                time_pos = i / sr
                matches.append((time_pos, similarity))

        print(f"  Chroma detection completed: {len(matches)} matches found")
        return matches

    def _combine_detections(self, corr_matches, mfcc_matches, chroma_matches):
        """Combine multiple detection methods."""
        print("Combining detection results...")

        all_matches = {}

        # Weight the different methods
        weights = {'correlation': 0.4, 'mfcc': 0.4, 'chroma': 0.2}

        # Add correlation matches
        for time_pos, score in corr_matches:
            time_key = round(time_pos, 1)
            all_matches[time_key] = all_matches.get(time_key, 0) + weights['correlation'] * score

        # Add MFCC matches
        for time_pos, score in mfcc_matches:
            time_key = round(time_pos, 1)
            all_matches[time_key] = all_matches.get(time_key, 0) + weights['mfcc'] * score

        # Add chroma matches
        for time_pos, score in chroma_matches:
            time_key = round(time_pos, 1)
            all_matches[time_key] = all_matches.get(time_key, 0) + weights['chroma'] * score

        # Filter by combined threshold
        combined_threshold = self.threshold * 0.8  # Slightly lower for combined
        filtered_matches = [(time, score) for time, score in all_matches.items()
                            if score >= combined_threshold]

        print(f"Combined detection found {len(filtered_matches)} candidates above threshold {combined_threshold:.3f}")
        if filtered_matches:
            print("Top candidates:")
            sorted_matches = sorted(filtered_matches, key=lambda x: x[1], reverse=True)
            for i, (time, score) in enumerate(sorted_matches[:5]):
                print(f"  {i + 1}. {time:.1f}s: score {score:.3f}")

        return filtered_matches

    def _filter_detections(self, candidates, audio, sr):
        """Filter and clean up detection candidates."""
        if not candidates:
            return []

        print(f"Correlation found {len(candidates)} candidates")

        # For correlation method, use a higher score threshold
        score_threshold = 0.8  # Only keep very high scoring candidates
        high_score_candidates = [(time, score) for time, score in candidates if score >= score_threshold]

        print(f"{len(high_score_candidates)} candidates above correlation threshold {score_threshold}")

        if not high_score_candidates:
            print("No high-scoring candidates found, lowering threshold to 0.6")
            score_threshold = 0.6
            high_score_candidates = [(time, score) for time, score in candidates if score >= score_threshold]
            print(f"{len(high_score_candidates)} candidates above threshold {score_threshold}")

        # Sort by score (highest first)
        high_score_candidates.sort(key=lambda x: x[1], reverse=True)

        print("Top candidates:")
        for i, (time_pos, score) in enumerate(high_score_candidates[:5]):
            print(f"  {i + 1}. {time_pos:.1f}s: score {score:.3f}")

        # Remove detections too close to each other, keeping the highest scoring ones
        filtered = []

        for time_pos, score in high_score_candidates:
            # Check if this detection is too close to any already accepted detection
            too_close = False
            for existing_time in filtered:
                if abs(time_pos - existing_time) < 300:  # Less than 5 minutes apart
                    too_close = True
                    break

            if not too_close:
                filtered.append(time_pos)
                print(f"Added {time_pos:.1f}s (score {score:.3f})")

            # Limit to reasonable number of detections
            if len(filtered) >= 20:
                print("Reached maximum 20 detections")
                break

        return filtered

    def split_audio_by_jingles(self, audio_file_path, jingle_times, output_dir="segments"):
        """
        Split the audio file into segments based on detected jingles.

        Args:
            audio_file_path: Path to the source audio file
            jingle_times: List of jingle start times
            output_dir: Directory to save the split files
        """
        if not jingle_times:
            print("No jingles detected - cannot split audio")
            return

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        print(f"Splitting audio into segments...")
        print(f"Loading audio file for splitting: {audio_file_path}")

        try:
            # Try loading with extended timeout for large M4B files
            audio, sr = librosa.load(audio_file_path, sr=None, duration=None)  # Keep original sample rate
        except Exception as e:
            print(f"Error loading audio file with librosa: {e}")
            print("Trying alternative approach with ffmpeg...")

            # Alternative: Use ffmpeg directly to extract segments
            self._split_with_ffmpeg(audio_file_path, jingle_times, output_dir)
            return

        total_duration = len(audio) / sr
        print(f"Audio loaded successfully: {total_duration / 60:.1f} minutes")

        # Define segment boundaries
        segment_starts = [0]  # Start with beginning of file
        segment_ends = []

        for jingle_time in jingle_times:
            # End previous segment before jingle
            segment_ends.append(jingle_time)
            # Start next segment after jingle (assuming 8-second jingle)
            segment_starts.append(jingle_time + 8)

        # Add final segment end
        segment_ends.append(total_duration)

        # Extract and save segments
        saved_segments = 0
        base_name = Path(audio_file_path).stem

        for i, (start_time, end_time) in enumerate(zip(segment_starts, segment_ends)):
            # Skip segments that are too short
            segment_duration = end_time - start_time
            if segment_duration < self.min_segment_length:
                print(f"Skipping segment {i + 1}: too short ({segment_duration:.1f}s)")
                continue

            # Extract segment
            start_sample = round(start_time * sr)
            end_sample = round(end_time * sr)
            segment = audio[start_sample:end_sample]

            # Save as WAV first
            output_file = os.path.join(output_dir, f"{base_name}_segment_{i + 1:02d}.wav")
            sf.write(output_file, segment, sr)
            saved_segments += 1

            print(f"Saved segment {i + 1}: {start_time:.1f}s - {end_time:.1f}s "
                  f"({segment_duration:.1f}s) -> {output_file}")

        print(f"\nSplit complete: {saved_segments} segments saved to '{output_dir}/'")

        # Convert to MP3 if ffmpeg is available
        self._convert_to_mp3(output_dir)

    def _split_with_ffmpeg(self, audio_file_path, jingle_times, output_dir):
        """Alternative splitting method using ffmpeg directly for problematic files."""
        import subprocess

        print("Using ffmpeg for direct audio splitting...")
        base_name = Path(audio_file_path).stem

        # Get total duration using ffprobe
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                   '-of', 'csv=p=0', audio_file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            total_duration = float(result.stdout.strip())
            print(f"Total duration: {total_duration / 60:.1f} minutes")
        except Exception as e:
            print(f"Could not get duration: {e}")
            return

        # Define segment boundaries
        segment_starts = [0]
        segment_ends = []

        for jingle_time in jingle_times:
            segment_ends.append(jingle_time)
            segment_starts.append(jingle_time + 8)

        segment_ends.append(total_duration)

        # Extract segments using ffmpeg
        saved_segments = 0

        for i, (start_time, end_time) in enumerate(zip(segment_starts, segment_ends)):
            segment_duration = end_time - start_time
            if segment_duration < self.min_segment_length:
                print(f"Skipping segment {i + 1}: too short ({segment_duration:.1f}s)")
                continue

            output_file = os.path.join(output_dir, f"{base_name}_segment_{i + 1:02d}.mp3")

            cmd = [
                'ffmpeg', '-i', audio_file_path,
                '-ss', str(start_time),
                '-t', str(segment_duration),
                '-codec:a', 'mp3', '-b:a', '128k',
                '-y', output_file
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    saved_segments += 1
                    print(f"Saved segment {i + 1}: {start_time:.1f}s - {end_time:.1f}s -> {output_file}")
                else:
                    print(f"Failed to extract segment {i + 1}: {result.stderr}")
            except Exception as e:
                print(f"Error extracting segment {i + 1}: {e}")

        print(f"\nFFmpeg split complete: {saved_segments} segments saved to '{output_dir}/'")

    def _convert_to_mp3(self, output_dir):
        """Convert WAV files to MP3 using ffmpeg."""
        try:
            import subprocess

            wav_files = list(Path(output_dir).glob("*.wav"))
            if not wav_files:
                return

            print(f"Converting {len(wav_files)} files to MP3...")

            for wav_file in wav_files:
                mp3_file = wav_file.with_suffix('.mp3')
                cmd = [
                    'ffmpeg', '-i', str(wav_file),
                    '-codec:a', 'mp3', '-b:a', '128k',
                    '-y', str(mp3_file)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    wav_file.unlink()  # Delete WAV file
                    print(f"  ✓ {mp3_file.name}")
                else:
                    print(f"  ✗ Failed to convert {wav_file.name}")

        except (ImportError, FileNotFoundError):
            print("ffmpeg not found - files saved as WAV format")
            print("Install ffmpeg to get MP3 output: https://ffmpeg.org/")

    def visualize_detections(self, audio_file_path, jingle_times):
        """Create a visualization of the detected jingles."""
        # Load audio for visualization (downsampled)
        audio, sr = librosa.load(audio_file_path, sr=8000)  # Lower sample rate for plotting
        duration = len(audio) / sr
        time_axis = np.linspace(0, duration, len(audio))

        plt.figure(figsize=(15, 8))

        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, audio, alpha=0.7, color='blue', linewidth=0.5)

        # Mark detected jingles
        for jingle_time in jingle_times:
            plt.axvline(jingle_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
            plt.axvspan(jingle_time, jingle_time + 8, alpha=0.2, color='red')

        plt.title('Audio Waveform with Detected Jingles')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        # Plot detection timeline
        plt.subplot(2, 1, 2)
        segments = []
        segment_starts = [0]
        for jingle_time in jingle_times:
            segments.append(jingle_time)
            segment_starts.append(jingle_time + 8)

        # Draw segments
        for i, start in enumerate(segment_starts[:-1]):
            end = segments[i] if i < len(segments) else duration
            plt.barh(0, end - start, left=start, height=0.5,
                     alpha=0.7, color='green', label='Content' if i == 0 else '')

        # Draw jingles
        for jingle_time in jingle_times:
            plt.barh(0, 8, left=jingle_time, height=0.5,
                     alpha=0.7, color='red', label='Jingle' if jingle_time == jingle_times[0] else '')

        plt.title('Segment Timeline')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Type')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def process_audio_with_saved_template(audio_file_path, template_path="jingle_template.wav",
                                      output_dir=None, threshold=0.5, min_segment_length=1500):
    """
    Process an audio file using a previously saved jingle template.

    Args:
        audio_file_path: Path to the audio file to process
        template_path: Path to the saved jingle template
        output_dir: Output directory (default: based on audio filename)
        threshold: Detection threshold
        min_segment_length: Minimum segment length in seconds
    """
    if output_dir is None:
        base_name = Path(audio_file_path).stem
        output_dir = f"{base_name}_segments"

    print("=" * 60)
    print("PROCESSING AUDIO WITH SAVED TEMPLATE")
    print(f"Audio file: {audio_file_path}")
    print(f"Template: {template_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    try:
        # Load detector with saved template
        detector = JingleDetector.from_template_file(
            template_path=template_path,
            threshold=threshold,
            min_segment_length=min_segment_length
        )

        # Detect jingles
        print("\n=== STARTING JINGLE DETECTION ===")
        jingle_times = detector.detect_jingles(audio_file_path, method='combined')

        if jingle_times:
            print(f"\n=== DETECTION SUCCESSFUL ===")
            print(f"Found {len(jingle_times)} jingle occurrences")
            print(f"Jingle times: {[f'{t:.1f}s' for t in jingle_times]}")

            # Visualize and split
            detector.visualize_detections(audio_file_path, jingle_times)
            detector.split_audio_by_jingles(audio_file_path, jingle_times, output_dir)

            print(f"✓ Processing complete! Segments saved to: {output_dir}/")
            return True
        else:
            print("\n=== NO JINGLES DETECTED ===")
            print("Try adjusting the detection threshold.")
            return False

    except Exception as e:
        print(f"\n=== ERROR OCCURRED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("JINGLE DETECTOR - INTERNAL TEMPLATE EXTRACTION VERSION")
    print("Version: 2.0 - Template extracted directly from source file")
    print("RUNNING VERSION 60 - STREAMLINED CORRELATION ONLY")
    print("=" * 60)
    print("DEBUG: Main function started successfully")

    # Configuration

    output_directory = "lectures"
    audio_file_path = "big_history1.m4b"  # Your M4B file
    template_path = "jingle_template.wav"  # Your saved template

    # Jingle template timing (extracted directly from source)
    jingle_start_time = 5.3  # Start of jingle in seconds
    jingle_end_time = 7.3  # End of jingle in seconds

    # Detection parameters
    detection_threshold = 0.3  # Much lower threshold for testing
    min_segment_length = 1500  # 25 minutes minimum segment length

    # Debug mode - set to True to see detection scores
    debug_mode = True

    try:
        # Initialize detector (this will extract template from source file)
        print("\n=== INITIALIZING DETECTOR ===")
        detector = JingleDetector(
            template_path=template_path,
            threshold=detection_threshold,
            min_segment_length=min_segment_length
        )

        print("DEBUG: Detector initialized successfully")
        print("\n=== STARTING JINGLE DETECTION ===")

        # Detect jingles using fast correlation method only
        jingle_times = detector.detect_jingles(audio_file_path, method='correlation')

        print(f"DEBUG: Detection completed, found {len(jingle_times) if jingle_times else 0} jingles")

        # Process results
        if jingle_times:
            print(f"\n=== DETECTION SUCCESSFUL ===")
            print(f"Found {len(jingle_times)} jingle occurrences")
            print(f"Jingle times: {[f'{t:.1f}s' for t in jingle_times]}")

            # Visualize results
            detector.visualize_detections(audio_file_path, jingle_times)

            # Split the audio
            detector.split_audio_by_jingles(audio_file_path, jingle_times, output_directory)

            print(f"Segments saved to: {output_directory}/")
        else:
            print("\n=== NO JINGLES DETECTED ===")
            print("Try adjusting the detection threshold.")

    except Exception as e:
        print(f"\n=== ERROR OCCURRED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you have installed required packages: pip install librosa soundfile matplotlib scipy")


if __name__ == "__main__":
    main()