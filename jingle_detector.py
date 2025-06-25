import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import correlate
import os
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from datetime import datetime


class JingleDetector:
    """Audio jingle detection and segmentation tool."""

    def __init__(self, template_path: str, threshold: float = 0.7,
                 min_segment_length: int = 30, template_offset: float = 0.0):

        """
        Initialize the jingle detector.

        Args:
            template_path: Path to the jingle template WAV file
            threshold: Correlation threshold for detection (0.0-1.0)
            min_segment_length: Minimum length of content segments in seconds
        """
        self.template_offset = template_offset  # How far into the jingle the template starts
        self.threshold = threshold
        self.min_segment_length = min_segment_length
        self.sample_rate = 22050  # Standardized sample rate

        # Load and validate template
        self._load_template(template_path)

    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        return datetime.now().strftime("%H:%M:%S")

    def _load_template(self, template_path: str) -> None:
        """Load and prepare the jingle template."""
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        print(f"[{self._get_timestamp()}] Loading jingle template from: {template_path}")

        try:
            self.template, _ = librosa.load(template_path, sr=self.sample_rate)
            self.template = self.template / np.max(np.abs(self.template))  # Normalize

            template_duration = len(self.template) / self.sample_rate
            print(f"[{self._get_timestamp()}] Template loaded: {template_duration:.2f}s ({len(self.template)} samples)")

            # Validate template quality
            if len(self.template) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                print(f"[{self._get_timestamp()}] ⚠ Warning: Template is very short, detection may be unreliable")
            elif len(self.template) > self.sample_rate * 30:  # More than 30 seconds
                print(f"[{self._get_timestamp()}] ⚠ Warning: Template is very long, consider using a shorter segment")
            else:
                print(f"[{self._get_timestamp()}] ✓ Template loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to load template: {e}")

    def detect_jingles(self, audio_file_path: str, max_duration: Optional[float] = None) -> List[float]:
        """
        Detect jingle occurrences in the audio file using cross-correlation.

        Args:
            audio_file_path: Path to the audio file to analyze
            max_duration: Maximum duration to load in seconds (None = load all)

        Returns:
            List of detected jingle start times in seconds
        """
        if max_duration:
            print(
                f"[{self._get_timestamp()}] Loading audio file: {audio_file_path} (first {max_duration / 60:.1f} minutes)")
        else:
            print(f"[{self._get_timestamp()}] Loading audio file: {audio_file_path}")

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            # Load and normalize audio with optional duration limit
            audio, _ = librosa.load(audio_file_path, sr=self.sample_rate, duration=max_duration)
            audio = audio / np.max(np.abs(audio))

            actual_duration = len(audio) / self.sample_rate
            if max_duration and actual_duration < max_duration:
                print(
                    f"[{self._get_timestamp()}] Audio loaded: {actual_duration / 60:.1f} minutes (file shorter than requested duration)")
            elif max_duration:
                print(
                    f"[{self._get_timestamp()}] Audio loaded: {actual_duration / 60:.1f} minutes (limited from longer file)")
            else:
                print(f"[{self._get_timestamp()}] Audio loaded: {actual_duration / 60:.1f} minutes (full file)")

            # Perform cross-correlation detection
            return self._cross_correlation_detection(audio)

        except Exception as e:
            raise RuntimeError(f"Failed to process audio file: {e}")

    def _cross_correlation_detection(self, audio: np.ndarray) -> List[float]:
        """Perform cross-correlation based jingle detection."""
        print(f"[{self._get_timestamp()}] Running cross-correlation detection...")

        # Compute normalized cross-correlation with progress tracking
        correlation = correlate(audio, self.template, mode='valid')

        # Normalize correlation values with progress reporting
        template_norm = np.linalg.norm(self.template)
        total_duration = len(audio) / self.sample_rate
        progress_interval = 30 * 60  # 30 minutes in seconds
        next_progress = progress_interval

        print(f"[{self._get_timestamp()}] Normalizing correlation for {total_duration / 60:.1f} minutes of audio...")

        for i in range(len(correlation)):
            current_time = i / self.sample_rate

            # Report progress every 30 minutes
            if current_time >= next_progress:
                progress_percent = (current_time / total_duration) * 100
                print(
                    f"[{self._get_timestamp()}]   Progress: {current_time / 60:.1f} minutes / {total_duration / 60:.1f} minutes ({progress_percent:.1f}%)")
                next_progress += progress_interval

            audio_segment_norm = np.linalg.norm(audio[i:i + len(self.template)])
            if audio_segment_norm > 0:
                correlation[i] = correlation[i] / (template_norm * audio_segment_norm)

        print(f"[{self._get_timestamp()}] Correlation analysis complete, finding peaks...")

        # Find peaks above threshold
        min_distance = int(self.sample_rate * 5)  # Minimum 5 seconds between detections
        peaks, properties = signal.find_peaks(
            correlation,
            height=self.threshold,
            distance=min_distance
        )

        # Convert sample indices to time and get scores
        jingle_times = peaks / self.sample_rate
        scores = correlation[peaks]

        # Filter and sort results
        return self._filter_detections(list(zip(jingle_times, scores)))

    def _filter_detections(self, candidates: List[Tuple[float, float]]) -> List[float]:
        """Filter and clean up detection candidates."""
        if not candidates:
            print(f"[{self._get_timestamp()}] No candidates found")
            return []

        print(f"[{self._get_timestamp()}] Found {len(candidates)} initial candidates")

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Show top candidates
        print(f"[{self._get_timestamp()}] Top candidates:")
        for i, (time_pos, score) in enumerate(candidates[:5]):
            print(f"[{self._get_timestamp()}]   {i + 1}. {time_pos:.1f}s: score {score:.3f}")

        # Remove candidates too close to each other (keep highest scoring)
        filtered_times = []
        min_separation = 60  # Minimum 1 minute between jingles

        for time_pos, score in candidates:
            # Check if too close to existing detections
            if not any(abs(time_pos - existing) < min_separation for existing in filtered_times):
                filtered_times.append(time_pos)
                print(f"[{self._get_timestamp()}] Added detection at {time_pos:.1f}s (score: {score:.3f})")

                # Limit total detections
                if len(filtered_times) >= 50:
                    print(f"[{self._get_timestamp()}] Reached maximum detection limit (50)")
                    break

        print(f"[{self._get_timestamp()}] Final result: {len(filtered_times)} jingle detections")
        return sorted(filtered_times)

    def split_audio_by_jingles(self, audio_file_path: str, jingle_times: List[float],
                               output_dir: str = "segments", jingle_duration: float = 8.0,
                               start_number: int = 1, max_duration: Optional[float] = None,
                               base_filename: Optional[str] = None) -> None:
        """      
        Split the audio file into segments based on detected jingles.
        
        Args:
            audio_file_path: Path to the source audio file
            jingle_times: List of jingle start times in seconds
            output_dir: Directory to save the split files
            jingle_duration: Duration of each jingle in seconds
            start_number: Starting number for the first segment (default: 1)
            max_duration: Maximum duration processed in seconds (None = full file)
            base_filename: Base name for output files (None = use audio filename)
        """
        if not jingle_times:
            print(f"[{self._get_timestamp()}] No jingles detected - cannot split audio")
            return

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Try ffmpeg first (handles more formats and is more memory efficient)
        if self._has_ffmpeg():
            self._split_with_ffmpeg(
                audio_file_path, jingle_times, output_dir, jingle_duration, start_number, max_duration, base_filename)
        else:
            self._split_with_librosa(audio_file_path, jingle_times, output_dir, jingle_duration, start_number,
                                     max_duration, base_filename)

    def _has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _get_audio_duration(self, audio_file_path: str) -> float:
        """Get audio duration using ffprobe."""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                   '-of', 'csv=p=0', audio_file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"[{self._get_timestamp()}] Could not get duration with ffprobe: {e}")
            # Fallback to librosa
            try:
                duration = librosa.get_duration(path=audio_file_path)
                return duration
            except Exception as e2:
                print(f"[{self._get_timestamp()}] Could not get duration with librosa: {e2}")
                return 0

    def _split_with_ffmpeg(self, audio_file_path: str, jingle_times: List[float],
                           output_dir: str, jingle_duration: float, start_number: int,
                           max_duration: Optional[float] = None,
                           base_filename: Optional[str] = None) -> None:

        """Split audio using ffmpeg (memory efficient for large files)."""
        print(f"[{self._get_timestamp()}] DEBUG: max_duration parameter = {max_duration}")
        print(f"[{self._get_timestamp()}] Using ffmpeg for audio splitting...")

        # Use limited duration if specified, otherwise get full file duration
        if max_duration:
            total_duration = max_duration
            print(f"[{self._get_timestamp()}] Using limited duration: {total_duration / 60:.1f} minutes")
        else:
            total_duration = self._get_audio_duration(audio_file_path)
            if total_duration == 0:
                print(f"[{self._get_timestamp()}] Could not determine audio duration")
                return
            print(f"[{self._get_timestamp()}] Total duration: {total_duration / 60:.1f} minutes")

        # Calculate segment boundaries
        segments = self._calculate_segments(jingle_times, total_duration, jingle_duration)

        base_name = Path(audio_file_path).stem  # Always use audio filename
        saved_segment_number = start_number  # NEW: Counter for saved segments only

        for i, (start_time, end_time) in enumerate(segments):
            duration = end_time - start_time

            if duration < self.min_segment_length:
                print(f"[{self._get_timestamp()}] Skipping segment: too short ({duration:.1f}s)")
                continue  # Don't increment saved_segment_number

            segment_word = base_filename if base_filename else "segment"
            output_file = Path(output_dir) / f"{base_name}_{segment_word}_{saved_segment_number:02d}.mp3"

            cmd = [
                'ffmpeg', '-i', audio_file_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-codec:a', 'mp3', '-b:a', '128k',
                '-y', str(output_file)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(
                    f"[{self._get_timestamp()}] Saved segment {saved_segment_number}: {start_time:.1f}s - {end_time:.1f}s "
                    f"({duration:.1f}s) -> {output_file.name}")
                saved_segment_number += 1  # Only increment after successful save
            except subprocess.CalledProcessError as e:
                print(f"[{self._get_timestamp()}] Failed to extract segment {saved_segment_number}: {e.stderr}")
                saved_segment_number += 1  # Still increment on failure to avoid duplicates

        print(
            f"[{self._get_timestamp()}] Split complete: {saved_segment_number - start_number} segments saved to '{output_dir}/'")

    def _split_with_librosa(self, audio_file_path: str, jingle_times: List[float],
                            output_dir: str, jingle_duration: float, start_number: int,
                            max_duration: Optional[float] = None,
                            base_filename: Optional[str] = None) -> None:

        """Split audio using librosa (fallback method)."""
        print(f"[{self._get_timestamp()}] Using librosa for audio splitting...")

        try:

            # Load audio with optional duration limit
            audio, sr = librosa.load(audio_file_path, sr=None, duration=max_duration)

            # Use the actual loaded duration (respects max_duration)
            total_duration = len(audio) / sr

            if max_duration and total_duration < max_duration:
                print(
                    f"[{self._get_timestamp()}] Audio loaded: {total_duration / 60:.1f} minutes (file shorter than limit)")
            elif max_duration:
                print(
                    f"[{self._get_timestamp()}] Audio loaded: {total_duration / 60:.1f} minutes (limited from longer file)")
            else:
                print(f"[{self._get_timestamp()}] Audio loaded: {total_duration / 60:.1f} minutes (full file)")

            # Calculate segments
            segments = self._calculate_segments(jingle_times, total_duration, jingle_duration)
            base_name = Path(audio_file_path).stem  # Always use audio filename
            saved_segment_number = start_number  # NEW: Counter for saved segments only

            for i, (start_time, end_time) in enumerate(segments):
                duration = end_time - start_time

                if duration < self.min_segment_length:
                    print(f"[{self._get_timestamp()}] Skipping segment: too short ({duration:.1f}s)")
                    continue  # Don't increment saved_segment_number

                # Extract segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment = audio[start_sample:end_sample]

                # Save as WAV using saved_segment_number
                segment_word = base_filename if base_filename else "segment"
                output_file = Path(output_dir) / f"{base_name}_{segment_word}_{saved_segment_number:02d}.wav"

                sf.write(output_file, segment, sr)

                print(
                    f"[{self._get_timestamp()}] Saved segment {saved_segment_number}: {start_time:.1f}s - {end_time:.1f}s "
                    f"({duration:.1f}s) -> {output_file.name}")

                saved_segment_number += 1  # Only increment after successful save

            print(
                f"[{self._get_timestamp()}] Split complete: {saved_segment_number - start_number} segments saved to '{output_dir}/'")

            print(f"[{self._get_timestamp()}] Note: Files saved as WAV. Install ffmpeg for MP3 output.")

        except Exception as e:
            raise RuntimeError(f"Failed to split audio with librosa: {e}")

    def _calculate_segments(self, jingle_times: List[float], total_duration: float,
                            jingle_duration: float) -> List[Tuple[float, float]]:
        """Calculate segment start and end times."""
        # Adjust jingle start times to account for template offset
        actual_jingle_starts = [max(0, t - self.template_offset) for t in jingle_times]

        segments = []

        if actual_jingle_starts:
            # First segment: from start to just before first jingle
            if actual_jingle_starts[0] > 0:
                segments.append((0, actual_jingle_starts[0]))

            # Subsequent segments: each starts WITH its jingle
            for i in range(len(actual_jingle_starts)):
                segment_start = actual_jingle_starts[i]  # Start WITH the jingle

                # End just before the next jingle (or at file end)
                if i < len(actual_jingle_starts) - 1:
                    segment_end = actual_jingle_starts[i + 1]
                else:
                    segment_end = total_duration

                segments.append((segment_start, segment_end))
        else:
            # No jingles found, return entire file as one segment
            segments.append((0, total_duration))

        return segments
    def visualize_detections(self, audio_file_path: str, jingle_times: List[float]) -> None:
        """Create a visualization of the detected jingles."""
        try:
            print(f"[{self._get_timestamp()}] Creating visualization...")
            # Load audio for visualization (downsampled for performance)
            audio, sr = librosa.load(audio_file_path, sr=8000, duration=1800)  # Max 30 minutes
            duration = len(audio) / sr
            time_axis = np.linspace(0, duration, len(audio))

            plt.figure(figsize=(15, 8))

            # Plot waveform
            plt.subplot(2, 1, 1)
            plt.plot(time_axis, audio, alpha=0.7, color='blue', linewidth=0.5)

            # Mark detected jingles
            for jingle_time in jingle_times:
                if jingle_time <= duration:  # Only show jingles within loaded duration
                    plt.axvline(jingle_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
                    plt.axvspan(jingle_time, min(jingle_time + 8, duration),
                                alpha=0.2, color='red')

            plt.title('Audio Waveform with Detected Jingles')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)

            # Plot timeline
            plt.subplot(2, 1, 2)
            self._plot_timeline(jingle_times, duration)

            plt.tight_layout()
            print(f"[{self._get_timestamp()}] Visualization ready - displaying plot...")
            plt.show()

        except Exception as e:
            print(f"[{self._get_timestamp()}] Visualization failed: {e}")

    def _plot_timeline(self, jingle_times: List[float], duration: float) -> None:
        """Plot the segment timeline."""
        # Calculate segments for visualization
        segments = self._calculate_segments(jingle_times, duration, 8.0)

        # Plot content segments
        for i, (start, end) in enumerate(segments):
            plt.barh(0, end - start, left=start, height=0.5,
                     alpha=0.7, color='green',
                     label='Content' if i == 0 else '')

        # Plot jingles
        for i, jingle_time in enumerate(jingle_times):
            if jingle_time <= duration:
                plt.barh(0, min(8, duration - jingle_time), left=jingle_time, height=0.5,
                         alpha=0.7, color='red',
                         label='Jingle' if i == 0 else '')

        plt.title('Segment Timeline')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Type')
        plt.legend()
        plt.grid(True, alpha=0.3)


def process_audio_file(audio_file_path: str, template_path: str,
                       output_dir: Optional[str] = None,
                       threshold: float = 0.5,
                       min_segment_length: int = 1500,
                       start_number: int = 1,
                       max_duration: Optional[float] = None,
                       base_filename: Optional[str] = None) -> bool:  # Add this line
    """
    Process an audio file to detect and split by jingles.

    Args:
        audio_file_path: Path to the audio file to process
        template_path: Path to the jingle template file
        output_dir: Output directory (default: based on audio filename)
        threshold: Detection threshold (0.0-1.0)
        min_segment_length: Minimum segment length in seconds
        start_number: Starting number for the first segment (default: 1)
        max_duration: Maximum duration to process in seconds (None = process all)

    Returns:
        True if successful, False otherwise
    """
    if output_dir is None:
        base_name = Path(audio_file_path).stem
        output_dir = f"{base_name}_segments"

    timestamp = datetime.now().strftime("%H:%M:%S")
    print("=" * 80)
    print(f"[{timestamp}] JINGLE DETECTOR - AUDIO PROCESSING")
    print(f"[{timestamp}] Audio file: {audio_file_path}")
    if max_duration:
        print(f"[{timestamp}] Duration limit: {max_duration / 60:.1f} minutes")
    print(f"[{timestamp}] Template: {template_path}")
    print(f"[{timestamp}] Output: {output_dir}")
    print(f"[{timestamp}] Threshold: {threshold}")
    print(f"[{timestamp}] Min segment: {min_segment_length}s")
    print(f"[{timestamp}] Start number: {start_number}")
    print(f"[{timestamp}] Base filename: {base_filename}")  # ADD THIS LINE
    print("=" * 80)

    try:
        # Initialize detector
        detector = JingleDetector(
            template_path=template_path,
            threshold=threshold,
            min_segment_length=min_segment_length,
            template_offset=5.3  # Template starts 5.3 seconds into the 8-second jingle
        )

        # Detect jingles with optional duration limit
        print(f"\n[{detector._get_timestamp()}] === STARTING JINGLE DETECTION ===")
        jingle_times = detector.detect_jingles(audio_file_path, max_duration=max_duration)

        if jingle_times:
            print(f"\n[{detector._get_timestamp()}] === DETECTION SUCCESSFUL ===")
            print(f"[{detector._get_timestamp()}] Found {len(jingle_times)} jingle occurrences")
            print(f"[{detector._get_timestamp()}] Jingle times: {[f'{t:.1f}s' for t in jingle_times]}")

            # Visualize and split
            # detector.visualize_detections(audio_file_path, jingle_times)

            detector.split_audio_by_jingles(
                audio_file_path,
                jingle_times,
                output_dir,
                jingle_duration=8.0,
                start_number=start_number,
                max_duration=max_duration,
                base_filename=base_filename)

            final_timestamp = detector._get_timestamp()
            print(f"[{final_timestamp}] ✓ Processing complete! Segments saved to: {output_dir}/")
            return True
        else:
            print(f"\n[{detector._get_timestamp()}] === NO JINGLES DETECTED ===")
            print(f"[{detector._get_timestamp()}] Try adjusting the detection threshold or check your template file.")
            return False

    except Exception as e:
        error_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{error_timestamp}] === ERROR OCCURRED ===")
        print(f"[{error_timestamp}] Error: {e}")
        return False


def main():
    """Main function with example usage."""
    start_time = datetime.now()
    timestamp = start_time.strftime("%H:%M:%S")

    print("=" * 80)
    print(f"[{timestamp}] JINGLE DETECTOR - TIMESTAMPED VERSION")
    print(f"[{timestamp}] Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Configuration
    audio_file_path = "file1.m4b"
    template_path = "jingle_template.wav" # reference jingle
    output_directory = "test_12min"
    base_filename = "lecture"  # NEW: Configure the segment naming
    detection_threshold = 0.7
    min_segment_length = 300  # 5 minutes, or skip file as too small
    start_number = 1
    test_duration = 12 * 60  # 70 minutes in seconds, Set to None to process the entire file: test_duration = None

    # Check if files exist before processing
    if not os.path.exists(audio_file_path):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Audio file not found: {audio_file_path}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Please check the file path and try again.")
        return

    if not os.path.exists(template_path):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Template file not found: {template_path}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Please check the template path and try again.")
        return

    success = process_audio_file(
        audio_file_path=audio_file_path,
        template_path=template_path,
        output_dir=output_directory,
        threshold=detection_threshold,
        min_segment_length=min_segment_length,
        start_number=start_number,
        max_duration=test_duration,
        base_filename=base_filename  # Pass the configuration variable
    )

    end_time = datetime.now()
    duration = end_time - start_time

    if not success:
        print(f"\n[{end_time.strftime('%H:%M:%S')}] Troubleshooting tips:")
        print(f"[{end_time.strftime('%H:%M:%S')}] 1. Check that your audio and template files exist")
        print(f"[{end_time.strftime('%H:%M:%S')}] 2. Try adjusting the threshold (lower = more sensitive)")
        print(f"[{end_time.strftime('%H:%M:%S')}] 3. Ensure your template is a clear, isolated jingle")
        print(f"[{end_time.strftime('%H:%M:%S')}] 4. Install ffmpeg for better format support")

    print(f"\n[{end_time.strftime('%H:%M:%S')}] === PROCESSING COMPLETE ===")
    print(f"[{end_time.strftime('%H:%M:%S')}] Total processing time: {duration}")


if __name__ == "__main__":
    main()