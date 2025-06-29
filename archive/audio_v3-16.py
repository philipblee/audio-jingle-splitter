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
from dataclasses import dataclass


@dataclass
class JingleDetectorConfig:
    """Configuration class for JingleDetector with sensible defaults."""

    # Audio processing settings
    sample_rate: int = 22050
    min_segment_length: int = 30  # seconds

    # Detection settings
    threshold: float = 0.7
    min_jingle_separation: int = 60  # seconds between jingles
    min_peak_distance: int = 5  # seconds between detection peaks
    max_detections: int = 50

    # Progress reporting
    progress_interval: int = 1800  # 30 minutes in seconds

    # Audio splitting settings
    default_jingle_duration: float = 8.0  # seconds

    # Template validation thresholds
    min_template_duration: float = 0.5  # seconds
    max_template_duration: float = 30.0  # seconds

    # Visualization settings
    viz_sample_rate: int = 8000
    viz_max_duration: int = 1800  # 30 minutes max for visualization

    # Output settings
    mp3_bitrate: str = "128k"

    @classmethod
    def create_for_lectures(cls, threshold: float = 0.3, min_segment_length: int = 1500) -> 'JingleDetectorConfig':
        """Create configuration optimized for lecture detection."""
        config = cls()
        config.threshold = threshold
        config.min_segment_length = min_segment_length
        config.min_jingle_separation = 300  # 5 minutes for lectures
        return config

    @classmethod
    def create_for_music(cls, threshold: float = 0.6, min_segment_length: int = 180) -> 'JingleDetectorConfig':
        """Create configuration optimized for music/podcast detection."""
        config = cls()
        config.threshold = threshold
        config.min_segment_length = min_segment_length
        config.min_jingle_separation = 30  # 30 seconds for music
        return config

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")

        if self.min_segment_length < 1:
            raise ValueError(f"Minimum segment length must be at least 1 second, got {self.min_segment_length}")

        if self.sample_rate < 8000:
            raise ValueError(f"Sample rate too low, minimum 8000 Hz, got {self.sample_rate}")

        if self.min_template_duration > self.max_template_duration:
            raise ValueError("Minimum template duration cannot be greater than maximum")


class JingleDetector:
    """Audio jingle detection and segmentation tool."""

    def __init__(self, template_path: str, config: Optional[JingleDetectorConfig] = None):
        """
        Initialize the jingle detector.

        Args:
            template_path: Path to the jingle template WAV file
            config: Configuration object (uses defaults if None)
        """
        self.config = config or JingleDetectorConfig()
        self.config.validate()

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
            self.template, _ = librosa.load(template_path, sr=self.config.sample_rate)
            self.template = self.template / np.max(np.abs(self.template))  # Normalize

            template_duration = len(self.template) / self.config.sample_rate
            print(f"[{self._get_timestamp()}] Template loaded: {template_duration:.2f}s ({len(self.template)} samples)")

            # Validate template quality
            if template_duration < self.config.min_template_duration:
                print(
                    f"[{self._get_timestamp()}] ⚠ Warning: Template is very short ({template_duration:.2f}s), detection may be unreliable")
            elif template_duration > self.config.max_template_duration:
                print(
                    f"[{self._get_timestamp()}] ⚠ Warning: Template is very long ({template_duration:.2f}s), consider using a shorter segment")
            else:
                print(f"[{self._get_timestamp()}] ✓ Template loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to load template: {e}")

    def detect_jingles(self, audio_file_path: str) -> List[float]:
        """
        Detect jingle occurrences in the audio file using cross-correlation.

        Args:
            audio_file_path: Path to the audio file to analyze

        Returns:
            List of detected jingle start times in seconds
        """
        print(f"[{self._get_timestamp()}] Loading audio file: {audio_file_path}")

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            # Load and normalize audio
            audio, _ = librosa.load(audio_file_path, sr=self.config.sample_rate)
            audio = audio / np.max(np.abs(audio))

            print(f"[{self._get_timestamp()}] Audio loaded: {len(audio) / self.config.sample_rate / 60:.1f} minutes")

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
        total_duration = len(audio) / self.config.sample_rate
        progress_interval = self.config.progress_interval
        next_progress = progress_interval

        print(f"[{self._get_timestamp()}] Normalizing correlation for {total_duration / 60:.1f} minutes of audio...")

        for i in range(len(correlation)):
            current_time = i / self.config.sample_rate

            # Report progress every interval
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
        min_distance = int(self.config.sample_rate * self.config.min_peak_distance)
        peaks, properties = signal.find_peaks(
            correlation,
            height=self.config.threshold,
            distance=min_distance
        )

        # Convert sample indices to time and get scores
        jingle_times = peaks / self.config.sample_rate
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
        min_separation = self.config.min_jingle_separation

        for time_pos, score in candidates:
            # Check if too close to existing detections
            if not any(abs(time_pos - existing) < min_separation for existing in filtered_times):
                filtered_times.append(time_pos)
                print(f"[{self._get_timestamp()}] Added detection at {time_pos:.1f}s (score: {score:.3f})")

                # Limit total detections
                if len(filtered_times) >= self.config.max_detections:
                    print(f"[{self._get_timestamp()}] Reached maximum detection limit ({self.config.max_detections})")
                    break

        print(f"[{self._get_timestamp()}] Final result: {len(filtered_times)} jingle detections")
        return sorted(filtered_times)

    def split_audio_by_jingles(self, audio_file_path: str, jingle_times: List[float],
                               output_dir: str = "segments",
                               jingle_duration: Optional[float] = None,
                               start_number: int = 1) -> None:
        """
        Split the audio file into segments based on detected jingles.

        Args:
            audio_file_path: Path to the source audio file
            jingle_times: List of jingle start times in seconds
            output_dir: Directory to save the split files
            jingle_duration: Duration of each jingle in seconds (uses config default if None)
            start_number: Starting number for the first segment (default: 1)
        """
        if jingle_duration is None:
            jingle_duration = self.config.default_jingle_duration
        if not jingle_times:
            print(f"[{self._get_timestamp()}] No jingles detected - cannot split audio")
            return

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Try ffmpeg first (handles more formats and is more memory efficient)
        if self._has_ffmpeg():
            self._split_with_ffmpeg(audio_file_path, jingle_times, output_dir, jingle_duration, start_number)
        else:
            self._split_with_librosa(audio_file_path, jingle_times, output_dir, jingle_duration, start_number)

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
                           output_dir: str, jingle_duration: float, start_number: int) -> None:
        """Split audio using ffmpeg (memory efficient for large files)."""
        print(f"[{self._get_timestamp()}] Using ffmpeg for audio splitting...")

        total_duration = self._get_audio_duration(audio_file_path)
        if total_duration == 0:
            print(f"[{self._get_timestamp()}] Could not determine audio duration")
            return

        print(f"[{self._get_timestamp()}] Total duration: {total_duration / 60:.1f} minutes")

        # Calculate segment boundaries
        segments = self._calculate_segments(jingle_times, total_duration, jingle_duration)

        # Extract segments
        base_name = Path(audio_file_path).stem
        saved_count = 0

        for i, (start_time, end_time) in enumerate(segments):
            duration = end_time - start_time
            segment_number = start_number + i

            if duration < self.config.min_segment_length:
                print(f"[{self._get_timestamp()}] Skipping segment {segment_number}: too short ({duration:.1f}s)")
                continue

            output_file = Path(output_dir) / f"{base_name}_segment_{segment_number:02d}.mp3"

            cmd = [
                'ffmpeg', '-i', audio_file_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-codec:a', 'mp3', '-b:a', self.config.mp3_bitrate,
                '-y', str(output_file)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                saved_count += 1
                print(f"[{self._get_timestamp()}] Saved segment {segment_number}: {start_time:.1f}s - {end_time:.1f}s "
                      f"({duration:.1f}s) -> {output_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"[{self._get_timestamp()}] Failed to extract segment {segment_number}: {e.stderr}")

        print(f"[{self._get_timestamp()}] Split complete: {saved_count} segments saved to '{output_dir}/'")

    def _split_with_librosa(self, audio_file_path: str, jingle_times: List[float],
                            output_dir: str, jingle_duration: float, start_number: int) -> None:
        """Split audio using librosa (fallback method)."""
        print(f"[{self._get_timestamp()}] Using librosa for audio splitting...")

        try:
            audio, sr = librosa.load(audio_file_path, sr=None)
            total_duration = len(audio) / sr
            print(f"[{self._get_timestamp()}] Audio loaded: {total_duration / 60:.1f} minutes")

            # Calculate segments
            segments = self._calculate_segments(jingle_times, total_duration, jingle_duration)

            # Extract and save segments
            base_name = Path(audio_file_path).stem
            saved_count = 0

            for i, (start_time, end_time) in enumerate(segments):
                duration = end_time - start_time
                segment_number = start_number + i

                if duration < self.config.min_segment_length:
                    print(f"[{self._get_timestamp()}] Skipping segment {segment_number}: too short ({duration:.1f}s)")
                    continue

                # Extract segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment = audio[start_sample:end_sample]

                # Save as WAV
                output_file = Path(output_dir) / f"{base_name}_segment_{segment_number:02d}.wav"
                sf.write(output_file, segment, sr)
                saved_count += 1

                print(f"[{self._get_timestamp()}] Saved segment {segment_number}: {start_time:.1f}s - {end_time:.1f}s "
                      f"({duration:.1f}s) -> {output_file.name}")

            print(f"[{self._get_timestamp()}] Split complete: {saved_count} segments saved to '{output_dir}/'")
            print(f"[{self._get_timestamp()}] Note: Files saved as WAV. Install ffmpeg for MP3 output.")

        except Exception as e:
            raise RuntimeError(f"Failed to split audio with librosa: {e}")

    def _calculate_segments(self, jingle_times: List[float], total_duration: float,
                            jingle_duration: float) -> List[Tuple[float, float]]:
        """Calculate segment start and end times."""
        segments = []

        # First segment: from start to first jingle
        if jingle_times:
            if jingle_times[0] > 0:
                segments.append((0, jingle_times[0]))

            # Middle segments: between jingles
            for i in range(len(jingle_times) - 1):
                start = jingle_times[i] + jingle_duration
                end = jingle_times[i + 1]
                segments.append((start, end))

            # Last segment: from last jingle to end
            last_segment_start = jingle_times[-1] + jingle_duration
            if last_segment_start < total_duration:
                segments.append((last_segment_start, total_duration))
        else:
            # No jingles found, return entire file as one segment
            segments.append((0, total_duration))

        return segments

    def visualize_detections(self, audio_file_path: str, jingle_times: List[float]) -> None:
        """Create a visualization of the detected jingles."""
        try:
            print(f"[{self._get_timestamp()}] Creating visualization...")
            # Load audio for visualization (downsampled for performance)
            audio, sr = librosa.load(audio_file_path, sr=self.config.viz_sample_rate,
                                     duration=self.config.viz_max_duration)
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
        segments = self._calculate_segments(jingle_times, duration, self.config.default_jingle_duration)

        # Plot content segments
        for i, (start, end) in enumerate(segments):
            plt.barh(0, end - start, left=start, height=0.5,
                     alpha=0.7, color='green',
                     label='Content' if i == 0 else '')

        # Plot jingles
        for i, jingle_time in enumerate(jingle_times):
            if jingle_time <= duration:
                plt.barh(0, min(self.config.default_jingle_duration, duration - jingle_time),
                         left=jingle_time, height=0.5,
                         alpha=0.7, color='red',
                         label='Jingle' if i == 0 else '')

        plt.title('Segment Timeline')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Type')
        plt.legend()
        plt.grid(True, alpha=0.3)


def process_audio_file(audio_file_path: str, template_path: str,
                       output_dir: Optional[str] = None,
                       config: Optional[JingleDetectorConfig] = None,
                       start_number: int = 1) -> bool:
    """
    Process an audio file to detect and split by jingles.

    Args:
        audio_file_path: Path to the audio file to process
        template_path: Path to the jingle template file
        output_dir: Output directory (default: based on audio filename)
        config: JingleDetectorConfig object (uses defaults if None)
        start_number: Starting number for the first segment (default: 1)

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = JingleDetectorConfig()

    if output_dir is None:
        base_name = Path(audio_file_path).stem
        output_dir = f"{base_name}_segments"

    timestamp = datetime.now().strftime("%H:%M:%S")
    print("=" * 80)
    print(f"[{timestamp}] JINGLE DETECTOR - AUDIO PROCESSING")
    print(f"[{timestamp}] Audio file: {audio_file_path}")
    print(f"[{timestamp}] Template: {template_path}")
    print(f"[{timestamp}] Output: {output_dir}")
    print(f"[{timestamp}] Threshold: {config.threshold}")
    print(f"[{timestamp}] Min segment: {config.min_segment_length}s")
    print("=" * 80)

    try:
        # Initialize detector
        detector = JingleDetector(
            template_path=template_path,
            config=config
        )

        # Detect jingles
        print(f"\n[{detector._get_timestamp()}] === STARTING JINGLE DETECTION ===")
        jingle_times = detector.detect_jingles(audio_file_path)

        if jingle_times:
            print(f"\n[{detector._get_timestamp()}] === DETECTION SUCCESSFUL ===")
            print(f"[{detector._get_timestamp()}] Found {len(jingle_times)} jingle occurrences")
            print(f"[{detector._get_timestamp()}] Jingle times: {[f'{t:.1f}s' for t in jingle_times]}")

            # Visualize and split
            detector.visualize_detections(audio_file_path, jingle_times)
            detector.split_audio_by_jingles(audio_file_path, jingle_times, output_dir, start_number=start_number)

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
    print(f"[{timestamp}] JINGLE DETECTOR - CONFIGURATION MANAGED VERSION")
    print(f"[{timestamp}] Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # File paths
    audio_file_path = "file2.m4b"
    template_path = "jingle_template.wav"
    output_directory = "lectures"
    start_number = 20

    # Create configuration for lecture processing
    config = JingleDetectorConfig.create_for_lectures(
        threshold=0.3,
        min_segment_length=1500  # 25 minutes
    )

    # You can also customize specific settings
    config.progress_interval = 1800  # Report progress every 30 minutes
    config.max_detections = 100  # Allow more detections for long lectures

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Configuration:")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Threshold: {config.threshold}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Min segment length: {config.min_segment_length}s")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Jingle separation: {config.min_jingle_separation}s")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Sample rate: {config.sample_rate} Hz")

    # Check if files exist before processing
    if not os.path.exists(audio_file_path):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Audio file not found: {audio_file_path}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Please check the file path and try again.")
        return

    if not os.path.exists(template_path):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Template file not found: {template_path}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Please check the template path and try again.")
        return

    # Process the audio file
    success = process_audio_file(
        audio_file_path=audio_file_path,
        template_path=template_path,
        output_dir=output_directory,
        config=config,
        start_number=start_number
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