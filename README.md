# Audio Jingle Detector

A Python tool for detecting audio jingles in long audio files and splitting them into segments.

## Features
- Cross-correlation based jingle detection
- Automatic audio segmentation
- Support for various audio formats
- Visualization of detection results
- Configurable detection parameters

## Requirements
- Python 3.7+
- librosa
- numpy
- scipy
- soundfile
- matplotlib
- ffmpeg (optional, for better format support)

## Usage
1. Place your audio file and jingle template in the project directory
2. Update the configuration in `main()` function
3. Run: `python jingle_detector.py`

## Installation
```bash
pip install librosa numpy scipy soundfile matplotlib