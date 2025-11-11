# Hand Pose Detection with MediaPipe
more or less opimized

Real-time hand skeleton tracking using MediaPipe, with WebSocket streaming capabilities. The system tracks 21 hand landmarks and provides smooth tracking using Kalman filtering.

## Features
- Real-time hand pose detection using MediaPipe (21 landmarks per hand)
- Kalman filter smoothing for stable tracking
- Multi-hand tracking support (default: 2 hands)
- WebSocket streaming of tracking data
- Support for both camera input and video files
- Configurable detection and tracking confidence thresholds
- Simple wrist-based hand tracking for consistent hand IDs
- Adjustable frame resolution for performance optimization

## Requirements
```bash
pip install mediapipe opencv-python websockets numpy tqdm
```

## Usage

1. Start the hand tracking server:
```bash
python hand_pose_mediapipe_ws.py --source 0 --show
```

2. Connect to WebSocket endpoint: `ws://localhost:8765`

The server broadcasts JSON messages with the following format:
```json
{
  "timestamp": 169xxxxx,
  "frame": 123,
  "tracks": {
    "1": {"id":1, "keypoints": [[x,y],...21], "handedness": "Right"},
    "2": {...}
  }
}
```

## Command Line Arguments
- `--source`: Camera index or video file path (default: "0")
- `--output`: Save annotated video to file (optional)
- `--show`: Enable preview window
- `--max-hands`: Maximum hands to detect (default: 2)
- `--min-det-conf`: Minimum detection confidence (default: 0.5)
- `--min-track-conf`: Minimum tracking confidence (default: 0.5)
- `--track-timeout`: Seconds before dropping a track (default: 2.0)
- `--match-max-dist`: Maximum normalized distance for ID matching (default: 0.15)
- `--ws-host`: WebSocket host (default: "localhost")
- `--ws-port`: WebSocket port (default: 8765)

## Performance Notes
- The system automatically reduces input resolution to 640x480 for better performance
- Adjusts to 60 FPS capture when available
- Uses Kalman filtering for smooth landmark tracking
- Implements efficient tracking ID assignment based on wrist positions

## Status
Development in progress. Core functionality working:
- Hand detection and tracking ✓
- WebSocket streaming ✓
- Kalman filtering ✓
- Multi-hand support ✓
