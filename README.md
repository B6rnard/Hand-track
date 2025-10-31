# Hand Pose Detection with MediaPipe (unfinished)

Real-time hand skeleton tracking using MediaPipe, with WebSocket streaming and browser visualization.

## Features
- Hand pose detection using MediaPipe
- Kalman filter smoothing
- WebSocket streaming of tracking data
- Browser-based visualization
- Support for camera input and video files

## Requirements
```bash
pip install mediapipe opencv-python websockets numpy tqdm
```

## Usage

1. Start the hand tracking server:
```bash
python hand_pose_mediapipe_ws.py --source 0 --show
```

2. Open `browser_client.html` in a web browser to see the visualization

3. (Optional) Run the example WebSocket client:
```bash
python ws_client_example.py
```

## Arguments
- `--source`: Camera index or video file path (default: 0)
- `--output`: Save annotated video to file
- `--show`: Show preview window
- `--max-hands`: Maximum hands to detect (default: 2)
- `--min-det-conf`: Minimum detection confidence (default: 0.5)
- `--min-track-conf`: Minimum tracking confidence (default: 0.5)