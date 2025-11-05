#!/usr/bin/env python3
"""
hand_pose_mediapipe_ws.py

MediaPipe-based real-time hand skeleton (21 landmarks) -> Kalman smoothing -> WebSocket broadcaster.

Usage:
    python hand_pose_mediapipe_ws.py --source 0 --show
    python hand_pose_mediapipe_ws.py --source input.mp4 --output out.mp4

WebSocket: ws://localhost:8765
Message (JSON) per frame:
{
  "timestamp": 169xxx,
  "frame": 123,
  "tracks": {
    "1": {"id":1, "keypoints": [[x,y],...21], "handedness": "Right"},
    "2": {...}
  }
}
"""
import argparse
import asyncio
import json
import math
import threading
import time
from collections import deque
from queue import Queue

import cv2
import mediapipe as mp
import numpy as np
import websockets
from tqdm import tqdm

# ---------- skeleton connections (MediaPipe 21 landmarks) ----------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ---------- Simple 2D Kalman filter (constant velocity) ----------
class Kalman2D:
    def __init__(self, x=0.0, y=0.0, dt=1/30.0, process_var=1e-4, meas_var=1e-2):
        self.dt = float(dt) if dt > 0 else 1/30.0
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)  # [x,y,vx,vy]
        self.P = np.eye(4) * 1e-2
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * meas_var

    def predict(self, dt=None):
        if dt is not None and dt > 0:
            self.dt = float(dt)
            self.F[0,2] = self.dt
            self.F[1,3] = self.dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, meas, meas_conf=1.0):
        if meas is None:
            return self.x[:2].copy()
        z = np.array(meas, dtype=float)
        conf = float(max(1e-6, meas_conf)) if meas_conf is not None else 1.0
        R = self.R / conf
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S + 1e-9*np.eye(2))
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x[:2].copy()

# ---------- WebSocket broadcaster ----------
class WebSocketBroadcaster:
    def __init__(self, queue: Queue, host='localhost', port=8765):
        self.queue = queue
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.server = None
        self._stop = threading.Event()

    async def _handler(self, websocket, path):
        print("WS client connected:", websocket.remote_address)
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print("WS client disconnected:", websocket.remote_address)

    async def _broadcaster(self):
        while not self._stop.is_set():
            try:
                while not self.queue.empty():
                    msg = self.queue.get_nowait()
                    if not self.clients:
                        continue
                    payload = json.dumps(msg)
                    await asyncio.gather(*(c.send(payload) for c in list(self.clients)), return_exceptions=True)
                await asyncio.sleep(0.01)
            except Exception as e:
                print("WS broadcast error:", e)
                await asyncio.sleep(0.05)

    def start(self):
        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.server = self.loop.run_until_complete(
                websockets.serve(self._handler, self.host, self.port)
            )
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            try:
                self.loop.run_until_complete(self._broadcaster())
            finally:
                self.loop.run_until_complete(self.shutdown_loop())
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    async def shutdown_loop(self):
        tasks = [t for t in asyncio.all_tasks(loop=self.loop) if t is not asyncio.current_task(loop=self.loop)]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.loop.stop()

    def stop(self):
        self._stop.set()
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.shutdown_loop(), self.loop)
        if getattr(self, "thread", None):
            self.thread.join(timeout=1.0)

# ---------- Utility: drawing ----------
def draw_skeleton(frame, kps_norm, color=(0,255,0), id_text=None):
    h,w = frame.shape[:2]
    kps = np.array(kps_norm)
    pts = (kps * np.array([w, h])).astype(int)
    for a,b in HAND_CONNECTIONS:
        x1,y1 = pts[a]
        x2,y2 = pts[b]
        if x1<=0 or y1<=0 or x2<=0 or y2<=0:
            continue
        cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
    for (x,y) in pts:
        if x<=0 or y<=0:
            continue
        cv2.circle(frame, (int(x),int(y)), 3, color, -1)
    if id_text is not None and len(pts)>0:
        x0,y0 = int(pts[0,0]), int(pts[0,1])
        cv2.putText(frame, f"ID:{id_text}", (x0+5, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# ---------- Simple tracker by wrist proximity ----------
class SimpleWristTracker:
    def __init__(self, max_dist=0.15):
        self.next_id = 1
        self.tracks = {}  # id -> last_wrist_norm (x,y)
        self.max_dist = max_dist

    def match(self, detections):
        """
        detections: list of dicts with 'wrist': (x,y) and optionally 'handedness'
        returns: list of assigned ids in same order as detections
        """
        assigned = []
        used = set()
        for det in detections:
            wrist = det.get("wrist")
            best_id = None
            best_d = None
            for tid, last in self.tracks.items():
                if tid in used:
                    continue
                d = math.hypot(wrist[0]-last[0], wrist[1]-last[1])
                if best_d is None or d < best_d:
                    best_d = d
                    best_id = tid
            if best_d is not None and best_d <= self.max_dist:
                assigned.append(best_id)
                used.add(best_id)
                self.tracks[best_id] = wrist  # update last pos (will be refined by Kalman)
            else:
                # new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = wrist
                assigned.append(tid)
                used.add(tid)
        return assigned

    def remove_stale(self, alive_ids):
        # remove tracks not alive
        to_remove = [tid for tid in list(self.tracks.keys()) if tid not in alive_ids]
        for tid in to_remove:
            self.tracks.pop(tid, None)

# ---------- Main ----------
def main(args):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf
    )

    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)

# --- Reduce resolution for faster processing ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # width in pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # height in pixels

    if not cap.isOpened():
        print("Cannot open source", args.source)
        return


    # Try to set a higher FPS (e.g., 60)
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (img_w, img_h))

    q = Queue(maxsize=1024)
    broadcaster = WebSocketBroadcaster(q, host=args.ws_host, port=args.ws_port)
    broadcaster.start()

    # per-track kalman filters: tid -> list(21 Kalman2D)
    track_filters = {}
    last_seen = {}
    tracker = SimpleWristTracker(max_dist=args.match_max_dist)

    frame_idx = 0
    pbar = tqdm(desc="Processing", unit="frames")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = time.time()
            h,w = frame.shape[:2]

            # run mediapipe (expects RGB)
            if frame is not None and frame.size > 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
            else:
                results = None

            # Preallocate for speed
            detections = []
            handedness = []
            multi_hand_landmarks = getattr(results, "multi_hand_landmarks", None)
            multi_handedness = getattr(results, "multi_handedness", None)

            if multi_hand_landmarks:
                if multi_handedness:
                    handedness = [
                        hh.classification[0].label if hh.classification else None
                        for hh in multi_handedness
                    ]
                else:
                    handedness = [None] * len(multi_hand_landmarks)

                # Use numpy for keypoint extraction for speed
                for idx, hand_landmarks in enumerate(multi_hand_landmarks):
                    kps = np.array(
                        [[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32
                    )
                    wrist = kps[0]
                    detections.append(
                        {
                            "keypoints": kps,
                            "wrist": tuple(wrist),
                            "handedness": handedness[idx],
                        }
                    )

            payload_tracks = {}
            for det in detections:
                kps_norm = det["keypoints"]
                # initialize Kalman filters if needed (one per hand, not per id)
                if "kf" not in det:
                    kfs = [
                        Kalman2D(
                            x=kps_norm[j][0] if kps_norm[j][0] is not None else 0.0,
                            y=kps_norm[j][1] if kps_norm[j][1] is not None else 0.0,
                            dt=1.0 / max(1.0, fps),
                        )
                        for j in range(21)
                    ]
                    det["kf"] = kfs
                else:
                    dt = 1.0 / max(1.0, fps)
                    for kf in det["kf"]:
                        kf.dt = dt
                        kf.F[0, 2] = dt
                        kf.F[1, 3] = dt

                kfs = det["kf"]
                smoothed = np.empty((21, 2), dtype=np.float32)
                for j in range(21):
                    kf = kfs[j]
                    kf.predict()
                    meas = kps_norm[j]
                    pt = kf.update(meas, meas_conf=1.0)
                    smoothed[j, 0] = min(max(pt[0], 0.0), 1.0)
                    smoothed[j, 1] = min(max(pt[1], 0.0), 1.0)

                # draw
                draw_skeleton(frame, smoothed.tolist())

                payload_tracks[str(len(payload_tracks))] = {
                    "keypoints": smoothed.tolist(),
                    "handedness": det.get("handedness"),
                }

            msg = {"timestamp": t, "frame": frame_idx, "tracks": payload_tracks}
            try:
                q.put_nowait(msg)
            except Exception:
                pass

            if args.show:
                cv2.imshow("mediapipe_hand_pose", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writer:
                writer.write(frame)

            frame_idx += 1
            pbar.update(1)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Shutting down...")
        broadcaster.stop()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        pbar.close()
        hands.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="camera index or video file")
    parser.add_argument("--output", type=str, default="", help="save annotated video")
    parser.add_argument("--show", action="store_true", help="show preview window")
    parser.add_argument("--max-hands", type=int, default=2, help="max hands to detect")
    parser.add_argument("--min-det-conf", type=float, default=0.5, help="min detection confidence")
    parser.add_argument("--min-track-conf", type=float, default=0.5, help="min tracking confidence")
    parser.add_argument("--conf-thresh", type=float, default=0.2, help="(unused) kept for API parity")
    parser.add_argument("--track-timeout", type=float, default=2.0, help="seconds before dropping a track")
    parser.add_argument("--match-max-dist", type=float, default=0.15, help="max normalized dist for ID matching")
    parser.add_argument("--ws-host", type=str, default="localhost", help="websocket host")
    parser.add_argument("--ws-port", type=int, default=8765, help="websocket port")
    args = parser.parse_args()
    main(args)

