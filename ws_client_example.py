#!/usr/bin/env python3
import asyncio
import websockets
import json

async def run():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket server!")
        try:
            async for msg in ws:
                data = json.loads(msg)
                tracks = data.get("tracks", {})
                print(f"Frame {data.get('frame')} — tracks: {list(tracks.keys())}")
        except websockets.ConnectionClosed:
            print("WebSocket closed")

if __name__ == "__main__":
    asyncio.run(run())
