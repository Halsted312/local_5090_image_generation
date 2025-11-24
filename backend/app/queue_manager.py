"""In-memory generation queue with WebSocket notifications."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, Optional, Set

from fastapi import WebSocket


class QueueFull(Exception):
    """Raised when the queue is at capacity."""


class WebSocketHub:
    """Tracks WebSocket connections per session and sends JSON messages."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Set[WebSocket]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._loop is None:
            self._loop = loop

    def register(self, session_id: str, ws: WebSocket) -> None:
        with self._lock:
            self._sessions.setdefault(session_id, set()).add(ws)

    def unregister(self, session_id: str, ws: WebSocket) -> None:
        with self._lock:
            conns = self._sessions.get(session_id)
            if conns and ws in conns:
                conns.remove(ws)
            if conns is not None and len(conns) == 0:
                self._sessions.pop(session_id, None)

    def send(self, session_id: str, payload: dict[str, Any]) -> None:
        """Schedule an async send to all websockets for this session."""
        with self._lock:
            targets = list(self._sessions.get(session_id, []))
            loop = self._loop
        if not targets or loop is None:
            return
        for ws in targets:
            asyncio.run_coroutine_threadsafe(ws.send_json(payload), loop)


class GenerationQueue:
    """
    FIFO queue that blocks callers until their ticket reaches the front.
    Broadcasts queue position changes to connected sessions.
    """

    def __init__(self, capacity: int = 64, hub: Optional[WebSocketHub] = None):
        self.capacity = capacity
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: Deque[dict[str, Any]] = deque()
        self._hub = hub or WebSocketHub()

    @property
    def hub(self) -> WebSocketHub:
        return self._hub

    def _broadcast_positions(self) -> None:
        """Notify all queued sessions of their current position."""
        for idx, job in enumerate(list(self._queue)):
            self._hub.send(
                job["session_id"],
                {
                    "type": "queue_update",
                    "generationId": job["generation_id"],
                    "queuePosition": idx,
                },
            )

    def enqueue(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if len(self._queue) >= self.capacity:
                raise QueueFull("Queue full")
            job = {
                "generation_id": uuid.uuid4().hex,
                "session_id": session_id,
                "enqueued_at": time.time(),
            }
            self._queue.append(job)
            position = len(self._queue) - 1
            est_wait = position * 15  # rough estimate in seconds
            self._hub.send(
                session_id,
                {
                    "type": "queued",
                    "generationId": job["generation_id"],
                    "queuePosition": position,
                    "estimatedWaitSeconds": est_wait,
                },
            )
            self._broadcast_positions()
            return job

    def wait_for_turn(self, generation_id: str) -> Optional[dict[str, Any]]:
        with self._cv:
            while True:
                if self._queue and self._queue[0]["generation_id"] == generation_id:
                    job = self._queue[0]
                    self._hub.send(
                        job["session_id"],
                        {
                            "type": "processing",
                            "generationId": generation_id,
                            "queuePosition": 0,
                        },
                    )
                    return job
                self._cv.wait()

    def release(self, generation_id: str, success: bool, payload: dict[str, Any]) -> None:
        with self._cv:
            if self._queue and self._queue[0]["generation_id"] == generation_id:
                job = self._queue.popleft()
                msg_type = "complete" if success else "error"
                message = {
                    "type": msg_type,
                    "generationId": generation_id,
                    **payload,
                }
                self._hub.send(job["session_id"], message)
            self._cv.notify_all()
            self._broadcast_positions()
