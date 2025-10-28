#!/usr/bin/env python3
"""Entry point for the Totetris web application."""

from __future__ import annotations

import asyncio
import configparser
import json
import os
import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


@dataclass
class GameConfig:
    width: int = 20
    height: int = 30
    tick_ms: int = 500
    countdown: int = 5
    game_duration: int = 300
    end_on_negative_score: bool = True

    @classmethod
    def from_file(cls, path: Path) -> "GameConfig":
        parser = configparser.ConfigParser()
        parser.read(path)
        section = parser["game"] if parser.has_section("game") else {}

        def get_bool(key: str, default: bool) -> bool:
            if key not in section:
                return default
            value = str(section.get(key, str(default))).strip().lower()
            return value in {"1", "true", "yes", "on"}

        return cls(
            width=int(section.get("width", cls.width)),
            height=int(section.get("height", cls.height)),
            tick_ms=int(section.get("tick_ms", cls.tick_ms)),
            countdown=int(section.get("countdown", cls.countdown)),
            game_duration=int(section.get("game_duration", cls.game_duration)),
            end_on_negative_score=get_bool(
                "end_on_negative_score", cls.end_on_negative_score
            ),
        )


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class ActivePiece:
    kind: str
    rotation: int
    x: int
    y: int


@dataclass
class PlayerState:
    pid: str
    score: int = 0
    piece: Optional[ActivePiece] = None
    next_queue: List[str] = field(default_factory=list)
    soft_drop: bool = False
    connected: bool = False
    speed_multiplier: float = 1.0
    fall_progress: float = 0.0


TETROMINOES: Dict[str, List[List[Tuple[int, int]]]] = {
    "I": [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
    ],
    "J": [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    "L": [
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
    "O": [
        [(1, 0), (2, 0), (1, 1), (2, 1)],
    ],
    "S": [
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
    ],
    "T": [
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
    ],
    "Z": [
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
    ],
}


PLAYER_COLORS = {"p1": "#70c1ff", "p2": "#ff6b6b"}

DANGER_ZONE_ROWS = 3
PENALTY_PAUSE_SECONDS = 1.0


def generate_room_id(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


# ---------------------------------------------------------------------------
# Game room implementation
# ---------------------------------------------------------------------------


class GameRoom:
    def __init__(self, room_id: str, config: GameConfig) -> None:
        self.room_id = room_id
        self.config = config
        self.width = config.width
        self.height = config.height
        self.board: List[List[Optional[str]]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]
        self.players: Dict[str, PlayerState] = {
            "p1": PlayerState("p1"),
            "p2": PlayerState("p2"),
        }
        self.connections: List[Tuple[WebSocket, str]] = []
        self.status: str = "waiting"
        self.countdown_left: int = config.countdown
        self.game_time_left: int = config.game_duration
        self.loop_task: Optional[asyncio.Task] = None
        self.timer_task: Optional[asyncio.Task] = None
        self.countdown_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.finished_reason: Optional[str] = None
        self.winner: Optional[str] = None
        self.last_cleared_lines: List[int] = []
        self.penalty_event_counter: int = 0
        self.last_penalty_event: Optional[Dict[str, object]] = None
        self.penalty_event_dirty: bool = False
        self.penalty_pause_until: Optional[float] = None
        self.pending_penalty_reset: Optional[asyncio.Task] = None

    # ------------------------- utility methods -------------------------

    def reset_board(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self.board[y][x] = None

    def serialize_board(self) -> List[List[int]]:
        values = {None: 0, "p1": 1, "p2": 2}
        return [[values[self.board[y][x]] for x in range(self.width)] for y in range(self.height)]

    def serialize_players(self) -> Dict[str, Dict[str, object]]:
        result = {}
        for pid, player in self.players.items():
            active = None
            if player.piece:
                active = {
                    "kind": player.piece.kind,
                    "cells": self.piece_cells(player.piece),
                }
            result[pid] = {
                "score": player.score,
                "color": PLAYER_COLORS[pid],
                "active": active,
            }
        return result

    def serialize_state(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "countdown": self.countdown_left,
            "time_left": self.game_time_left,
            "board": self.serialize_board(),
            "players": self.serialize_players(),
            "winner": self.winner,
            "reason": self.finished_reason,
            "cleared_lines": list(self.last_cleared_lines),
            "penalty_event": dict(self.last_penalty_event)
            if self.penalty_event_dirty and self.last_penalty_event
            else None,
        }

    def piece_rotations(self, kind: str) -> List[List[Tuple[int, int]]]:
        return TETROMINOES[kind]

    def piece_cells(self, piece: ActivePiece) -> List[Tuple[int, int]]:
        rotation = self.piece_rotations(piece.kind)[piece.rotation]
        return [(piece.x + dx, piece.y + dy) for dx, dy in rotation]

    def active_piece_cells(self, exclude_pid: Optional[str] = None) -> Set[Tuple[int, int]]:
        occupied: Set[Tuple[int, int]] = set()
        for pid, player in self.players.items():
            if exclude_pid is not None and pid == exclude_pid:
                continue
            if player.piece:
                occupied.update(self.piece_cells(player.piece))
        return occupied

    def collision_type(self, piece: ActivePiece, owner: Optional[str] = None) -> Optional[str]:
        blocked_cells = self.active_piece_cells(exclude_pid=owner)
        for x, y in self.piece_cells(piece):
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return "bounds"
            if self.board[y][x] is not None:
                return "board"
            if (x, y) in blocked_cells:
                return "opponent"
        return None

    def can_place(self, piece: ActivePiece, owner: Optional[str] = None) -> bool:
        return self.collision_type(piece, owner=owner) is None

    def spawn_anchor(self, pid: str) -> Tuple[int, int]:
        top_y = 0
        if pid == "p1":
            return 0, top_y
        if pid == "p2":
            return max(0, self.width - 4), top_y
        return max(0, self.width // 2 - 2), top_y

    def spawn_piece(self, player: PlayerState) -> bool:
        if not player.next_queue:
            player.next_queue.extend(self.generate_bag())
        kind = player.next_queue.pop(0)
        anchor_x, anchor_y = self.spawn_anchor(player.pid)
        direction = 1 if anchor_x <= self.width // 2 else -1
        tried: Set[int] = set()
        offsets = [0]
        for step in range(1, self.width):
            offsets.append(step * direction)
            offsets.append(-step * direction)
        for delta in offsets:
            x = anchor_x + delta
            if x < 0 or x >= self.width:
                continue
            if x in tried:
                continue
            tried.add(x)
            for rotation_index in range(len(self.piece_rotations(kind))):
                candidate = ActivePiece(kind, rotation_index, x, anchor_y)
                if self.can_place(candidate, owner=player.pid):
                    player.piece = candidate
                    player.fall_progress = 0.0
                    return True
        player.piece = None
        return False

    def generate_bag(self) -> List[str]:
        kinds = list(TETROMINOES.keys())
        random.shuffle(kinds)
        return kinds

    def lock_piece(self, player: PlayerState) -> None:
        if not player.piece:
            return
        placed_cells = self.piece_cells(player.piece)
        touched_danger_zone = any(y < DANGER_ZONE_ROWS for _, y in placed_cells)
        for x, y in placed_cells:
            if 0 <= y < self.height and 0 <= x < self.width:
                self.board[y][x] = player.pid
        player.piece = None
        player.fall_progress = 0.0
        cleared = self.clear_full_lines()
        self.last_cleared_lines = cleared
        if cleared:
            player.score += len(cleared)
        if touched_danger_zone:
            self.apply_penalty_and_reset(
                player, highlight_zone=True, blink_cells=placed_cells
            )
            return
        if not self.spawn_piece(player):
            self.apply_penalty_and_reset(player)
            return
        if self.config.end_on_negative_score and player.score < 0:
            self.finish_game(
                winner=self.opponent_id(player.pid), reason="negative_score"
            )

    def apply_penalty_and_reset(
        self,
        offender: PlayerState,
        *,
        highlight_zone: bool = False,
        blink_cells: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        offender.score -= 10
        if self.config.end_on_negative_score and offender.score < 0:
            self.finish_game(
                winner=self.opponent_id(offender.pid), reason="negative_score"
            )
            return
        pause_ms = int(PENALTY_PAUSE_SECONDS * 1000) if highlight_zone else 0
        self._record_penalty_event(
            offender.pid,
            highlight_zone=highlight_zone,
            blink_cells=blink_cells,
            pause_ms=pause_ms,
        )
        if highlight_zone:
            self._start_penalty_pause(delay=PENALTY_PAUSE_SECONDS)
        else:
            self._cancel_pending_penalty_reset()
            self._complete_penalty_reset()

    def clear_full_lines(self) -> List[int]:
        removed: List[int] = []
        y = self.height - 1
        while y >= 0:
            if all(self.board[y][x] is not None for x in range(self.width)):
                removed.append(y)
                del self.board[y]
                self.board.insert(0, [None for _ in range(self.width)])
            else:
                y -= 1
        removed.sort()
        return removed

    def opponent_id(self, pid: str) -> Optional[str]:
        return "p2" if pid == "p1" else "p1" if pid == "p2" else None

    def _record_penalty_event(
        self,
        player_id: str,
        *,
        highlight_zone: bool,
        blink_cells: Optional[List[Tuple[int, int]]],
        pause_ms: int,
    ) -> None:
        self.penalty_event_counter += 1
        event: Dict[str, object] = {
            "id": self.penalty_event_counter,
            "player": player_id,
            "highlight_zone": highlight_zone,
        }
        if highlight_zone and blink_cells:
            event["blink_cells"] = list(blink_cells)
            if pause_ms > 0:
                event["pause_ms"] = pause_ms
        elif pause_ms > 0:
            event["pause_ms"] = pause_ms
        self.last_penalty_event = event
        self.penalty_event_dirty = True

    def _cancel_pending_penalty_reset(self) -> None:
        if self.pending_penalty_reset:
            self.pending_penalty_reset.cancel()
            self.pending_penalty_reset = None
        self.penalty_pause_until = None

    def _start_penalty_pause(self, *, delay: float) -> None:
        self._cancel_pending_penalty_reset()
        self.penalty_pause_until = time.monotonic() + delay
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.penalty_pause_until = None
            self._complete_penalty_reset()
            return
        self.pending_penalty_reset = loop.create_task(self._delayed_penalty_reset(delay))

    async def _delayed_penalty_reset(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            async with self.lock:
                self._complete_penalty_reset()
                self.penalty_pause_until = None
        except asyncio.CancelledError:
            return
        finally:
            self.pending_penalty_reset = None
        await self.broadcast_state()

    def _complete_penalty_reset(self) -> None:
        self.reset_board()
        self.last_cleared_lines = []
        for participant in self.players.values():
            participant.piece = None
            participant.soft_drop = False
            participant.speed_multiplier = 1.0
            participant.fall_progress = 0.0
            if not participant.next_queue:
                participant.next_queue.extend(self.generate_bag())
            self.spawn_piece(participant)

    def penalty_pause_active(self) -> bool:
        if self.penalty_pause_until is None:
            return False
        if time.monotonic() < self.penalty_pause_until:
            return True
        if self.pending_penalty_reset and not self.pending_penalty_reset.done():
            return True
        self.penalty_pause_until = None
        return False

    def move_piece(self, player: PlayerState, dx: int, dy: int, rotate: int = 0) -> None:
        if not player.piece:
            return
        piece = player.piece
        rotation_count = len(self.piece_rotations(piece.kind))
        new_rotation = (piece.rotation + rotate) % rotation_count
        candidate = ActivePiece(piece.kind, new_rotation, piece.x + dx, piece.y + dy)
        if self.can_place(candidate, owner=player.pid):
            player.piece = candidate

    def soft_drop_piece(self, player: PlayerState) -> None:
        if not player.piece:
            return
        player.fall_progress = 0.0
        while True:
            next_piece = ActivePiece(
                player.piece.kind, player.piece.rotation, player.piece.x, player.piece.y + 1
            )
            collision = self.collision_type(next_piece, owner=player.pid)
            if collision is None:
                player.piece = next_piece
            elif collision == "opponent":
                break
            else:
                self.lock_piece(player)
                break

    def set_speed_multiplier(self, player: PlayerState, multiplier: float) -> None:
        try:
            value = float(multiplier)
        except (TypeError, ValueError):
            return
        player.speed_multiplier = max(0.1, min(value, 5.0))

    def tick_player(self, player: PlayerState) -> None:
        if not player.piece:
            return
        multiplier = max(player.speed_multiplier, 0.1)
        player.fall_progress += multiplier
        if player.soft_drop:
            player.fall_progress += 1.0
            player.soft_drop = False
        steps = int(player.fall_progress)
        player.fall_progress -= steps
        if steps <= 0:
            return
        for _ in range(steps):
            next_piece = ActivePiece(
                player.piece.kind, player.piece.rotation, player.piece.x, player.piece.y + 1
            )
            collision = self.collision_type(next_piece, owner=player.pid)
            if collision is None:
                player.piece = next_piece
            elif collision == "opponent":
                break
            else:
                self.lock_piece(player)
                break

    # ------------------------- lifecycle control -------------------------

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        async with self.lock:
            pid = self.next_available_player()
            if pid is None:
                await websocket.send_text(json.dumps({"type": "error", "message": "room_full"}))
                await websocket.close(code=1000)
                raise HTTPException(status_code=400, detail="Room already has two players")
            player = self.players[pid]
            player.connected = True
            player.speed_multiplier = 1.0
            player.fall_progress = 0.0
            player.soft_drop = False
            if not player.next_queue:
                player.next_queue.extend(self.generate_bag())
            if not player.piece:
                self.spawn_piece(player)
            self.connections.append((websocket, pid))
            await self.broadcast_state()
            await self.ensure_game_flow()
            return pid

    async def disconnect(self, websocket: WebSocket, pid: str) -> None:
        async with self.lock:
            player = self.players.get(pid)
            if player:
                player.connected = False
                player.speed_multiplier = 1.0
                player.fall_progress = 0.0
                player.soft_drop = False
            for index, (ws, owner) in enumerate(list(self.connections)):
                if ws is websocket and owner == pid:
                    del self.connections[index]
                    break
            if self.status in {"running", "countdown"}:
                self.finish_game(winner=self.opponent_id(pid), reason="disconnect")
            elif self.status == "waiting":
                await self.broadcast_state()

    def next_available_player(self) -> Optional[str]:
        for pid, player in self.players.items():
            if not player.connected:
                return pid
        return None

    async def ensure_game_flow(self) -> None:
        if self.status == "waiting" and all(p.connected for p in self.players.values()):
            self.status = "countdown"
            self.countdown_left = self.config.countdown
            self.countdown_task = asyncio.create_task(self._run_countdown())

    async def _run_countdown(self) -> None:
        while self.countdown_left > 0:
            await self.broadcast_state()
            await asyncio.sleep(1)
            self.countdown_left -= 1
        self.status = "running"
        self.loop_task = asyncio.create_task(self._run_game_loop())
        self.timer_task = asyncio.create_task(self._run_game_timer())
        await self.broadcast_state()

    async def _run_game_loop(self) -> None:
        interval = self.config.tick_ms / 1000.0
        try:
            while self.status == "running":
                async with self.lock:
                    if not self.penalty_pause_active():
                        for player in self.players.values():
                            self.tick_player(player)
                await self.broadcast_state()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def _run_game_timer(self) -> None:
        try:
            while self.status == "running" and self.game_time_left > 0:
                await asyncio.sleep(1)
                self.game_time_left -= 1
                await self.broadcast_state()
            if self.status == "running":
                self.finish_game(winner=self.determine_winner(), reason="time_up")
                await self.broadcast_state()
        except asyncio.CancelledError:
            pass

    def determine_winner(self) -> Optional[str]:
        scores = {pid: player.score for pid, player in self.players.items()}
        if scores["p1"] > scores["p2"]:
            return "p1"
        if scores["p2"] > scores["p1"]:
            return "p2"
        return None

    def finish_game(self, winner: Optional[str], reason: str) -> None:
        if self.status == "finished":
            return
        self.status = "finished"
        self.winner = winner
        self.finished_reason = reason
        self._cancel_pending_penalty_reset()
        if self.loop_task:
            self.loop_task.cancel()
            self.loop_task = None
        if self.timer_task:
            self.timer_task.cancel()
            self.timer_task = None
        if self.countdown_task:
            self.countdown_task.cancel()
            self.countdown_task = None

    async def broadcast_state(self) -> None:
        state = self.serialize_state()
        payloads: List[Tuple[WebSocket, str]] = []
        for websocket, pid in list(self.connections):
            personal_state = dict(state)
            personal_state["you"] = pid
            payloads.append((websocket, json.dumps(personal_state)))
        for websocket, payload in payloads:
            try:
                await websocket.send_text(payload)
            except RuntimeError:
                continue
        self.last_cleared_lines = []
        self.penalty_event_dirty = False

    async def handle_message(self, pid: str, data: Dict[str, object]) -> None:
        player = self.players[pid]
        action = data.get("action")
        if self.status != "running":
            if action == "restart" and self.status == "finished":
                await self.restart()
            return
        if self.penalty_pause_active():
            return
        if action == "move":
            direction = data.get("direction")
            if direction == "left":
                self.move_piece(player, dx=-1, dy=0)
            elif direction == "right":
                self.move_piece(player, dx=1, dy=0)
        elif action == "rotate":
            self.move_piece(player, dx=0, dy=0, rotate=1)
        elif action == "drop":
            player.soft_drop = True
        elif action == "slam":
            self.soft_drop_piece(player)
        elif action == "speed":
            self.set_speed_multiplier(player, data.get("multiplier", 1))
        await self.broadcast_state()

    async def restart(self) -> None:
        self._cancel_pending_penalty_reset()
        self.reset_board()
        for player in self.players.values():
            player.score = 0
            player.soft_drop = False
            player.next_queue.clear()
            player.piece = None
            player.speed_multiplier = 1.0
            player.fall_progress = 0.0
            player.next_queue.extend(self.generate_bag())
            self.spawn_piece(player)
        self.status = "waiting"
        self.countdown_left = self.config.countdown
        self.game_time_left = self.config.game_duration
        await self.broadcast_state()
        await self.ensure_game_flow()


# ---------------------------------------------------------------------------
# Game manager
# ---------------------------------------------------------------------------


class GameManager:
    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.rooms: Dict[str, GameRoom] = {}
        self.lock = asyncio.Lock()

    async def create_room(self) -> GameRoom:
        async with self.lock:
            for _ in range(5):
                room_id = generate_room_id()
                if room_id not in self.rooms:
                    room = GameRoom(room_id, self.config)
                    self.rooms[room_id] = room
                    return room
        raise RuntimeError("Unable to create room id")

    async def get_room(self, room_id: str) -> GameRoom:
        async with self.lock:
            room = self.rooms.get(room_id)
            if room is None:
                room = GameRoom(room_id, self.config)
                self.rooms[room_id] = room
            return room


# ---------------------------------------------------------------------------
# Web application
# ---------------------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent

config = GameConfig.from_file(BASE_DIR / "totetris.ini")
manager = GameManager(config)

app = FastAPI(title="Totetris")


INDEX_HTML = """
<!DOCTYPE html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Totetris</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; }
    main { max-width: 720px; margin: 0 auto; padding: 3rem 1.5rem; text-align: center; display: flex; flex-direction: column; gap: 2rem; align-items: center; }
    h1 { margin: 0; font-size: clamp(2.25rem, 4vw, 3rem); }
    .lead { font-size: 1.1rem; line-height: 1.6; margin: 0; color: #cbd5f5; }
    .rules { list-style: none; padding: 0; margin: 0; display: grid; gap: 0.75rem; }
    .rules li { background: rgba(15, 23, 42, 0.6); padding: 0.85rem 1rem; border-radius: 0.75rem; border: 1px solid rgba(148, 163, 184, 0.15); }
    .details { width: 100%; background: rgba(15, 23, 42, 0.55); border-radius: 1rem; padding: 1.75rem; border: 1px solid rgba(148, 163, 184, 0.12); text-align: left; display: grid; gap: 1.5rem; box-shadow: 0 18px 45px rgba(15, 23, 42, 0.4); }
    .details h2 { margin: 0; font-size: 1.4rem; }
    .details p { margin: 0; line-height: 1.6; color: #cbd5f5; }
    .details-grid { display: grid; gap: 1rem; }
    @media (min-width: 720px) { .details-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    .details-card { background: rgba(15, 23, 42, 0.75); border-radius: 0.9rem; border: 1px solid rgba(148, 163, 184, 0.14); padding: 1.1rem 1.25rem; display: grid; gap: 0.75rem; }
    .details-card h3 { margin: 0; font-size: 1.1rem; color: #f8fafc; }
    .details-card ul { margin: 0; padding-left: 1.1rem; display: grid; gap: 0.5rem; color: #cbd5f5; }
    .details-card li { line-height: 1.5; }
    .play-button { padding: 1.1rem 2.75rem; font-size: 1.25rem; border-radius: 999px; border: none; cursor: pointer; background: linear-gradient(135deg, #38bdf8, #818cf8); color: #0f172a; font-weight: 700; box-shadow: 0 20px 40px rgba(56, 189, 248, 0.35); transition: transform 0.2s ease, box-shadow 0.2s ease; }
    .play-button:hover { transform: translateY(-2px); box-shadow: 0 24px 50px rgba(56, 189, 248, 0.45); }
    .play-button:disabled { opacity: 0.6; cursor: wait; box-shadow: none; }
    .note { font-size: 0.95rem; color: #94a3b8; margin: 0; line-height: 1.5; }
  </style>
</head>
<body>
  <main>
    <h1>Totetris</h1>
    <p class=\"lead\">Сыграйте в дуэльный тетрис с другом: пригласите соперника и выясните, кто продержится дольше.</p>
    <ul class=\"rules\">
      <li>Первый игрок создаёт приватную комнату и получает ссылку-приглашение.</li>
      <li>Второй игрок переходит по ссылке и присоединяется к общей комнате.</li>
      <li>После пятисекундного отсчёта оба получают фигуры и начинают зарабатывать очки.</li>
    </ul>
    <section class=\"details\">
      <div>
        <h2>Как идёт матч</h2>
        <p>Игра длится ограниченное время: после отсчёта оба игрока получают свои фигуры и очищают линии на общем поле. Если кто-то переполняет поле, ему выдаётся штраф, но матч продолжается.</p>
      </div>
      <div class=\"details-grid\">
        <article class=\"details-card\">
          <h3>Ход игры</h3>
          <ul>
            <li>Каждый игрок управляет собственной фигурой на общем поле, при этом фигуры не сталкиваются между собой.</li>
            <li>Полностью заполненные горизонтальные линии сгорают, а верхние строки опускаются вниз, освобождая место.</li>
            <li>Как только фигура установлена, игрок мгновенно получает следующую из своей очереди.</li>
          </ul>
        </article>
        <article class=\"details-card\">
          <h3>Подсчёт очков</h3>
          <ul>
            <li>За каждую сожжённую линию игрок, совершивший действие, получает 1 балл.</li>
            <li>Если игрок переполняет поле, с его счёта снимается 10 баллов, доска очищается и матч продолжается.</li>
            <li>Если счёт становится отрицательным, игрок сразу проигрывает; иначе по таймеру побеждает тот, у кого очков больше.</li>
          </ul>
        </article>
      </div>
    </section>
    <button id=\"play\" class=\"play-button\">Сыграть с другом</button>
    <p class=\"note\">Управление: стрелки &larr; &rarr; — движение, стрелка вверх — замедление в 3 раза, стрелка вниз — ускорение в 3 раза, пробел — поворот.</p>
  </main>
  <script>
    const playButton = document.getElementById('play');

    async function startGame() {
      if (playButton.disabled) return;
      const originalText = playButton.textContent;
      playButton.disabled = true;
      playButton.textContent = 'Создание комнаты...';
      try {
        const response = await fetch('/api/create', { method: 'POST' });
        if (!response.ok) {
          throw new Error('failed');
        }
        const data = await response.json();
        window.location.href = `/game/${data.room}`;
      } catch (error) {
        alert('Не удалось создать игру. Попробуйте ещё раз.');
        playButton.disabled = false;
        playButton.textContent = originalText;
      }
    }

    playButton.addEventListener('click', startGame);
  </script>
</body>
</html>
"""


GAME_HTML = """
<!DOCTYPE html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Totetris — игра</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }
    main { position: relative; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 2rem 1rem; gap: 1.5rem; min-height: 100vh; box-sizing: border-box; }
    h1 { margin: 0; }
    #board { background: #1e293b; border: 2px solid #334155; border-radius: 0.5rem; box-shadow: 0 12px 40px rgba(15,23,42,0.35); }
    .game-area { display: flex; flex-direction: column; align-items: center; gap: 1.25rem; }
    .game-area.hidden { display: none !important; }
    .hud { display: flex; flex-wrap: wrap; gap: 1.5rem; justify-content: center; }
    .panel { background: rgba(15, 23, 42, 0.75); padding: 1.25rem 1.5rem; border-radius: 0.75rem; min-width: 200px; box-shadow: 0 18px 45px rgba(15, 23, 42, 0.45); border: 1px solid rgba(148, 163, 184, 0.12); }
    h2 { margin: 0 0 0.75rem 0; font-size: 1rem; }
    .scores { list-style: none; padding: 0; margin: 0; }
    .scores li { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; }
    .badge { display: inline-block; width: 12px; height: 12px; border-radius: 999px; margin-right: 0.5rem; }
    .status { font-weight: 600; }
    .link { color: #38bdf8; cursor: pointer; }
    .actions { display: flex; gap: 0.75rem; margin-top: 0.75rem; flex-wrap: wrap; }
    button { padding: 0.65rem 1.1rem; border-radius: 0.65rem; border: none; cursor: pointer; background: #38bdf8; color: #0f172a; font-weight: 600; }
    button:hover { background: #0ea5e9; }
    .overlay { position: fixed; inset: 0; background: rgba(15, 23, 42, 0.94); display: flex; align-items: center; justify-content: center; padding: 2rem; z-index: 20; }
    .overlay.hidden { display: none; }
    .overlay-box { max-width: 440px; width: min(90vw, 440px); background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 1rem; padding: 2.5rem 2rem; text-align: center; box-shadow: 0 30px 90px rgba(15, 23, 42, 0.65); }
    .overlay-box h2 { margin: 0 0 1rem 0; font-size: 1.6rem; }
    .overlay-box p { margin: 0 0 1.75rem 0; line-height: 1.6; color: #cbd5f5; }
    .overlay-share { display: flex; flex-direction: column; gap: 0.75rem; align-items: center; }
    .overlay-share.hidden { display: none; }
    .overlay-link { background: rgba(30, 41, 59, 0.92); padding: 0.75rem 1rem; border-radius: 0.75rem; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; word-break: break-all; width: 100%; box-sizing: border-box; border: 1px solid rgba(148, 163, 184, 0.25); color: #e2e8f0; }
    .overlay-copy { padding: 0.65rem 1.6rem; border-radius: 999px; border: none; cursor: pointer; background: #38bdf8; color: #0f172a; font-weight: 600; }
    .overlay-copy:hover { background: #0ea5e9; }
    .hidden { display: none !important; }
  </style>
</head>
<body>
  <main>
    <h1>Totetris</h1>
    <div id=\"game-area\" class=\"game-area hidden\">
      <canvas id=\"board\" width=\"400\" height=\"600\"></canvas>
      <div class=\"hud\">
        <div class=\"panel\">
          <h2>Статус</h2>
          <div id=\"status\" class=\"status\"></div>
          <div id=\"timer\"></div>
          <div class=\"actions\">
            <button id=\"restart\" disabled>Рестарт</button>
            <a class=\"link\" id=\"invite\" target=\"_blank\">Скопировать ссылку</a>
          </div>
        </div>
        <div class=\"panel\">
          <h2>Счёт</h2>
          <ul id=\"scores\" class=\"scores\"></ul>
        </div>
      </div>
    </div>
  </main>
  <div id=\"overlay\" class=\"overlay\">
    <div class=\"overlay-box\">
      <h2 id=\"overlay-title\">Подключение...</h2>
      <p id=\"overlay-text\">Подключаемся к комнате, пожалуйста, подождите.</p>
      <div id=\"overlay-share\" class=\"overlay-share hidden\">
        <span id=\"overlay-link\" class=\"overlay-link\"></span>
        <button id=\"overlay-copy\" class=\"overlay-copy\">Скопировать ссылку</button>
      </div>
    </div>
  </div>
  <script>
    const roomId = "{room_id}";
    const CELL_SIZE = 20;
    const CLEAR_ANIMATION_DURATION = 480;
    const DANGER_ZONE_ROWS = 3;
    const DANGER_FLASH_DURATION = 1000;
    const BOARD_RESET_DURATION = 900;
    const PENALTY_BLINK_BASE_DURATION = 1000;
    const PENALTY_BLINK_FREQUENCY = 9;
    const canvas = document.getElementById('board');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('status');
    const timerEl = document.getElementById('timer');
    const scoresEl = document.getElementById('scores');
    const restartButton = document.getElementById('restart');
    const inviteLink = document.getElementById('invite');
    const inviteLinkDefaultText = inviteLink.textContent;
    const gameArea = document.getElementById('game-area');
    const overlay = document.getElementById('overlay');
    const overlayTitle = document.getElementById('overlay-title');
    const overlayText = document.getElementById('overlay-text');
    const overlayShare = document.getElementById('overlay-share');
    const overlayLink = document.getElementById('overlay-link');
    const overlayCopy = document.getElementById('overlay-copy');
    const overlayCopyDefaultText = overlayCopy.textContent;

    inviteLink.href = window.location.href;
    inviteLink.addEventListener('click', (event) => {
      event.preventDefault();
      copyInviteLink(inviteLink);
    });

    overlayCopy.addEventListener('click', (event) => {
      event.preventDefault();
      copyInviteLink(overlayCopy);
    });

    let ws = null;
    let you = null;
    let currentState = null;
    let animationFrame = null;
    let clearAnimation = null;
    let penaltyFlash = null;
    let penaltyBlink = null;
    let boardResetAnimation = null;
    let lastPenaltyEventId = 0;

    function copyInviteLink(target) {
      const originalText = target.textContent;
      const showSuccess = () => {
        target.textContent = 'Ссылка скопирована!';
        setTimeout(() => { target.textContent = originalText; }, 2000);
      };
      const fallback = () => {
        const response = window.prompt('Скопируйте ссылку и отправьте другу:', window.location.href);
        if (response !== null) {
          showSuccess();
        }
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(window.location.href).then(showSuccess).catch(fallback);
      } else {
        fallback();
      }
    }

    function showOverlay(title, text, { showShare = false } = {}) {
      overlayTitle.textContent = title;
      overlayText.textContent = text;
      if (showShare) {
        overlayShare.classList.remove('hidden');
        overlayLink.textContent = window.location.href;
        overlayCopy.textContent = overlayCopyDefaultText;
        inviteLink.textContent = inviteLinkDefaultText;
      } else {
        overlayShare.classList.add('hidden');
      }
      overlay.classList.remove('hidden');
    }

    function hideOverlay() {
      overlay.classList.add('hidden');
    }

    function connect() {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${protocol}://${window.location.host}/ws/${roomId}`);

      ws.addEventListener('message', (event) => {
        const state = JSON.parse(event.data);
        you = state.you;
        updateUI(state);
      });

      ws.addEventListener('close', () => {
        statusEl.textContent = 'Соединение потеряно';
        showOverlay('Соединение потеряно', 'Перезагрузите страницу, чтобы попробовать снова.');
        gameArea.classList.add('hidden');
      });
    }

    function scheduleRender() {
      if (animationFrame === null) {
        animationFrame = requestAnimationFrame(render);
      }
    }

    function render(timestamp) {
      animationFrame = null;
      if (!currentState) return;
      const now = typeof timestamp === 'number' ? timestamp : performance.now();
      let animation = null;
      let effects = {};
      let needsMoreFrames = false;
      if (clearAnimation) {
        const progress = Math.min((now - clearAnimation.start) / CLEAR_ANIMATION_DURATION, 1);
        animation = { lines: clearAnimation.lines, progress };
        if (progress < 1) {
          needsMoreFrames = true;
        } else {
          clearAnimation = null;
        }
      }
      if (penaltyFlash) {
        const elapsed = now - penaltyFlash.start;
        if (elapsed < DANGER_FLASH_DURATION) {
          effects.dangerFlash = { progress: elapsed / DANGER_FLASH_DURATION };
          needsMoreFrames = true;
        } else {
          penaltyFlash = null;
        }
      }
      if (penaltyBlink) {
        const duration = Math.max(
          penaltyBlink.duration ?? PENALTY_BLINK_BASE_DURATION,
          0
        );
        const effectiveDuration = duration > 0 ? duration : PENALTY_BLINK_BASE_DURATION;
        const elapsed = now - penaltyBlink.start;
        if (elapsed < effectiveDuration) {
          const seconds = elapsed / 1000;
          const flash = Math.pow(
            Math.sin(2 * Math.PI * PENALTY_BLINK_FREQUENCY * seconds),
            2
          );
          const progress = Math.min(elapsed / effectiveDuration, 1);
          effects.blink = {
            cells: penaltyBlink.cells,
            player: penaltyBlink.player,
            flash,
            progress,
          };
          needsMoreFrames = true;
        } else {
          penaltyBlink = null;
        }
      }
      if (boardResetAnimation) {
        const delay = Math.max(boardResetAnimation.delay ?? 0, 0);
        const elapsed = now - boardResetAnimation.start;
        if (elapsed < delay) {
          needsMoreFrames = true;
        } else {
          const waveElapsed = elapsed - delay;
          if (waveElapsed < BOARD_RESET_DURATION) {
            effects.boardClear = {
              progress: waveElapsed / BOARD_RESET_DURATION,
            };
            needsMoreFrames = true;
          } else {
            boardResetAnimation = null;
          }
        }
      }
      drawBoard(currentState.board, currentState.players, animation, effects);
      if (needsMoreFrames) {
        animationFrame = requestAnimationFrame(render);
      }
    }

    function drawBoard(board, players, animation, effects = {}) {
      const rows = board.length;
      const cols = board[0].length;
      canvas.width = cols * CELL_SIZE;
      canvas.height = rows * CELL_SIZE;

      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const dangerFlash = effects.dangerFlash;
      const blinking = effects.blink;
      let dangerAlpha = 0.12;
      if (dangerFlash) {
        const t = Math.min(Math.max(dangerFlash.progress, 0), 1);
        const pulse = Math.sin(Math.PI * t);
        dangerAlpha = Math.min(0.75, 0.12 + 0.28 * (1 - t) + 0.42 * pulse);
      }
      ctx.fillStyle = `rgba(248,113,113,${dangerAlpha})`;
      for (let y = 0; y < Math.min(DANGER_ZONE_ROWS, rows); y++) {
        ctx.fillRect(0, y * CELL_SIZE, canvas.width, CELL_SIZE);
      }
      if (dangerFlash) {
        ctx.save();
        const glowStrength = Math.min(Math.max(1 - dangerFlash.progress, 0), 1);
        ctx.globalAlpha = 0.35 * glowStrength;
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, CELL_SIZE * DANGER_ZONE_ROWS);
        gradient.addColorStop(0, 'rgba(248,250,252,0.0)');
        gradient.addColorStop(0.5, 'rgba(254,226,226,0.9)');
        gradient.addColorStop(1, 'rgba(248,250,252,0.0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, CELL_SIZE * Math.min(DANGER_ZONE_ROWS, rows));
        ctx.restore();
        ctx.save();
        ctx.globalAlpha = 0.4 * glowStrength;
        ctx.strokeStyle = 'rgba(248,113,113,0.9)';
        ctx.lineWidth = 2;
        ctx.strokeRect(1, 1, canvas.width - 2, CELL_SIZE * Math.min(DANGER_ZONE_ROWS, rows) - 2);
        ctx.restore();
      }

      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const owner = board[y][x];
          if (owner === 0) {
            ctx.strokeStyle = 'rgba(51,65,85,0.4)';
            ctx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            continue;
          }
          const pid = owner === 1 ? 'p1' : 'p2';
          ctx.fillStyle = players[pid].color;
          ctx.fillRect(x * CELL_SIZE + 1, y * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2);
        }
      }

      if (
        blinking &&
        Array.isArray(blinking.cells) &&
        blinking.cells.length > 0
      ) {
        const flash = Math.min(Math.max(blinking.flash ?? 0, 0), 1);
        const progress = Math.min(Math.max(blinking.progress ?? 0, 0), 1);
        const offenderColor =
          (blinking.player && players[blinking.player]?.color) || '#f87171';
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        ctx.globalAlpha = 0.45 + 0.45 * flash;
        ctx.fillStyle = offenderColor;
        blinking.cells.forEach((cell) => {
          const [cx, cy] = cell;
          if (typeof cx !== 'number' || typeof cy !== 'number') return;
          if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) return;
          ctx.fillRect(cx * CELL_SIZE + 1, cy * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2);
        });
        ctx.restore();

        const innerInset = CELL_SIZE * (0.2 + 0.15 * (1 - progress));
        ctx.save();
        ctx.globalAlpha = 0.35 + 0.55 * flash;
        ctx.fillStyle = '#f8fafc';
        blinking.cells.forEach((cell) => {
          const [cx, cy] = cell;
          if (typeof cx !== 'number' || typeof cy !== 'number') return;
          if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) return;
          ctx.fillRect(
            cx * CELL_SIZE + innerInset,
            cy * CELL_SIZE + innerInset,
            CELL_SIZE - innerInset * 2,
            CELL_SIZE - innerInset * 2
          );
        });
        ctx.restore();

        ctx.save();
        ctx.globalAlpha = 0.55 + 0.4 * flash;
        ctx.strokeStyle = '#fef2f2';
        ctx.lineWidth = 2;
        blinking.cells.forEach((cell) => {
          const [cx, cy] = cell;
          if (typeof cx !== 'number' || typeof cy !== 'number') return;
          if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) return;
          ctx.strokeRect(
            cx * CELL_SIZE + 1.5,
            cy * CELL_SIZE + 1.5,
            CELL_SIZE - 3,
            CELL_SIZE - 3
          );
        });
        ctx.restore();
      }

      for (const pid of Object.keys(players)) {
        const active = players[pid].active;
        if (!active) continue;
        ctx.fillStyle = players[pid].color;
        ctx.globalAlpha = pid === you ? 0.9 : 0.5;
        for (const [x, y] of active.cells) {
          ctx.fillRect(x * CELL_SIZE + 1, y * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2);
        }
        ctx.globalAlpha = 1;
      }

      if (effects.boardClear) {
        const t = Math.min(Math.max(effects.boardClear.progress, 0), 1);
        const waveFront = t * (rows + 6);
        for (let y = 0; y < rows; y++) {
          const distance = waveFront - y;
          if (distance <= 0) continue;
          const local = Math.min(distance / 3, 1);
          const alpha = 0.45 * (1 - Math.pow(local, 0.8));
          if (alpha <= 0) continue;
          const top = y * CELL_SIZE;
          ctx.save();
          ctx.globalAlpha = alpha;
          const gradient = ctx.createLinearGradient(0, top, canvas.width, top + CELL_SIZE);
          gradient.addColorStop(0, 'rgba(56,189,248,0)');
          gradient.addColorStop(0.5, 'rgba(226,232,240,0.75)');
          gradient.addColorStop(1, 'rgba(56,189,248,0)');
          ctx.fillStyle = gradient;
          ctx.fillRect(0, top, canvas.width, CELL_SIZE);
          ctx.restore();
        }
        ctx.save();
        const glow = Math.max(0, 1 - t * 0.85);
        ctx.globalAlpha = 0.35 * glow;
        const centerY = canvas.height * (0.25 + 0.65 * t);
        const radius = canvas.width * (0.45 + 0.55 * t);
        const radial = ctx.createRadialGradient(
          canvas.width / 2,
          centerY,
          Math.max(8, radius * 0.25),
          canvas.width / 2,
          centerY,
          radius
        );
        radial.addColorStop(0, 'rgba(56,189,248,0.55)');
        radial.addColorStop(1, 'rgba(15,23,42,0)');
        ctx.fillStyle = radial;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
      }

      if (animation && Array.isArray(animation.lines) && animation.lines.length > 0) {
        const progress = Math.min(Math.max(animation.progress ?? 0, 0), 1);
        const fade = Math.sin(Math.PI * progress);
        ctx.save();
        animation.lines.forEach((line) => {
          if (line < 0 || line >= rows) {
            return;
          }
          const top = line * CELL_SIZE;
          const centerY = top + CELL_SIZE / 2;

          ctx.save();
          ctx.globalAlpha = 0.65 * (1 - progress * 0.8);
          const gradient = ctx.createLinearGradient(0, top, canvas.width, top + CELL_SIZE);
          gradient.addColorStop(0, 'rgba(56,189,248,0)');
          gradient.addColorStop(0.5, `rgba(248,250,252,${0.8 * (1 - progress)})`);
          gradient.addColorStop(1, 'rgba(56,189,248,0)');
          ctx.fillStyle = gradient;
          ctx.fillRect(0, top, canvas.width, CELL_SIZE);
          ctx.restore();

          ctx.save();
          const collapse = Math.min(progress * 1.05, 1);
          const beamWidth = canvas.width * (1 - 0.55 * collapse);
          const beamHeight = CELL_SIZE * (1 - 0.6 * collapse);
          ctx.translate(canvas.width / 2, centerY);
          ctx.rotate((Math.PI / 32) * (1 - collapse));
          ctx.fillStyle = `rgba(148,163,184,${0.38 * (1 - collapse)})`;
          ctx.fillRect(-beamWidth / 2, -beamHeight / 2, beamWidth, beamHeight);
          ctx.fillStyle = `rgba(226,232,240,${0.75 * (1 - collapse)})`;
          ctx.fillRect(-beamWidth / 4, -beamHeight / 2, beamWidth / 2, beamHeight);
          ctx.restore();

          const sparkleCount = 5;
          for (let i = 0; i < sparkleCount; i++) {
            const t = (i + 1) / (sparkleCount + 1);
            const x = canvas.width * t;
            const size = CELL_SIZE * (0.3 - 0.2 * progress);
            if (size <= 0) continue;
            ctx.save();
            ctx.translate(x, centerY);
            ctx.rotate((Math.PI / 2) * progress + i * 0.12);
            ctx.globalAlpha = 0.5 * (1 - progress);
            ctx.fillStyle = '#e0f2fe';
            ctx.fillRect(-size / 2, -size / 2, size, size);
            ctx.restore();
          }

          ctx.save();
          ctx.globalAlpha = 0.35 * fade;
          const inset = CELL_SIZE * 0.45 * progress;
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(0, top + inset, canvas.width, CELL_SIZE - inset * 2);
          ctx.restore();
        });
        ctx.restore();
      }
    }

    function formatStatus(state) {
      if (state.status === 'waiting') return 'Ждём соперника';
      if (state.status === 'countdown') return `Старт через ${state.countdown} сек.`;
      if (state.status === 'running') return 'Игра идёт';
      if (state.status === 'finished') {
        if (!state.winner) return 'Ничья!';
        return state.winner === you ? 'Вы победили!' : 'Вы проиграли';
      }
      return 'Неизвестно';
    }

    function formatTimer(state) {
      if (state.status === 'waiting') return '';
      if (state.status === 'countdown') return '';
      const minutes = Math.floor(state.time_left / 60);
      const seconds = state.time_left % 60;
      return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    function updateScores(players) {
      scoresEl.innerHTML = '';
      for (const pid of Object.keys(players)) {
        const item = document.createElement('li');
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.style.backgroundColor = players[pid].color;
        item.appendChild(badge);
        const label = document.createElement('span');
        label.textContent = pid === you ? 'Вы' : (pid === 'p1' ? 'Игрок 1' : 'Игрок 2');
        const score = document.createElement('span');
        score.textContent = players[pid].score;
        item.appendChild(label);
        item.appendChild(score);
        scoresEl.appendChild(item);
      }
    }

    function updateUI(state) {
      currentState = state;
      if (state.penalty_event && state.penalty_event.id) {
        if (state.penalty_event.id !== lastPenaltyEventId) {
          lastPenaltyEventId = state.penalty_event.id;
          const now = performance.now();
          const pause = typeof state.penalty_event.pause_ms === 'number'
            ? Math.max(state.penalty_event.pause_ms, 0)
            : 0;
          boardResetAnimation = { start: now, delay: pause };
          if (state.penalty_event.highlight_zone) {
            penaltyFlash = { start: now };
            const blinkCells = Array.isArray(state.penalty_event.blink_cells)
              ? state.penalty_event.blink_cells
              : [];
            const duration = pause > 0 ? pause : PENALTY_BLINK_BASE_DURATION;
            if (blinkCells.length > 0) {
              penaltyBlink = {
                start: now,
                cells: blinkCells.map((cell) =>
                  Array.isArray(cell) ? [...cell] : cell
                ),
                duration,
                player: state.penalty_event.player || null,
              };
            } else if (pause > 0) {
              penaltyBlink = {
                start: now,
                cells: [],
                duration,
                player: state.penalty_event.player || null,
              };
            } else {
              penaltyBlink = null;
            }
          } else {
            penaltyBlink = null;
          }
        }
      }
      if (Array.isArray(state.cleared_lines) && state.cleared_lines.length > 0) {
        clearAnimation = { lines: state.cleared_lines, start: performance.now() };
      }
      scheduleRender();
      statusEl.textContent = formatStatus(state);
      timerEl.textContent = formatTimer(state);
      restartButton.disabled = state.status !== 'finished';
      updateScores(state.players);

      const showBoard = state.status === 'running' || state.status === 'finished';
      gameArea.classList.toggle('hidden', !showBoard);

      if (!showBoard) {
        if (state.status === 'waiting') {
          showOverlay('Ожидание соперника', 'Отправьте своему другу ссылку, чтобы начать игру.', { showShare: true });
        } else if (state.status === 'countdown') {
          const seconds = Math.max(state.countdown, 0);
          const text = seconds > 0 ? `Игра начнётся через ${seconds} сек.` : 'Игра начинается!';
          showOverlay('Готовьтесь!', text);
        } else {
          showOverlay('Подключение...', 'Ожидаем состояние комнаты.');
        }
      } else {
        hideOverlay();
      }
    }

    function sendAction(action, data = {}) {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({ action, ...data }));
    }

    const SPEED_FAST = 3;
    const SPEED_SLOW = 1 / 3;
    const speedState = { slow: false, fast: false, current: 1 };

    function updateSpeed() {
      let multiplier = 1;
      if (speedState.fast && !speedState.slow) {
        multiplier = SPEED_FAST;
      } else if (speedState.slow && !speedState.fast) {
        multiplier = SPEED_SLOW;
      }
      if (speedState.current !== multiplier) {
        speedState.current = multiplier;
        sendAction('speed', { multiplier });
      }
    }

    document.addEventListener('keydown', (event) => {
      if (['ArrowLeft', 'ArrowRight', 'ArrowDown', 'ArrowUp', 'Space'].includes(event.code)) {
        event.preventDefault();
      }
      switch (event.code) {
        case 'ArrowLeft':
          sendAction('move', { direction: 'left' });
          break;
        case 'ArrowRight':
          sendAction('move', { direction: 'right' });
          break;
        case 'ArrowDown':
          if (!speedState.fast) {
            speedState.fast = true;
            updateSpeed();
          }
          break;
        case 'ArrowUp':
          if (!speedState.slow) {
            speedState.slow = true;
            updateSpeed();
          }
          break;
        case 'Space':
          sendAction('rotate');
          break;
      }
    });

    document.addEventListener('keyup', (event) => {
      switch (event.code) {
        case 'ArrowDown':
          if (speedState.fast) {
            speedState.fast = false;
            updateSpeed();
          }
          break;
        case 'ArrowUp':
          if (speedState.slow) {
            speedState.slow = false;
            updateSpeed();
          }
          break;
      }
    });

    restartButton.addEventListener('click', () => {
      sendAction('restart');
    });

    connect();
  </script>
</body>
</html>
"""


@app.post("/api/create")
async def api_create() -> JSONResponse:
    room = await manager.create_room()
    return JSONResponse({"room": room.room_id})


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/game/{room_id}")
async def game(room_id: str) -> HTMLResponse:
    room = await manager.get_room(room_id)
    html = GAME_HTML.replace("{room_id}", room.room_id)
    return HTMLResponse(html)


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str) -> None:
    room = await manager.get_room(room_id)
    try:
        pid = await room.connect(websocket)
    except HTTPException:
        return
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            async with room.lock:
                await room.handle_message(pid, payload)
    except WebSocketDisconnect:
        await room.disconnect(websocket, pid)


def main() -> None:
    uvicorn.run(
        "totetris:app",
        host=os.environ.get("TOTETRIS_HOST", "0.0.0.0"),
        port=int(os.environ.get("TOTETRIS_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

