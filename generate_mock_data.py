#!/usr/bin/env python3
"""
Badger Brothers Moving — Mock data generator

Creates:
- badger_mock.db (SQLite)
- moves.csv
- move_items.csv
- item_catalog.csv
- moves_ml_wide.csv (one-row-per-move, bag-of-items columns)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import os
import random
import sqlite3
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# -----------------------------
# Item catalog (edit as needed)
# -----------------------------
ITEM_CATALOG: List[dict] = [
    {
        "item_code": "BOX_S",
        "item_name": "Small box",
        "category": "box",
        "volume_cuft": 1.5,
        "difficulty": 1,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "BOX_M",
        "item_name": "Medium box",
        "category": "box",
        "volume_cuft": 3.0,
        "difficulty": 1,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "BOX_L",
        "item_name": "Large box",
        "category": "box",
        "volume_cuft": 4.5,
        "difficulty": 1,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "BOX_WARD",
        "item_name": "Wardrobe box",
        "category": "box",
        "volume_cuft": 8.0,
        "difficulty": 2,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "SOFA_2S",
        "item_name": "Loveseat / 2-seat sofa",
        "category": "furniture",
        "volume_cuft": 45.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "SOFA_3S",
        "item_name": "3-seat sofa",
        "category": "furniture",
        "volume_cuft": 60.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "SECTIONAL",
        "item_name": "Sectional sofa",
        "category": "furniture",
        "volume_cuft": 95.0,
        "difficulty": 4,
        "requires_disassembly": 1,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "COFFEE_TBL",
        "item_name": "Coffee table",
        "category": "furniture",
        "volume_cuft": 12.0,
        "difficulty": 2,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "END_TBL",
        "item_name": "End table",
        "category": "furniture",
        "volume_cuft": 6.0,
        "difficulty": 2,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "TV_40",
        "item_name": "TV (up to ~40\")",
        "category": "electronics",
        "volume_cuft": 6.0,
        "difficulty": 2,
        "requires_disassembly": 0,
        "is_fragile": 1,
        "is_specialty": 0
    },
    {
        "item_code": "TV_60",
        "item_name": "TV (~41\u201365\")",
        "category": "electronics",
        "volume_cuft": 9.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 1,
        "is_specialty": 0
    },
    {
        "item_code": "TV_75",
        "item_name": "TV (66\"+)",
        "category": "electronics",
        "volume_cuft": 14.0,
        "difficulty": 4,
        "requires_disassembly": 0,
        "is_fragile": 1,
        "is_specialty": 0
    },
    {
        "item_code": "DINING_TBL",
        "item_name": "Dining table",
        "category": "furniture",
        "volume_cuft": 40.0,
        "difficulty": 3,
        "requires_disassembly": 1,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "DINING_CHAIR",
        "item_name": "Dining chair",
        "category": "furniture",
        "volume_cuft": 6.0,
        "difficulty": 1,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "BAR_STOOL",
        "item_name": "Bar stool",
        "category": "furniture",
        "volume_cuft": 4.0,
        "difficulty": 1,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "MATTRESS",
        "item_name": "Mattress",
        "category": "furniture",
        "volume_cuft": 35.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "BOX_SPRING",
        "item_name": "Box spring",
        "category": "furniture",
        "volume_cuft": 25.0,
        "difficulty": 2,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "BED_FRAME",
        "item_name": "Bed frame",
        "category": "furniture",
        "volume_cuft": 25.0,
        "difficulty": 3,
        "requires_disassembly": 1,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "DRESSER",
        "item_name": "Dresser",
        "category": "furniture",
        "volume_cuft": 30.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "NIGHTSTAND",
        "item_name": "Nightstand",
        "category": "furniture",
        "volume_cuft": 6.0,
        "difficulty": 1,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "BOOKCASE",
        "item_name": "Bookcase",
        "category": "furniture",
        "volume_cuft": 18.0,
        "difficulty": 2,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "DESK",
        "item_name": "Desk",
        "category": "furniture",
        "volume_cuft": 22.0,
        "difficulty": 3,
        "requires_disassembly": 1,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "OFFICE_CHAIR",
        "item_name": "Office chair",
        "category": "furniture",
        "volume_cuft": 10.0,
        "difficulty": 2,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "FILE_CAB",
        "item_name": "File cabinet",
        "category": "furniture",
        "volume_cuft": 12.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "FRIDGE",
        "item_name": "Refrigerator",
        "category": "appliance",
        "volume_cuft": 32.0,
        "difficulty": 4,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "WASHER",
        "item_name": "Washer",
        "category": "appliance",
        "volume_cuft": 25.0,
        "difficulty": 4,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "DRYER",
        "item_name": "Dryer",
        "category": "appliance",
        "volume_cuft": 25.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "DISHWASHER",
        "item_name": "Dishwasher",
        "category": "appliance",
        "volume_cuft": 18.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "MIRROR_L",
        "item_name": "Large mirror",
        "category": "fragile",
        "volume_cuft": 8.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 1,
        "is_specialty": 0
    },
    {
        "item_code": "ART_L",
        "item_name": "Large framed art",
        "category": "fragile",
        "volume_cuft": 6.0,
        "difficulty": 3,
        "requires_disassembly": 0,
        "is_fragile": 1,
        "is_specialty": 0
    },
    {
        "item_code": "LAMP",
        "item_name": "Lamp",
        "category": "decor",
        "volume_cuft": 3.0,
        "difficulty": 1,
        "requires_disassembly": 0,
        "is_fragile": 1,
        "is_specialty": 0
    },
    {
        "item_code": "RUG",
        "item_name": "Rug",
        "category": "decor",
        "volume_cuft": 6.0,
        "difficulty": 1,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 0
    },
    {
        "item_code": "PIANO_UP",
        "item_name": "Upright piano",
        "category": "specialty",
        "volume_cuft": 60.0,
        "difficulty": 5,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 1
    },
    {
        "item_code": "SAFE_SM",
        "item_name": "Safe (small)",
        "category": "specialty",
        "volume_cuft": 18.0,
        "difficulty": 5,
        "requires_disassembly": 0,
        "is_fragile": 0,
        "is_specialty": 1
    },
    {
        "item_code": "TREADMILL",
        "item_name": "Treadmill",
        "category": "specialty",
        "volume_cuft": 30.0,
        "difficulty": 5,
        "requires_disassembly": 1,
        "is_fragile": 0,
        "is_specialty": 1
    }
]

ITEM_BY_CODE = {it["item_code"]: it for it in ITEM_CATALOG}
BOX_CODES = ["BOX_S", "BOX_M", "BOX_L", "BOX_WARD"]


# -----------------------------
# Random feature generation
# -----------------------------
def draw_home_sqft(rng: random.Random) -> int:
    # Triangular distribution: low 350, high 3500, mode 1200
    return int(rng.triangular(350, 3500, 1200))


def draw_years_lived(rng: random.Random) -> float:
    # Gamma-like: more short stays, some long stays
    y = rng.gammavariate(2.0, 3.0)  # mean ~6
    # sprinkle in some longer tenures
    if rng.random() < 0.08:
        y += rng.uniform(8, 18)
    y = min(max(y, 0.0), 30.0)
    return round(y, 1)


def draw_num_rooms(rng: random.Random, sqft: int) -> int:
    base = sqft / 420.0 + rng.uniform(-0.6, 0.6)
    rooms = int(round(base))
    return int(min(max(rooms, 1), 10))


def draw_num_floors(rng: random.Random) -> int:
    # 1-4 floors distribution
    r = rng.random()
    if r < 0.62:
        return 1
    if r < 0.90:
        return 2
    if r < 0.98:
        return 3
    return 4


def allocate_boxes(total_boxes: int, rng: random.Random) -> Dict[str, int]:
    # Base proportions: small/medium/large/wardrobe
    weights = [0.25, 0.45, 0.25, 0.05]
    # jitter proportions slightly to avoid identical patterns
    jitter = rng.uniform(-0.03, 0.03)
    weights = [
        max(0.01, weights[0] + jitter),
        max(0.01, weights[1] + jitter),
        max(0.01, weights[2] + jitter),
        max(0.01, weights[3] + jitter * 0.5),
    ]
    s = sum(weights)
    weights = [w / s for w in weights]
    alloc = rng.choices(BOX_CODES, weights=weights, k=total_boxes)
    counts = Counter(alloc)
    return {c: int(counts.get(c, 0)) for c in BOX_CODES}


def generate_move_items(
    sqft: int, years: float, rooms: int, floors: int, rng: random.Random
) -> Dict[str, int]:
    items: Dict[str, int] = defaultdict(int)

    # Boxes
    boxes_total = int(round((sqft / 18.0) + (years * 2.2) + (rooms * 3.0) + rng.gauss(0, 10)))
    boxes_total = max(10, min(boxes_total, 260))
    box_alloc = allocate_boxes(boxes_total, rng)
    for code, qty in box_alloc.items():
        if qty > 0:
            items[code] += qty

    # Bedrooms heuristic
    bedrooms = int(round(rooms * 0.45 + rng.uniform(-0.3, 0.8)))
    bedrooms = min(max(bedrooms, 1), 5)

    items["MATTRESS"] += bedrooms
    items["BED_FRAME"] += bedrooms
    if rng.random() < 0.85:
        items["BOX_SPRING"] += max(0, bedrooms - (1 if rng.random() < 0.4 else 0))

    items["DRESSER"] += max(1, int(round(bedrooms + rng.uniform(-0.2, 0.8))))
    items["NIGHTSTAND"] += int(round(bedrooms * 2 + rng.uniform(-1, 1)))

    # Living room
    items["COFFEE_TBL"] += 1 if rng.random() < 0.9 else 0
    items["END_TBL"] += int(round(1 + rng.random() * 2))
    if rooms >= 2:
        items["SOFA_3S"] += 1
        if rng.random() < 0.45:
            items["SOFA_2S"] += 1
        if sqft > 1600 and rng.random() < 0.18:
            items["SECTIONAL"] += 1

    # Dining
    if rooms >= 3 and rng.random() < 0.75:
        items["DINING_TBL"] += 1
        chair_count = int(round(rng.triangular(4, 10, 6)))
        items["DINING_CHAIR"] += chair_count
        if rng.random() < 0.35:
            items["BAR_STOOL"] += int(round(rng.triangular(2, 6, 3)))

    # Office/storage
    items["BOOKCASE"] += int(round(rng.triangular(0, 4, 1)))
    if rng.random() < 0.65:
        items["DESK"] += int(round(rng.triangular(0, 2, 1)))
        items["OFFICE_CHAIR"] += int(round(rng.triangular(0, 2, 1)))
    if rng.random() < 0.25:
        items["FILE_CAB"] += 1

    # Appliances
    if sqft > 450:
        items["FRIDGE"] += 1
    if rooms >= 3 and rng.random() < 0.80:
        items["WASHER"] += 1
        items["DRYER"] += 1
    if sqft > 900 and rng.random() < 0.65:
        items["DISHWASHER"] += 1

    # TVs
    tv_count = int(round(rng.triangular(1, 5, 2))) if rooms > 1 else 1
    for _ in range(tv_count):
        code = rng.choices(["TV_40", "TV_60", "TV_75"], weights=[0.45, 0.45, 0.10])[0]
        items[code] += 1

    # Fragile decor
    if rng.random() < 0.55:
        items["MIRROR_L"] += 1
    if rng.random() < 0.40:
        items["ART_L"] += int(round(rng.triangular(1, 4, 2)))
    items["LAMP"] += int(round(rng.triangular(1, 8, 3)))
    items["RUG"] += int(round(rng.triangular(0, 6, 2)))

    # Specialty
    if rng.random() < 0.04:
        items["PIANO_UP"] += 1
    if rng.random() < 0.08:
        items["SAFE_SM"] += 1
    if rng.random() < 0.12:
        items["TREADMILL"] += 1

    # Drop zeros
    return {k: int(v) for k, v in items.items() if v and v > 0}


def compute_features(items: Dict[str, int]) -> Dict[str, float]:
    total_items = 0
    total_volume = 0.0
    fragile = 0
    bulky = 0
    disassembly = 0
    specialty = 0
    difficulty_weighted = 0.0

    for code, qty in items.items():
        it = ITEM_BY_CODE[code]
        total_items += qty
        total_volume += float(it["volume_cuft"]) * qty
        fragile += qty * int(it["is_fragile"])
        disassembly += qty * int(it["requires_disassembly"])
        specialty += qty * int(it["is_specialty"])
        if float(it["volume_cuft"]) >= 25.0:
            bulky += qty
        difficulty_weighted += int(it["difficulty"]) * qty

    avg_difficulty = difficulty_weighted / total_items if total_items else 0.0

    return {
        "total_items": int(total_items),
        "total_volume_cuft": round(total_volume, 1),
        "fragile_items": int(fragile),
        "bulky_items": int(bulky),
        "disassembly_items": int(disassembly),
        "specialty_items": int(specialty),
        "avg_difficulty": round(avg_difficulty, 3),
    }


def choose_crew_size(sqft: int, total_volume: float, total_items: int, floors: int) -> int:
    crew = 2
    if sqft > 1200 or total_volume > 800 or total_items > 95:
        crew += 1
    if sqft > 2200 or total_volume > 1200 or total_items > 140:
        crew += 1
    if sqft > 3000 or total_volume > 1600:
        crew += 1
    if floors >= 3:
        crew += 1
    return int(min(max(crew, 2), 5))


def generate_targets(
    sqft: int, floors: int, feats: Dict[str, float], crew: int, rng: random.Random
) -> Tuple[float, float, float, float]:
    total_items = int(feats["total_items"])
    fragile = int(feats["fragile_items"])
    disassembly = int(feats["disassembly_items"])
    specialty = int(feats["specialty_items"])
    bulky = int(feats["bulky_items"])

    # crew-hours baseline (synthetic)
    work_units = (
        1.5
        + 0.0015 * sqft
        + 0.028 * total_items
        + 0.05 * fragile
        + 0.08 * disassembly
        + 0.18 * specialty
        + 0.015 * bulky
        + 0.35 * max(floors - 1, 0)
    )
    noise = math.exp(rng.gauss(0, 0.18))
    crew_hours = work_units * noise

    # Clock-hours (what a customer often cares about)
    duration_hours = crew_hours / crew + 0.30  # overhead
    duration_hours = max(1.0, min(duration_hours, 16.0))
    duration_hours = round(duration_hours, 2)

    # Pricing knobs (synthetic)
    hourly_rate_per_mover = round(rng.uniform(50, 85), 2)
    base_trip_fee = round(rng.uniform(0, 150), 2)

    specialty_fee = 50.0 * specialty
    disassembly_fee = 15.0 * disassembly
    fragile_fee = 2.0 * fragile

    cost = duration_hours * crew * hourly_rate_per_mover + base_trip_fee + specialty_fee + disassembly_fee + fragile_fee
    cost *= (1.0 + rng.gauss(0, 0.07))
    cost = max(200.0, cost)
    cost = round(cost, 2)

    return duration_hours, cost, hourly_rate_per_mover, base_trip_fee


# -----------------------------
# IO helpers
# -----------------------------
def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def create_sqlite_db(db_path: str, item_catalog: List[dict], moves_rows: List[dict], move_items_rows: List[dict]) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS move_items;
        DROP TABLE IF EXISTS moves;
        DROP TABLE IF EXISTS item_catalog;

        CREATE TABLE item_catalog (
            item_code TEXT PRIMARY KEY,
            item_name TEXT NOT NULL,
            category TEXT NOT NULL,
            volume_cuft REAL NOT NULL,
            difficulty INTEGER NOT NULL,
            requires_disassembly INTEGER NOT NULL CHECK (requires_disassembly IN (0,1)),
            is_fragile INTEGER NOT NULL CHECK (is_fragile IN (0,1)),
            is_specialty INTEGER NOT NULL CHECK (is_specialty IN (0,1))
        );

        CREATE TABLE moves (
            move_id INTEGER PRIMARY KEY,
            home_sqft INTEGER NOT NULL,
            years_lived REAL NOT NULL,
            num_rooms INTEGER NOT NULL,
            num_floors INTEGER NOT NULL,
            crew_size INTEGER NOT NULL,

            total_items INTEGER NOT NULL,
            total_volume_cuft REAL NOT NULL,
            fragile_items INTEGER NOT NULL,
            bulky_items INTEGER NOT NULL,
            disassembly_items INTEGER NOT NULL,
            specialty_items INTEGER NOT NULL,
            avg_difficulty REAL NOT NULL,

            hourly_rate_per_mover REAL NOT NULL,
            base_trip_fee REAL NOT NULL,

            actual_hours REAL NOT NULL,
            actual_cost REAL NOT NULL,

            created_at TEXT NOT NULL
        );

        CREATE TABLE move_items (
            move_id INTEGER NOT NULL,
            item_code TEXT NOT NULL,
            qty INTEGER NOT NULL CHECK (qty >= 0),
            PRIMARY KEY (move_id, item_code),
            FOREIGN KEY (move_id) REFERENCES moves(move_id) ON DELETE CASCADE,
            FOREIGN KEY (item_code) REFERENCES item_catalog(item_code)
        );

        CREATE INDEX idx_move_items_move_id ON move_items(move_id);
        CREATE INDEX idx_move_items_item_code ON move_items(item_code);
        """
    )

    cur.executemany(
        """
        INSERT INTO item_catalog
            (item_code, item_name, category, volume_cuft, difficulty, requires_disassembly, is_fragile, is_specialty)
        VALUES
            (:item_code, :item_name, :category, :volume_cuft, :difficulty, :requires_disassembly, :is_fragile, :is_specialty);
        """,
        item_catalog,
    )

    cur.executemany(
        """
        INSERT INTO moves
            (move_id, home_sqft, years_lived, num_rooms, num_floors, crew_size,
             total_items, total_volume_cuft, fragile_items, bulky_items, disassembly_items, specialty_items, avg_difficulty,
             hourly_rate_per_mover, base_trip_fee,
             actual_hours, actual_cost, created_at)
        VALUES
            (:move_id, :home_sqft, :years_lived, :num_rooms, :num_floors, :crew_size,
             :total_items, :total_volume_cuft, :fragile_items, :bulky_items, :disassembly_items, :specialty_items, :avg_difficulty,
             :hourly_rate_per_mover, :base_trip_fee,
             :actual_hours, :actual_cost, :created_at);
        """,
        moves_rows,
    )

    cur.executemany(
        """
        INSERT INTO move_items (move_id, item_code, qty)
        VALUES (:move_id, :item_code, :qty);
        """,
        move_items_rows,
    )

    conn.commit()
    conn.close()


# -----------------------------
# Main generator
# -----------------------------
def generate_dataset(n: int, seed: int) -> Tuple[List[dict], List[dict], List[dict]]:
    rng = random.Random(seed)

    moves_rows: List[dict] = []
    move_items_rows: List[dict] = []
    wide_rows: List[dict] = []

    all_item_codes = [it["item_code"] for it in ITEM_CATALOG]
    created_at = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    items_by_move: Dict[int, Dict[str, int]] = defaultdict(dict)

    for move_id in range(1, n + 1):
        sqft = draw_home_sqft(rng)
        years = draw_years_lived(rng)
        rooms = draw_num_rooms(rng, sqft)
        floors = draw_num_floors(rng)

        items = generate_move_items(sqft, years, rooms, floors, rng)
        feats = compute_features(items)
        crew = choose_crew_size(sqft, float(feats["total_volume_cuft"]), int(feats["total_items"]), floors)
        actual_hours, actual_cost, rate_per_mover, base_trip_fee = generate_targets(sqft, floors, feats, crew, rng)

        move_row = {
            "move_id": move_id,
            "home_sqft": sqft,
            "years_lived": years,
            "num_rooms": rooms,
            "num_floors": floors,
            "crew_size": crew,

            "total_items": int(feats["total_items"]),
            "total_volume_cuft": float(feats["total_volume_cuft"]),
            "fragile_items": int(feats["fragile_items"]),
            "bulky_items": int(feats["bulky_items"]),
            "disassembly_items": int(feats["disassembly_items"]),
            "specialty_items": int(feats["specialty_items"]),
            "avg_difficulty": float(feats["avg_difficulty"]),

            "hourly_rate_per_mover": float(rate_per_mover),
            "base_trip_fee": float(base_trip_fee),

            "actual_hours": float(actual_hours),
            "actual_cost": float(actual_cost),

            "created_at": created_at,
        }
        moves_rows.append(move_row)

        for code, qty in items.items():
            move_items_rows.append({"move_id": move_id, "item_code": code, "qty": int(qty)})
            items_by_move[move_id][code] = int(qty)

        # Wide row (one row per move)
        wide = {
            "move_id": move_id,
            "home_sqft": sqft,
            "years_lived": years,
            "num_rooms": rooms,
            "num_floors": floors,

            "crew_size": crew,
            "total_items": int(feats["total_items"]),
            "total_volume_cuft": float(feats["total_volume_cuft"]),
            "fragile_items": int(feats["fragile_items"]),
            "bulky_items": int(feats["bulky_items"]),
            "disassembly_items": int(feats["disassembly_items"]),
            "specialty_items": int(feats["specialty_items"]),
            "avg_difficulty": float(feats["avg_difficulty"]),

            "actual_hours": float(actual_hours),
            "actual_cost": float(actual_cost),
        }

        for code in all_item_codes:
            wide[code] = items_by_move[move_id].get(code, 0)

        wide_rows.append(wide)

    return moves_rows, move_items_rows, wide_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of mock moves to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    db_path = os.path.join(args.out, "badger_mock.db")
    moves_csv = os.path.join(args.out, "moves.csv")
    move_items_csv = os.path.join(args.out, "move_items.csv")
    item_catalog_csv = os.path.join(args.out, "item_catalog.csv")
    wide_csv = os.path.join(args.out, "moves_ml_wide.csv")

    moves_rows, move_items_rows, wide_rows = generate_dataset(args.n, args.seed)

    # Write CSVs
    moves_fields = [
        "move_id","home_sqft","years_lived","num_rooms","num_floors","crew_size",
        "total_items","total_volume_cuft","fragile_items","bulky_items","disassembly_items","specialty_items","avg_difficulty",
        "hourly_rate_per_mover","base_trip_fee","actual_hours","actual_cost","created_at"
    ]
    write_csv(moves_csv, moves_rows, moves_fields)

    write_csv(move_items_csv, move_items_rows, ["move_id","item_code","qty"])

    item_fields = ["item_code","item_name","category","volume_cuft","difficulty","requires_disassembly","is_fragile","is_specialty"]
    write_csv(item_catalog_csv, ITEM_CATALOG, item_fields)

    wide_fields = [
        "move_id","home_sqft","years_lived","num_rooms","num_floors",
        "crew_size","total_items","total_volume_cuft","fragile_items","bulky_items","disassembly_items","specialty_items","avg_difficulty",
        "actual_hours","actual_cost"
    ] + [it["item_code"] for it in ITEM_CATALOG]
    write_csv(wide_csv, wide_rows, wide_fields)

    # Write SQLite DB
    create_sqlite_db(db_path, ITEM_CATALOG, moves_rows, move_items_rows)

    print("Wrote:")
    print(" -", db_path)
    print(" -", moves_csv)
    print(" -", move_items_csv)
    print(" -", item_catalog_csv)
    print(" -", wide_csv)


if __name__ == "__main__":
    main()
