# Badger Brothers Moving — Mock Dataset (SQLite + CSV)

This folder contains a fully synthetic (mock) dataset you can use to build your pipeline end‑to‑end
(database → feature engineering → model training → evaluation → inference). Replace the synthetic
`actual_hours` / `actual_cost` with real job outcomes when you have them.

## Files

- `badger_mock.db` — SQLite database (normalized schema).
- `item_catalog.csv` — the reference list of item codes.
- `moves.csv` — one row per move (inputs + engineered features + targets).
- `move_items.csv` — variable-length inventory list (move_id, item_code, qty).
- `moves_ml_wide.csv` — ML-ready wide table (one row per move, one column per item_code count).

## Schema (SQLite)

Tables:

- `item_catalog(item_code PK, item_name, category, volume_cuft, difficulty, requires_disassembly, is_fragile, is_specialty)`
- `moves(move_id PK, home_sqft, years_lived, num_rooms, num_floors, ... engineered features ..., actual_hours, actual_cost)`
- `move_items(move_id FK, item_code FK, qty, PRIMARY KEY(move_id, item_code))`

## Open the DB in VS Code

Option A (recommended): use the **SQLite** extension (`alexcvzz.vscode-sqlite`).

1. VS Code → Extensions → install “SQLite” by alexcvzz.
2. Command Palette:
   - `SQLite: Open Database` → select `badger_mock.db`.
3. Browse tables and run queries in the SQLite explorer.

Option B: use the `sqlite3` command line

```bash
sqlite3 badger_mock.db
.tables
.schema moves
SELECT * FROM moves LIMIT 5;
```

## Regenerate new synthetic data

Use `generate_mock_data.py` to regenerate the same files with different random values:

```bash
python generate_mock_data.py --n 1000 --seed 123 --out .
```

- `--n` = number of move rows to generate
- `--seed` = random seed for reproducibility
- `--out` = output directory (writes the DB + CSVs there)

## Training tip

Use `moves_ml_wide.csv` for your first model training pass:
- tabular features (sqft/rooms/floors/years + item counts)
- targets: `actual_hours`, `actual_cost`

Later, you can replace the “bag-of-items” columns with embeddings or more detailed item features if you want.
