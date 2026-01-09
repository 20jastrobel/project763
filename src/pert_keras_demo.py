#!/usr/bin/env python3
"""
Standalone demo: train a Keras regression model and compare PERT vs Gaussian estimates.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_KEY_FEATURES = [
    "home_sqft",
    "total_volume_cuft",
    "num_rooms",
    "bulky_items",
]


def load_from_sqlite(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        moves = pd.read_sql_query("SELECT * FROM moves", conn)
        move_items = pd.read_sql_query(
            "SELECT move_id, item_code, qty FROM move_items", conn
        )
    finally:
        conn.close()

    if move_items.empty:
        return moves

    item_counts = (
        move_items.pivot_table(
            index="move_id", columns="item_code", values="qty", aggfunc="sum", fill_value=0
        )
        .reset_index()
    )
    return moves.merge(item_counts, on="move_id", how="left").fillna(0)


def load_from_csvs(moves_csv: Path, move_items_csv: Path) -> pd.DataFrame:
    moves = pd.read_csv(moves_csv)
    move_items = pd.read_csv(move_items_csv)
    if move_items.empty:
        return moves
    item_counts = (
        move_items.pivot_table(
            index="move_id", columns="item_code", values="qty", aggfunc="sum", fill_value=0
        )
        .reset_index()
    )
    return moves.merge(item_counts, on="move_id", how="left").fillna(0)


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    data_source = args.data_source
    db_path = Path(args.db_path)
    wide_csv_path = Path(args.wide_csv_path)
    moves_csv_path = Path(args.moves_csv_path)
    move_items_csv_path = Path(args.move_items_csv_path)

    if data_source == "db" or (data_source == "auto" and db_path.exists()):
        return load_from_sqlite(db_path)
    if data_source == "wide_csv" or (data_source == "auto" and wide_csv_path.exists()):
        return pd.read_csv(wide_csv_path)
    if data_source == "csv_join" or (
        data_source == "auto"
        and moves_csv_path.exists()
        and move_items_csv_path.exists()
    ):
        return load_from_csvs(moves_csv_path, move_items_csv_path)

    raise FileNotFoundError("No suitable data source found.")


def prepare_dataset(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    for col in ("move_id", "created_at"):
        if col in df.columns:
            df = df.drop(columns=[col])

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    target = df[target_column].values.astype(np.float32)
    feature_df = df.drop(columns=[target_column])
    feature_df = feature_df.select_dtypes(include=[np.number]).fillna(0)
    return feature_df, target


def build_model(input_dim: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def pert_expected_value(optimistic: float, most_likely: float, pessimistic: float) -> float:
    return (optimistic + 4 * most_likely + pessimistic) / 6.0


def pert_sample(optimistic: float, most_likely: float, pessimistic: float, size: int) -> np.ndarray:
    if pessimistic <= optimistic:
        return np.full(size, most_likely, dtype=np.float32)
    lam = 4.0
    alpha = 1.0 + lam * (most_likely - optimistic) / (pessimistic - optimistic)
    beta = 1.0 + lam * (pessimistic - most_likely) / (pessimistic - optimistic)
    return optimistic + np.random.beta(alpha, beta, size=size) * (pessimistic - optimistic)


def adjust_row(
    row: pd.Series, key_features: list[str], pct: float, direction: str
) -> pd.Series:
    adjusted = row.copy()
    for feature in key_features:
        if feature not in adjusted.index:
            continue
        base = float(adjusted[feature])
        delta = abs(base) * pct
        if direction == "optimistic":
            adjusted[feature] = max(0.0, base - delta)
        elif direction == "pessimistic":
            adjusted[feature] = base + delta
    return adjusted


def predict_row(
    model: tf.keras.Model,
    scaler: StandardScaler,
    feature_cols: list[str],
    row: pd.Series,
) -> float:
    row_df = pd.DataFrame([row[feature_cols]])
    scaled = scaler.transform(row_df.values)
    return float(model.predict(scaled, verbose=0).flatten()[0])


def monte_carlo_predictions(
    model: tf.keras.Model,
    scaler: StandardScaler,
    feature_cols: list[str],
    base_row: pd.Series,
    key_features: list[str],
    pct: float,
    samples: int,
) -> np.ndarray:
    base_values = base_row[feature_cols].astype(float).values
    rows = np.repeat(base_values[None, :], samples, axis=0)
    col_index = {col: idx for idx, col in enumerate(feature_cols)}

    for feature in key_features:
        if feature not in col_index:
            continue
        idx = col_index[feature]
        most_likely = float(base_row[feature])
        delta = abs(most_likely) * pct
        optimistic = max(0.0, most_likely - delta)
        pessimistic = most_likely + delta
        rows[:, idx] = pert_sample(optimistic, most_likely, pessimistic, samples)

    scaled = scaler.transform(rows)
    return model.predict(scaled, verbose=0).flatten()


def parse_key_features(raw: str, feature_cols: list[str]) -> list[str]:
    if raw:
        selected = [feature.strip() for feature in raw.split(",") if feature.strip()]
    else:
        selected = [feature for feature in DEFAULT_KEY_FEATURES if feature in feature_cols]
    if not selected:
        selected = feature_cols[:3]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Keras model and compare PERT vs Gaussian estimates."
    )
    parser.add_argument(
        "--data_source",
        choices=["auto", "db", "wide_csv", "csv_join"],
        default="auto",
        help="Choose which data source to use.",
    )
    parser.add_argument("--db_path", default="badger_mock.db", help="SQLite DB path.")
    parser.add_argument(
        "--wide_csv_path", default="moves_ml_wide.csv", help="Wide CSV path."
    )
    parser.add_argument("--moves_csv_path", default="moves.csv", help="Moves CSV path.")
    parser.add_argument(
        "--move_items_csv_path", default="move_items.csv", help="Move items CSV path."
    )
    parser.add_argument(
        "--target_column", default="actual_cost", help="Target column for prediction."
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--pert_margin",
        type=float,
        default=0.1,
        help="Percent margin to derive optimistic/pessimistic values.",
    )
    parser.add_argument(
        "--key_features",
        default="",
        help="Comma-separated key features for PERT modeling.",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=200,
        help="Monte Carlo samples (0 to skip).",
    )
    parser.add_argument(
        "--model_output",
        default="models/pert_estimator_model.keras",
        help="Output path for the saved model.",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = load_data(args)
    print(f"Loaded data shape: {df.shape}")

    feature_df, target = prepare_dataset(df, args.target_column)
    feature_cols = feature_df.columns.tolist()
    key_features = parse_key_features(args.key_features, feature_cols)
    print(f"Using key features for PERT: {key_features}")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        feature_df, target, test_size=args.test_size, random_state=args.seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values)

    model = build_model(X_train.shape[1], args.learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {loss:.4f} | Test MAE: {mae:.4f}")

    sample_count = min(5, len(X_test))
    sample_preds = model.predict(X_test[:sample_count], verbose=0).flatten()
    for idx in range(sample_count):
        print(
            f"Sample {idx + 1} - Predicted: {sample_preds[idx]:.2f} | Actual: {y_test[idx]:.2f}"
        )

    example_row = X_test_df.iloc[0]
    optimistic_row = adjust_row(example_row, key_features, args.pert_margin, "optimistic")
    pessimistic_row = adjust_row(example_row, key_features, args.pert_margin, "pessimistic")

    o_pred = predict_row(model, scaler, feature_cols, optimistic_row)
    m_pred = predict_row(model, scaler, feature_cols, example_row)
    p_pred = predict_row(model, scaler, feature_cols, pessimistic_row)

    pert_estimate = pert_expected_value(o_pred, m_pred, p_pred)
    gaussian_estimate = (o_pred + p_pred) / 2.0

    print("\nPERT vs Gaussian estimate (single example):")
    print(f"Optimistic prediction: {o_pred:.2f}")
    print(f"Most likely prediction: {m_pred:.2f}")
    print(f"Pessimistic prediction: {p_pred:.2f}")
    print(f"PERT weighted estimate: {pert_estimate:.2f}")
    print(f"Gaussian midpoint estimate: {gaussian_estimate:.2f}")

    if args.mc_samples > 0:
        mc_preds = monte_carlo_predictions(
            model,
            scaler,
            feature_cols,
            example_row,
            key_features,
            args.pert_margin,
            args.mc_samples,
        )
        p05, p95 = np.percentile(mc_preds, [5, 95])
        print("\nMonte Carlo (PERT samples):")
        print(f"Mean: {mc_preds.mean():.2f} | 5th pct: {p05:.2f} | 95th pct: {p95:.2f}")

    output_path = Path(args.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nSaved model to {output_path} (Keras load_model ready).")


if __name__ == "__main__":
    main()
