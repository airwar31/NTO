"""
AstroClassify - CatBoost Gradient Boosted Decision Trees Solution
National Technological Olympiad (NTO) - Space Object Classification
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from catboost import CatBoostClassifier, Pool

SEED = 65
N_SPLITS = 5
LR = 0.03
DEPTH = 6
ITERATIONS = 1200
ES_ROUNDS = 60
L2_REG = 3.0
RUN_NAME = "catboost_cv"


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)


def detect_columns(df: pd.DataFrame):
    target_candidates = ["target", "type", "label", "y"]
    id_candidates = ["object_id", "idx", "id"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError("Could not detect target column.")
    return target_col, id_col


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Astrophysical feature engineering for tabular photometric data."""
    df = df.copy()

    # Color indices
    if 'u_mag' in df.columns and 'g_mag' in df.columns:
        df['color_u_g'] = df['u_mag'] - df['g_mag']
    if 'g_mag' in df.columns and 'r_mag' in df.columns:
        df['color_g_r'] = df['g_mag'] - df['r_mag']
    if 'r_mag' in df.columns and 'i_mag' in df.columns:
        df['color_r_i'] = df['r_mag'] - df['i_mag']
    if 'i_mag' in df.columns and 'z_mag' in df.columns:
        df['color_i_z'] = df['i_mag'] - df['z_mag']

    # Total apparent flux statistics
    mag_cols = [c for c in df.columns if c.endswith('_mag')]
    if len(mag_cols) > 0:
        df['total_mag'] = df[mag_cols].mean(axis=1)
        df['mag_span'] = df[mag_cols].max(axis=1) - df[mag_cols].min(axis=1)

    # Proper motion vector magnitude
    if 'pm_ra' in df.columns and 'pm_dec' in df.columns:
        df['proper_motion_magnitude'] = np.sqrt(df['pm_ra'] ** 2 + df['pm_dec'] ** 2)

    # Signal-to-Noise Ratio summary
    snr_cols = [c for c in df.columns if c.startswith('snr_')]
    if len(snr_cols) > 0:
        df['total_snr'] = df[snr_cols].sum(axis=1)
        df['avg_snr'] = df[snr_cols].mean(axis=1)

    return df


def get_feature_lists(df: pd.DataFrame, target_col: str, id_col: Optional[str]):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in [target_col, id_col]:
        if col in num_cols:
            num_cols.remove(col)
        if col in cat_cols:
            cat_cols.remove(col)
    return num_cols, cat_cols


def run_catboost_pipeline(train_path="train.csv", test_path="test.csv", iterations=ITERATIONS, lr=LR):
    set_seed()

    print("=" * 60)
    print("  AstroClassify: CatBoost Gradient Boosted Trees Pipeline")
    print(f"  Iterations: {iterations} | Learning Rate: {lr} | Depth: {DEPTH}")
    print("=" * 60)

    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    target_col, id_col = detect_columns(df)

    # Feature engineering
    print("Extracting domain features...")
    df = engineer_features(df)
    if test_df is not None:
        test_df = engineer_features(test_df)

    num_cols, cat_cols = get_feature_lists(df, target_col, id_col)
    for c in cat_cols:
        df[c] = df[c].astype("category")
        if test_df is not None and c in test_df.columns:
            test_df[c] = test_df[c].astype("category")

    feature_cols = [c for c in df.columns if c not in {target_col, id_col}]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    X_test = test_df[feature_cols].copy() if test_df is not None else None

    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    classes = sorted(y.unique())

    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"Target classes ({len(classes)}): {classes}")

    params = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        learning_rate=lr,
        depth=DEPTH,
        l2_leaf_reg=L2_REG,
        random_strength=0.8,
        bagging_temperature=0.2,
        auto_class_weights="Balanced",
        iterations=iterations,
        early_stopping_rounds=ES_ROUNDS,
        allow_writing_files=False,
        thread_count=-1,
        random_state=SEED,
        verbose=100
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_pred = np.empty(len(X), dtype=object)
    fold_models: List[CatBoostClassifier] = []
    fold_scores = []
    test_preds = []
    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n=== FOLD {fold + 1}/{N_SPLITS} ===")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_idx)
        val_pool = Pool(X_val, y_val, cat_features=cat_idx)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        val_pred = model.predict(X_val)
        if val_pred.ndim > 1:
            val_pred = val_pred.flatten()
        oof_pred[val_idx] = val_pred

        fold_f1 = f1_score(y_val, val_pred, average='macro')
        fold_scores.append(fold_f1)
        fold_models.append(model)
        print(f"Fold {fold + 1} Macro-F1: {fold_f1:.4f}")

        if X_test is not None:
            t_pred = model.predict(X_test)
            if t_pred.ndim > 1:
                t_pred = t_pred.flatten()
            test_preds.append(t_pred)

        del X_train, X_val, y_train, y_val, train_pool, val_pool
        gc.collect()

    overall_f1 = f1_score(y, oof_pred, average='macro')
    print("\n" + "=" * 60)
    print(f"  CROSS-VALIDATION RESULTS: Mean Macro-F1 = {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print(f"  Overall Out-of-Fold Macro-F1 = {overall_f1:.4f}")
    print(f"  Total Training Time: {time.time() - start_time:.1f}s")
    print("=" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y, oof_pred, digits=4))

    # Feature Importance
    last_model = fold_models[-1]
    importances = last_model.get_feature_importance()
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
    feat_imp.to_csv("catboost_feature_importance.csv", index=False)
    print("\nTop 10 Most Influential Features:")
    print(feat_imp.head(10).to_string(index=False))

    # Submission
    if test_df is not None and len(test_preds) > 0:
        final_test_pred = []
        for i in range(len(X_test)):
            votes = [pred[i] for pred in test_preds]
            majority = max(set(votes), key=votes.count)
            final_test_pred.append(majority)

        submission = pd.DataFrame()
        submission[id_col] = test_df[id_col]
        for class_name in classes:
            submission[class_name] = (np.array(final_test_pred) == class_name).astype(int)

        sub_path = f"{RUN_NAME}_submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"\nSubmission saved to: {sub_path}")

    return overall_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CatBoost on Space Object Dataset")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="Number of boosting iterations")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    args = parser.parse_args()

    run_catboost_pipeline(iterations=args.iterations, lr=args.lr)
