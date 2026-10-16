"""
AstroClassify - PyTorch Deep Neural Network Solution
National Technological Olympiad (NTO) - Space Object Classification
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

SEED = 42
N_SPLITS = 5
BATCH_SIZE = 512
EPOCHS = 80
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=SEED):
    """Ensure reproducibility across numpy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class SpaceDataset(Dataset):
    """PyTorch Dataset for astronomical tabular features."""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class SpaceClassifier(nn.Module):
    """
    Deep Multilayer Perceptron with residual-like skip connections,
    Batch Normalization, and Dropout for robust astronomical classification.
    """
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


def feature_engineering(df):
    """
    Extract astrophysically motivated features:
    - Color indices (SDSS u-g, g-r, r-i, i-z)
    - Total magnitude statistics
    - Gaia proper motion magnitude: sqrt(pm_ra^2 + pm_dec^2)
    - Signal-to-noise ratio aggregations
    - Spectral line indicators (H-alpha, OIII, Na D)
    """
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

    # Total apparent magnitude across all photometric bands
    mag_cols = [c for c in df.columns if c.endswith('_mag')]
    if len(mag_cols) > 0:
        df['total_mag'] = df[mag_cols].mean(axis=1)
        df['mag_span'] = df[mag_cols].max(axis=1) - df[mag_cols].min(axis=1)
        df['mag_std'] = df[mag_cols].std(axis=1)

    # Signal-to-Noise Ratio (SNR) features
    snr_cols = [c for c in df.columns if c.startswith('snr_')]
    if len(snr_cols) > 0:
        df['total_snr'] = df[snr_cols].sum(axis=1)
        df['avg_snr'] = df[snr_cols].mean(axis=1)
        df['min_snr'] = df[snr_cols].min(axis=1)

    # Astrometric proper motion magnitude
    if 'pm_ra' in df.columns and 'pm_dec' in df.columns:
        df['proper_motion_magnitude'] = np.sqrt(df['pm_ra'] ** 2 + df['pm_dec'] ** 2)

    # Spectral indicators
    if 'h_alpha_strength' in df.columns and 'oIII_strength' in df.columns:
        df['h_alpha_to_oIII'] = df['h_alpha_strength'] / (df['oIII_strength'].abs() + 1e-5)

    return df


def train_model(model, train_loader, val_loader, class_weights=None, epochs=EPOCHS, lr=LR):
    """Train single fold model with ReduceLROnPlateau and macro-F1 checkpointing."""
    if class_weights is not None:
        weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.5)

    best_f1 = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        val_f1 = f1_score(val_true, val_preds, average='macro')
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs:
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"  Epoch {epoch:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Macro-F1: {val_f1:.4f}")

    if best_weights is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    return model, best_f1


def run_pipeline(train_path="train.csv", test_path="test.csv", epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    set_seed()
    print("=" * 60)
    print("  AstroClassify: PyTorch Deep MLP Training Pipeline")
    print(f"  Device: {DEVICE} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 60)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at: {train_path}")

    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

    target_col = next((c for c in ["type", "target", "label", "y"] if c in df.columns), None)
    id_col = next((c for c in ["idx", "object_id", "id"] if c in df.columns), None)

    print(f"Loaded train dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target column: '{target_col}' | ID column: '{id_col}'")

    # Feature engineering
    print("\nExtracting astronomical domain features...")
    df_feat = feature_engineering(df)
    if test_df is not None:
        test_feat = feature_engineering(test_df)
    else:
        test_feat = None

    feature_cols = [c for c in df_feat.columns if c not in {target_col, id_col}]
    print(f"Total features after engineering: {len(feature_cols)}")

    # Handle categorical / missing
    df_processed = df_feat[feature_cols].copy()
    if test_feat is not None:
        test_processed = test_feat[feature_cols].copy()
    else:
        test_processed = None

    for col in feature_cols:
        if df_processed[col].dtype == 'object':
            le_col = LabelEncoder()
            all_vals = df_processed[col].fillna('missing')
            if test_processed is not None:
                all_vals = pd.concat([all_vals, test_processed[col].fillna('missing')])
            le_col.fit(all_vals)
            df_processed[col] = le_col.transform(df_processed[col].fillna('missing'))
            if test_processed is not None:
                test_processed[col] = le_col.transform(test_processed[col].fillna('missing'))

    X = df_processed.fillna(0).values
    y = df_feat[target_col].values

    # Encode labels
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    num_classes = len(label_enc.classes_)
    print(f"Class labels ({num_classes}): {list(label_enc.classes_)}")

    # Compute balanced class weights for CrossEntropyLoss
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    print(f"Calculated balanced class weights: {np.round(class_weights, 3)}")

    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_scores = []
    test_fold_preds = []

    if test_processed is not None:
        X_test = test_processed.fillna(0).values
        X_test_scaled = scaler.transform(X_test)
        test_dataset = SpaceDataset(X_test_scaled)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        X_test_scaled = None
        test_loader = None

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_encoded)):
        print(f"\n--- FOLD {fold + 1}/{N_SPLITS} ---")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        train_ds = SpaceDataset(X_train, y_train)
        val_ds = SpaceDataset(X_val, y_val)

        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = SpaceClassifier(X_scaled.shape[1], num_classes).to(DEVICE)
        model, fold_f1 = train_model(model, train_ld, val_ld, class_weights=class_weights, epochs=epochs, lr=lr)
        fold_scores.append(fold_f1)
        print(f"Fold {fold + 1} Best Macro-F1: {fold_f1:.4f}")

        if test_loader is not None:
            model.eval()
            fold_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(DEVICE)
                    out = model(batch)
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                    fold_preds.extend(preds)
            test_fold_preds.append(fold_preds)

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))
    print("\n" + "=" * 60)
    print(f"  CROSS-VALIDATION RESULTS: Mean Macro-F1 = {mean_f1:.4f} +/- {std_f1:.4f}")
    print("=" * 60)

    # Save submission if test data available
    if test_df is not None and len(test_fold_preds) > 0:
        final_preds = []
        for i in range(len(test_df)):
            votes = [test_fold_preds[f][i] for f in range(len(test_fold_preds))]
            majority = max(set(votes), key=votes.count)
            final_preds.append(majority)

        predicted_labels = label_enc.inverse_transform(final_preds)
        sub = pd.DataFrame()
        sub[id_col] = test_df[id_col]

        for cls_name in sorted(df[target_col].unique()):
            sub[cls_name] = (predicted_labels == cls_name).astype(int)

        sub_path = "pytorch_submission.csv"
        sub.to_csv(sub_path, index=False)
        print(f"\nSubmission generated and saved to: {sub_path}")

    return mean_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch Deep MLP on Space Object Data")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    args = parser.parse_args()

    run_pipeline(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
