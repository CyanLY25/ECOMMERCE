#!/usr/bin/env python3
"""
Script para verificar las columnas de X_train.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.training.common import load_datasets, separate_features_target

config = AIConfig()
train_df, val_df, test_df = load_datasets(config)

X_train, y_train = separate_features_target(train_df, config.TARGET_VARIABLE)
print("=" * 60)
print("Columnas de X_train (características):")
print("=" * 60)
for i, col in enumerate(train_df.drop(columns=[config.TARGET_VARIABLE]).select_dtypes(exclude=['object']).columns):
    print(f"{i+1}. {col}")
print(f"\nTotal de columnas: {X_train.shape[1]}")
print(f"Forma de X_train: {X_train.shape}")
print("=" * 60)
