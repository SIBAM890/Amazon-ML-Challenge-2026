import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.config import TRAIN_GROUND_TRUTH, TRAIN_SOURCE2, TRAIN_SOURCE3
from src.entity_resolution.evaluation import macro_f0_5

def run_baselines():
    print("Loading Ground Truth...")
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH, sep='\t', dtype=str)
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna("")
    
    # Validation split (last 20%)
    val_size = int(len(gt_df) * 0.2)
    val_df = gt_df.iloc[-val_size:].copy()
    print(f"Validation set size: {len(val_df)} Source-1 entities")
    
    y_true = {}
    for _, row in val_df.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        y_true[s1] = set(matches.split(',')) if matches.strip() else set()
        
    print("\n--- Baseline 1: All Singletons ---")
    y_pred_singletons = {s1: set() for s1 in y_true.keys()}
    score_singletons = macro_f0_5(y_true, y_pred_singletons)
    print(f"All-Singletons F0.5 Score: {score_singletons:.4f}")
    
    print("\n--- Baseline 2: Naive Random Predictor ---")
    # A naive predictor could randomly pick 1 match for non-singletons, but we don't know the exact candidate pool.
    # We can randomly assign a single random fake match to some portion of entities.
    # To keep it simple, let's predict 1 random match for 10% of entities, and singletons for the rest.
    np.random.seed(42)
    y_pred_random = {}
    for s1 in y_true.keys():
        if np.random.rand() < 0.1:
            y_pred_random[s1] = {"S2-RANDOM"} 
        else:
            y_pred_random[s1] = set()
            
    score_random = macro_f0_5(y_true, y_pred_random)
    print(f"Naive Random F0.5 Score: {score_random:.4f}")

if __name__ == "__main__":
    run_baselines()
