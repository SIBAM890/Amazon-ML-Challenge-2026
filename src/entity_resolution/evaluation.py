def calculate_f0_5(true_matches: set, pred_matches: set) -> float:
    """
    Calculate F0.5 score for a single Source-1 entity.
    
    Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    - If there are no true matches (singleton), predicting no matches scores 1.0.
    - If it's a singleton and we predict matches, score is 0.0.
    - If precision and recall are both 0, F0.5 is 0.0.
    """
    if not true_matches:
        return 1.0 if not pred_matches else 0.0
        
    if not pred_matches:
        return 0.0
        
    true_positives = len(true_matches.intersection(pred_matches))
    if true_positives == 0:
        return 0.0
        
    precision = true_positives / len(pred_matches)
    recall = true_positives / len(true_matches)
    
    f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f0_5

def macro_f0_5(y_true: dict, y_pred: dict) -> float:
    """
    Calculate macro-averaged F0.5 across all Source-1 entities.
    y_true: dict mapping source1_entity_id -> set of true matched S2/S3 IDs
    y_pred: dict mapping source1_entity_id -> set of predicted S2/S3 IDs
    """
    scores = []
    for s1_id, true_matches in y_true.items():
        pred_matches = y_pred.get(s1_id, set())
        scores.append(calculate_f0_5(true_matches, pred_matches))
        
    return sum(scores) / len(scores) if scores else 0.0

if __name__ == "__main__":
    # Unit test against the worked example in the problem statement
    # S1-00001 -> predicted: [S2-00047, S2-00193, S3-00812], true: [S2-00047, S3-00812]
    # Expected F0.5 ≈ 0.714
    
    true_matches = {"S2-00047", "S3-00812"}
    pred_matches = {"S2-00047", "S2-00193", "S3-00812"}
    
    score = calculate_f0_5(true_matches, pred_matches)
    print(f"Worked Example F0.5: {score:.3f}")
    assert abs(score - 0.714) < 0.001, f"Expected ~0.714, got {score}"
    
    # Test singleton correct
    assert calculate_f0_5(set(), set()) == 1.0
    
    # Test singleton incorrect
    assert calculate_f0_5(set(), {"S2-0001"}) == 0.0
    
    # Test missed match
    assert calculate_f0_5({"S2-0001"}, set()) == 0.0
    
    print("All evaluation unit tests passed!")
