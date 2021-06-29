def get_precision_recall_f1(num_correct: int, num_predicted: int, num_gt: int):
    """
    For t2g evaluation
    """
    assert 0 <= num_correct <= num_predicted
    assert 0 <= num_correct <= num_gt

    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    recall = num_correct / num_gt if num_gt > 0 else 0.0
    f1 = 2.0 / (1.0 / precision + 1.0 / recall) if num_correct > 0 else 0.0

    return precision, recall, f1
