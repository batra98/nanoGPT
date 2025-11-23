"""
Verifier function for RLVR (RL from Verifier Rewards).

Simple deterministic scoring function: count of 's' characters (case-insensitive),
capped at Rmax.
"""


def compute_verifier_score(text: str, rmax: int = 50) -> float:
    """
    Compute verifier score for a text completion.
    
    Verifier: v(y) = min(count('s' in y), Rmax)
    
    Args:
        text: Generated text completion
        rmax: Maximum reward cap (default: 50)
    
    Returns:
        Scalar reward score (0 to rmax)
    
    Constraints:
        - Case-insensitive counting
        - Capped at Rmax to prevent extreme values
        - Single EOS token (handled by generation, stop at <|endoftext|>)
        - Max tokens: 200 (handled by generation parameters)
    """
    # Count 's' characters (case-insensitive)
    count_s = text.lower().count('s')
    
    # Cap at Rmax
    score = min(count_s, rmax)
    
    return float(score)


def compute_verifier_scores(texts: list, rmax: int = 50) -> list:
    """
    Compute verifier scores for multiple texts.
    
    Args:
        texts: List of text completions
        rmax: Maximum reward cap
    
    Returns:
        List of scores
    """
    return [compute_verifier_score(text, rmax) for text in texts]


def report_verifier_statistics(scores: list) -> dict:
    """
    Compute statistics for verifier scores.
    
    Args:
        scores: List of verifier scores
    
    Returns:
        Dictionary with statistics
    """
    import numpy as np
    
    scores_array = np.array(scores)
    
    stats = {
        'mean': float(np.mean(scores_array)),
        'std': float(np.std(scores_array)),
        'min': float(np.min(scores_array)),
        'max': float(np.max(scores_array)),
        'median': float(np.median(scores_array)),
        'count': len(scores)
    }
    
    return stats

