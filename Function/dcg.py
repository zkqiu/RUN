import numpy as np

def calculate_dcg(scores, k=5):
    """
    Calculate the Discounted Cumulative Gain (DCG) for the top-k results.

    Parameters:
        scores (list or array): Relevance scores of the results, ordered by their ranking.
        k (int): Number of top results to consider for DCG.

    Returns:
        float: The DCG value for the top-k results.
    """
    scores = np.array(scores[:k])  # Consider only the top-k results
    discounts = np.log2(np.arange(2, k + 2))  # Discount factors: log2(i+1)
    dcg = np.sum(scores / discounts)
    return dcg

def calculate_ndcg(scores, k=5):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) for the top-k results.

    Parameters:
        scores (list or array): Relevance scores of the results, ordered by their ranking.
        k (int): Number of top results to consider for NDCG.

    Returns:
        float: The NDCG value for the top-k results.
    """
    ideal_scores = sorted(scores, reverse=True)  # Ideal ranking of scores
    dcg = calculate_dcg(scores, k)
    idcg = calculate_dcg(ideal_scores, k)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

# Example usage
if __name__ == "__main__":
    # Example relevance scores for a ranked list of results
    relevance_scores = [3, 2, 3, 0, 1, 2]

    # Calculate the top-5 DCG
    top5_dcg = calculate_dcg(relevance_scores, k=5)
    print(f"Top-5 DCG: {top5_dcg:.4f}")

    # Calculate the top-5 NDCG
    top5_ndcg = calculate_ndcg(relevance_scores, k=5)
    print(f"Top-5 NDCG: {top5_ndcg:.4f}")
