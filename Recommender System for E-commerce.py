import os
import pickle
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error


# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = "./data/ecommerce_data.csv"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
SVD_K = 50  # latent factors for SVD (reduce for small datasets)
TFIDF_MAX_FEATURES = 5000
HYBRID_ALPHA = 0.7  # CF weight
HYBRID_BETA = 0.3   # Content weight


# ---------------------------
# UTILITIES & DATA
# ---------------------------

def generate_synthetic_dataset(path: str, n_users=50, n_items=100, sparsity=0.05):
    """Generates a small synthetic dataset and writes to path."""
    np.random.seed(42)
    users = [f"u{u}" for u in range(1, n_users + 1)]
    items = [f"i{i}" for i in range(1, n_items + 1)]

    rows = []
    for u in users:
        for i in items:
            if np.random.rand() < sparsity:
                rating = np.random.randint(1, 6)  # 1-5 ratings
                title = f"Product {i}"
                description = f"This is a description for product {i}. It's great for use case {np.random.randint(1,10)}."
                rows.append((u, i, rating, title, description))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "product_title", "description"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Synthetic dataset generated at {path} with {len(df)} interactions")
    return df


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Data not found at {path}. Generating synthetic dataset...")
        return generate_synthetic_dataset(path)

    df = pd.read_csv(path)
    # Basic expected columns: user_id, item_id, product_title, description, rating (optional)
    expected = ["user_id", "item_id"]
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"Input CSV must contain column: {c}")

    # Fill missing textual columns
    if "product_title" not in df.columns:
        df["product_title"] = df["item_id"].astype(str)
    if "description" not in df.columns:
        df["description"] = ""

    # If rating not present, create implicit feedback (1 for interaction)
    if "rating" not in df.columns:
        df["rating"] = 1

    df = df.dropna(subset=["user_id", "item_id"]).reset_index(drop=True)
    return df


# ---------------------------
# PREPROCESS & MATRICES
# ---------------------------

def build_user_item_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return user-item matrix (users x items) filled with ratings (0 for missing)."""
    users = df["user_id"].unique().tolist()
    items = df["item_id"].unique().tolist()
    pivot = df.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean", fill_value=0)
    # Ensure consistent ordering
    pivot = pivot.reindex(index=users, columns=items, fill_value=0)
    return pivot, users, items


# ---------------------------
# COLLABORATIVE FILTERING
# ---------------------------

def compute_item_similarity(user_item_matrix: pd.DataFrame) -> np.ndarray:
    """Compute cosine similarity between items."""
    print("Computing item-item cosine similarity...")
    item_matrix = user_item_matrix.T.values  # items x users
    sim = cosine_similarity(item_matrix)
    return sim


def item_item_recommend(user_id: str, user_item_matrix: pd.DataFrame, item_similarity: np.ndarray, items: List[str], top_k=10) -> List[Tuple[str, float]]:
    """Recommend items for a user based on item-item similarity and user's existing ratings."""
    if user_id not in user_item_matrix.index:
        raise ValueError(f"Unknown user_id: {user_id}")

    user_ratings = user_item_matrix.loc[user_id].values  # vector of length num_items
    scores = item_similarity.dot(user_ratings)  # score for each item

    # zero-out items already interacted with (optional)
    interacted = user_ratings > 0
    scores[interacted] = -np.inf

    top_idx = np.argpartition(-scores, range(top_k))[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(items[i], float(scores[i])) for i in top_idx]


# ---------------------------
# MATRIX FACTORIZATION (SVD)
# ---------------------------

def train_svd(user_item_matrix: pd.DataFrame, k: int = SVD_K):
    """Train SVD on user-item matrix and return predicted rating dataframe."""
    print("Training SVD... (this can be slow for very large matrices)")
    R = user_item_matrix.values
    user_means = np.true_divide(R.sum(1), (R != 0).sum(1)).reshape(-1, 1)
    # replace nan means with global mean
    global_mean = np.nanmean(R[R != 0]) if np.any(R != 0) else 0.0
    user_means = np.where(np.isnan(user_means), global_mean, user_means)

    R_demeaned = np.where(R != 0, R - user_means, 0)

    # choose k <= min(R.shape)-1
    k = min(k, min(R.shape) - 1)
    if k <= 0:
        print("Dataset too small for SVD latent factors; returning baseline predictions")
        return None, None, None

    U, sigma, Vt = svds(R_demeaned, k=k)
    sigma_diag = np.diag(sigma)
    all_user_predicted = np.dot(np.dot(U, sigma_diag), Vt) + user_means
    preds_df = pd.DataFrame(all_user_predicted, index=user_item_matrix.index, columns=user_item_matrix.columns)
    return preds_df, U, Vt


def svd_recommend(user_id: str, preds_df: pd.DataFrame, user_item_matrix: pd.DataFrame, top_k=10) -> List[Tuple[str, float]]:
    if preds_df is None:
        return []
    if user_id not in preds_df.index:
        raise ValueError(f"Unknown user_id: {user_id}")
    user_pred = preds_df.loc[user_id].values
    interacted = user_item_matrix.loc[user_id].values > 0
    user_pred[interacted] = -np.inf
    top_idx = np.argpartition(-user_pred, range(top_k))[:top_k]
    top_idx = top_idx[np.argsort(-user_pred[top_idx])]
    items = preds_df.columns.tolist()
    return [(items[i], float(user_pred[i])) for i in top_idx]


# ---------------------------
# CONTENT-BASED (TF-IDF)
# ---------------------------

def build_tfidf_embeddings(df_items: pd.DataFrame, max_features=TFIDF_MAX_FEATURES):
    """Build TF-IDF matrix for items (title + description) and return vectorizer + matrix."""
    print("Building TF-IDF embeddings for product text...")
    texts = (df_items["product_title"].fillna("") + " \n " + df_items["description"].fillna("")).values
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf = vectorizer.fit_transform(texts)
    return vectorizer, tfidf


def compute_content_similarity(tfidf_matrix) -> np.ndarray:
    print("Computing content-content similarity (linear kernel)...")
    return linear_kernel(tfidf_matrix, tfidf_matrix)


# ---------------------------
# HYBRID RECOMMENDER
# ---------------------------

def hybrid_recommend_by_item(item_id: str, items: List[str], item_similarity: np.ndarray, content_similarity: np.ndarray, top_k=10, alpha=HYBRID_ALPHA, beta=HYBRID_BETA) -> List[Tuple[str, float]]:
    """Given an item_id, find similar items using weighted hybrid between CF similarity and content similarity."""
    if item_id not in items:
        raise ValueError(f"Unknown item_id: {item_id}")
    idx = items.index(item_id)
    cf_scores = item_similarity[idx]
    content_scores = content_similarity[idx]
    hybrid_scores = alpha * cf_scores + beta * content_scores

    hybrid_scores[idx] = -np.inf  # exclude self
    top_idx = np.argpartition(-hybrid_scores, range(top_k))[:top_k]
    top_idx = top_idx[np.argsort(-hybrid_scores[top_idx])]
    return [(items[i], float(hybrid_scores[i])) for i in top_idx]


def hybrid_recommend_for_user(user_id: str, user_item_matrix: pd.DataFrame, items: List[str], item_similarity: np.ndarray, content_similarity: np.ndarray, top_k=10, alpha=HYBRID_ALPHA, beta=HYBRID_BETA) -> List[Tuple[str, float]]:
    """Recommend for a user by combining item-item CF scores (weighted by user's ratings) and content similarity.

    Score per item = alpha * sum_j (item_sim[i,j] * rating_j) + beta * sum_j (content_sim[i,j] * rating_j)
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"Unknown user_id: {user_id}")

    user_ratings = user_item_matrix.loc[user_id].values
    interacted = user_ratings > 0

    cf_scores = item_similarity.dot(user_ratings)
    content_scores = content_similarity.dot(user_ratings)

    hybrid_scores = alpha * cf_scores + beta * content_scores
    hybrid_scores[interacted] = -np.inf

    top_idx = np.argpartition(-hybrid_scores, range(top_k))[:top_k]
    top_idx = top_idx[np.argsort(-hybrid_scores[top_idx])]
    return [(items[i], float(hybrid_scores[i])) for i in top_idx]


# ---------------------------
# EVALUATION
# ---------------------------

def rmse_between(preds_df: pd.DataFrame, df: pd.DataFrame) -> float:
    # only compute RMSE for observed ratings
    merged = df.merge(preds_df.reset_index().melt(id_vars="user_id", var_name="item_id", value_name="pred_rating"), on=["user_id", "item_id"], how="inner")
    if merged.empty:
        return float('nan')
    rmse = np.sqrt(mean_squared_error(merged["rating"], merged["pred_rating"]))
    return rmse


def precision_recall_at_k(recommendations: dict, ground_truth: dict, k=10) -> Tuple[float, float]:
    """recommendations: {user: [item1, item2,...]}
       ground_truth: {user: set(items)}
    """
    precisions = []
    recalls = []
    for user, recs in recommendations.items():
        topk = recs[:k]
        gt = ground_truth.get(user, set())
        if len(gt) == 0:
            continue
        hit = len([r for r in topk if r in gt])
        precisions.append(hit / len(topk))
        recalls.append(hit / len(gt))
    if not precisions:
        return float('nan'), float('nan')
    return float(np.mean(precisions)), float(np.mean(recalls))


# ---------------------------
# MAIN / DEMO
# ---------------------------

def main(data_path: str = DATA_PATH):
    df = load_data(data_path)
    print(f"Loaded {len(df)} interactions")

    user_item_matrix, users, items = build_user_item_matrix(df)
    print(f"Matrix size: {user_item_matrix.shape} (users x items)")

    # Compute item-item CF similarity
    item_sim = compute_item_similarity(user_item_matrix)

    # Train SVD and produce predictions (if feasible)
    preds_df, U, Vt = train_svd(user_item_matrix, k=SVD_K)
    if preds_df is not None:
        print("SVD trained. Example predicted ratings head:")
        print(preds_df.iloc[:3, :3])
        # Save predictions for later use
        preds_df.to_pickle(os.path.join(MODEL_DIR, "svd_preds.pkl"))

    # Prepare item-level dataframe for text
    # we need one row per item for TF-IDF; aggregate product_title/description by item_id
    item_text = df.groupby("item_id").agg({
        "product_title": lambda x: x.dropna().unique()[0] if len(x.dropna().unique()) > 0 else "",
        "description": lambda x: " ".join(x.dropna().unique())
    }).reset_index()

    vectorizer, tfidf_matrix = build_tfidf_embeddings(item_text)
    content_sim = compute_content_similarity(tfidf_matrix)

    # Save vectorizer and similarities
    with open(os.path.join(MODEL_DIR, "item_similarity.pkl"), "wb") as f:
        pickle.dump(item_sim, f)
    with open(os.path.join(MODEL_DIR, "content_similarity.pkl"), "wb") as f:
        pickle.dump(content_sim, f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print("Models and similarity matrices saved to models/ folder")

    # Demo: pick a random user and show recommendations
    demo_user = user_item_matrix.index[0]
    print(f"Demo user: {demo_user}")

    cf_recs = item_item_recommend(demo_user, user_item_matrix, item_sim, items, top_k=10)
    print("Top CF recommendations (item_id, score):")
    print(cf_recs[:5])

    if preds_df is not None:
        svd_recs = svd_recommend(demo_user, preds_df, user_item_matrix, top_k=10)
        print("Top SVD recommendations (item_id, predicted_rating):")
        print(svd_recs[:5])

    # For item-based hybrid example, pick an item from item_text
    demo_item = item_text["item_id"].iloc[0]
    hybrid_item_recs = hybrid_recommend_by_item(demo_item, item_text["item_id"].tolist(), item_sim[[items.index(demo_item)] if demo_item in items else 0], content_sim, top_k=5) if demo_item in items else []

    # More robust hybrid for user:
    hybrid_user_recs = hybrid_recommend_for_user(demo_user, user_item_matrix, items, item_sim, content_sim, top_k=10)
    print("Top Hybrid recommendations for user (item_id, hybrid_score):")
    print(hybrid_user_recs[:5])

    # Example evaluation: Precision@K using holdout (simple temporal or random split not implemented here)
    # Quick demo with a naive holdout: for users with >=2 interactions, hide 1 and see if recommender recovers it
    ground_truth = {}
    for u in user_item_matrix.index:
        interacted_items = user_item_matrix.loc[u][user_item_matrix.loc[u] > 0].index.tolist()
        if len(interacted_items) >= 2:
            # hide last one as ground truth
            ground_truth[u] = set([interacted_items[-1]])

    # Generate recommendations for these users using hybrid
    recommendations = {}
    for u in ground_truth.keys():
        try:
            recs = [r[0] for r in hybrid_recommend_for_user(u, user_item_matrix, items, item_sim, content_sim, top_k=10)]
            recommendations[u] = recs
        except Exception:
            continue

    prec, rec = precision_recall_at_k(recommendations, ground_truth, k=10)
    print(f"Precision@10: {prec:.4f}, Recall@10: {rec:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-commerce Recommender Demo")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to CSV with user-item interactions")
    args = parser.parse_args()
    main(args.data)
