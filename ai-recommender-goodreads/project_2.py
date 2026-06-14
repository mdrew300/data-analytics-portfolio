# -*- coding: utf-8 -*-
"""

# Project 2

### Key Requirements
1. Explore the dataset and provide any insights you find about users, reviews, and the books. This is an important step as unsupervised learning projects are exploratory by nature. Highlight patterns, anomalies, or questions the data raises that could affect modeling or recommendations.
2. Compare and evaluate a set of recommendation models based on user-based and item-based collaborative filtering (in Python, using surprise). Include a simple popularity/mean baseline as a benchmark. Evaluate performance using RMSE for rating prediction and Precision@K / Recall@K for Top-N recommendations. Which model performs better and why? If your collaborative-filtering models do not outperform the baseline, discuss why (e.g., data sparsity, the baseline being a strong "popularity" recommender) and what that implies for model selection.
3. Add an AI re-ranking layer on top of your recommender (Refer to Week 4 materials). Take the Top-N candidates from your best collaborative-filtering model and use an LLM to re-rank and personalize them to a user’s stated preference (e.g., a mood, favorite genre, or other preferences), returning a short explanation for each pick. The LLM should re-rank the candidate recommendations produced by your collaborative-filtering model rather than generate entirely new recommendations using the book metadata (e.g., author, title, year, average rating, ...) as context. You may experiment with different prompt styles or personalization strategies. You may use any LLM (cloud or local) for this layer - Gemini is recommended because it's free (see the API-key guide) - but cite the model and provider you used, and never include your key in the submission.
4. Build a Streamlit app that wraps your recommender. (Refer to Week 3 and 4 demos and class materials and the Guides section below on how to build and run a Streamlit app). The app should let a user be selected, show the collaborative-filtering Top-N, and offer the LLM re-ranking.

5. Record a short (~2–3 minute) video demo walking through your app and briefly explaining how it works - the CF step, the LLM step, and one design choice you made. Do not include your API key in your code or submission.
Business discussion. What are the business applications of each model (collaborative filtering and the LLM layer) and what challenges might a business face when setting these up? What would be your recommended approach for this dataset?
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from surprise import KNNBasic, BaselineOnly, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

from pydantic import BaseModel, Field

from pathlib import Path
DATA_DIR = Path(__file__).parent


books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")

"""# Data Dictionary
---
---

## Books dataset:
---

book_id: unique book ID

isbn: the International Standard Book Number (10-digit)

authors: book authors, separated by comma

original_publication_year: the year the book was published

title: main title of the book (also indicates whether it is part of a series)

language_code: language of the edition

average_rating: average rating submitted by reviewers

ratings_count: number of submitted ratings

text_reviews_count: number of submitted text reviews

ratings_1 … ratings_5: number of 1- through 5-star reviews

image_url / small_image_url: links to the book image


## Rating dataset:
---

book_id: the book being rated

user_id: the user who submitted the rating

rating: the user's rating, from 1 to 5

# EDA
---
---
"""

books.head()

ratings.head()

print(books.info())

print(ratings.info())

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Create subplots for books numeric columns
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Books Dataset - Numeric Columns Distribution', fontsize=16, fontweight='bold', y=1.00)

# Books columns to visualize
books_cols = ['original_publication_year', 'average_rating', 'ratings_count',
              'text_reviews_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']

# Flatten axes for easier iteration
axes_flat = axes.flatten()

for idx, col in enumerate(books_cols):
    ax = axes_flat[idx]

    # Use log scale for count columns to better see the distribution
    if 'count' in col.lower():
        data = books[col][books[col] > 0]  # Remove zeros for log scale
        ax.hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xscale('log')
        ax.set_title(f'{col} (log scale)', fontweight='bold')
    else:
        ax.hist(books[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{col}', fontweight='bold')

    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics for books numeric columns
print("\n" + "="*80)
print("BOOKS DATASET - SUMMARY STATISTICS")
print("="*80)
print(books[books_cols].describe())

# Visualize ratings distribution from ratings dataset
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Ratings Dataset - Distribution Analysis', fontsize=16, fontweight='bold')

# 1. Distribution of individual ratings
ax1 = axes[0]
rating_counts = ratings['rating'].value_counts().sort_index()
colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
ax1.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black', alpha=0.8, width=0.6)
ax1.set_xlabel('Rating (1-5)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Individual User Ratings', fontsize=13, fontweight='bold')
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for i, v in enumerate(rating_counts.values):
    ax1.text(rating_counts.index[i], v + max(rating_counts.values)*0.01, str(v),
             ha='center', va='bottom', fontweight='bold')

# 2. Distribution of average user ratings (per user)
ax2 = axes[1]
user_avg_ratings = ratings.groupby('user_id')['rating'].mean()
ax2.hist(user_avg_ratings, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Average Rating per User', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
ax2.set_title('Distribution of Average User Ratings', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(user_avg_ratings.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {user_avg_ratings.mean():.2f}')
ax2.axvline(user_avg_ratings.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {user_avg_ratings.median():.2f}')
ax2.legend()

plt.tight_layout()
plt.show()

# Print ratings summary statistics
print("\n" + "="*80)
print("RATINGS DATASET - SUMMARY STATISTICS")
print("="*80)
print(f"Total Ratings: {len(ratings)}")
print(f"Unique Users: {ratings['user_id'].nunique()}")
print(f"Unique Books: {ratings['book_id'].nunique()}")
print(f"\nRating Distribution (counts):")
print(rating_counts)
print(f"\nAverage Rating Statistics (per user):")
print(user_avg_ratings.describe())
print(f"\nSparsity: {1 - (len(ratings) / (ratings['user_id'].nunique() * ratings['book_id'].nunique())): .4%}")

"""## Top 100 Books
---
"""

top100 = books.sort_values("average_rating", ascending=False).head(n=100)

top100

"""NOTES TO BE REMOVED LATER: Added Additional EDA:
- overall rating distribution from the books dataset
- avg rating by publication year
- avg rating by author
- look at optimistic and pessimistic reviewers

"""

# Overall rating distribution from the Books dataset
books["average_rating"].describe()

plt.figure(figsize=(10, 6))
sns.histplot(books["average_rating"], bins=30, kde=True)

plt.title("Distribution of Average Book Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Number of Books")
plt.show()

# Average rating by publication year
rating_by_year = (
    books
    .dropna(subset=["original_publication_year"])
    .groupby("original_publication_year")
    .agg(
        avg_rating=("average_rating", "mean"),
        book_count=("book_id", "count")
    )
    .reset_index()
)

# Filter to years with enough books so the trend is not driven by tiny samples
rating_by_year_filtered = rating_by_year[rating_by_year["book_count"] >= 10]

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=rating_by_year_filtered,
    x="original_publication_year",
    y="avg_rating"
)

plt.title("Average Rating by Publication Year")
plt.xlabel("Original Publication Year")
plt.ylabel("Average Rating")
plt.show()

# Average rating by author
author_summary = (
    books
    .groupby("authors")
    .agg(
        avg_rating=("average_rating", "mean"),
        book_count=("book_id", "count"),
        total_ratings=("ratings_count", "sum")
    )
    .reset_index()
)

# Filter to authors with multiple books and meaningful rating volume
author_summary_filtered = author_summary[
    (author_summary["book_count"] >= 3) &
    (author_summary["total_ratings"] >= 10000)
]

top_authors = author_summary_filtered.sort_values("avg_rating", ascending=False).head(15)

plt.figure(figsize=(12, 7))
sns.barplot(
    data=top_authors,
    x="avg_rating",
    y="authors"
)

plt.title("Highest Rated Authors with Multiple Books")
plt.xlabel("Average Rating")
plt.ylabel("Author")
plt.show()

# Reviewer-level behavior: identify optimistic and pessimistic reviewers
user_rating_behavior = (
    ratings
    .groupby("user_id")
    .agg(
        user_avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )
    .reset_index()
)

# Filter to users with enough ratings to make the pattern meaningful
active_reviewers = user_rating_behavior[user_rating_behavior["rating_count"] >= 20]

active_reviewers["reviewer_type"] = np.where(
    active_reviewers["user_avg_rating"] >= 4.5,
    "Overly optimistic",
    np.where(
        active_reviewers["user_avg_rating"] <= 2.5,
        "Overly pessimistic",
        "Moderate"
    )
)

active_reviewers["reviewer_type"].value_counts()

plt.figure(figsize=(10, 6))
sns.histplot(active_reviewers["user_avg_rating"], bins=30, kde=True)

plt.title("Distribution of Average Ratings by Active Reviewers")
plt.xlabel("User Average Rating")
plt.ylabel("Number of Users")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(
    data=active_reviewers,
    x="reviewer_type",
    order=["Overly pessimistic", "Moderate", "Overly optimistic"]
)

plt.title("Reviewer Rating Tendencies")
plt.xlabel("Reviewer Type")
plt.ylabel("Number of Active Reviewers")
plt.show()

"""The ratings are generally skewed toward higher values, which suggests that users tend to rate books favorably overall. Ratings also appear to vary by publication year and author, although author-level comparisons should be interpreted carefully because some authors have far more books and ratings than others. At the user level, some reviewers consistently rate books very high or very low, which may introduce bias into recommendation models.

---

#Recommendation System Code
"""



# Wrap the Goodreads ratings into a Surprise dataset.
# Goodreads ratings use a 1–5 star scale.
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    ratings[["user_id", "book_id", "rating"]],
    reader
)

# Use all ratings to build the training set.
# This is useful for generating recommendations; later we'll use
# a train/test split to evaluate recommendation performance.
full_trainset = data.build_full_trainset()

print(
    f"Trainset: {full_trainset.n_users:,} users, "
    f"{full_trainset.n_items:,} books, "
    f"{full_trainset.n_ratings:,} ratings"
)

"""##Collaborative Filtering

###User-Based CF
"""

# User-based Collaborative Filtering (UBCF)
# Similar readers tend to enjoy similar books.
# Pearson correlation measures similarity between users based on
# their shared rating behavior.
# The model uses the 10 nearest neighbors to estimate unknown ratings.

ubcf = KNNBasic(
    k=10,
    sim_options={"name": "pearson", "user_based": True},
    verbose=False
)

ubcf.fit(full_trainset)

"""###Top-N Recommendations"""

MIN_RATINGS = 20
counts = ratings["book_id"].value_counts()
popular_books = set(counts[counts >= MIN_RATINGS].index)

book_lookup = books.set_index("book_id")[["title", "authors"]].to_dict("index")

def top_n_for_user(model, user_id, top_n=5):
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])

    scored = [
        (
            book_lookup[b]["title"],
            book_lookup[b]["authors"],
            model.predict(user_id, b).est
        )
        for b in books["book_id"]
        if b not in seen and b in popular_books
    ]

    return sorted(scored, key=lambda x: -x[2])[:top_n]

top_n_for_user(ubcf, user_id=1, top_n=5)

"""###Item-Based CF"""

# Item-based collaborative filtering:
# Similarity is calculated between books using cosine similarity.
# Books that are rated similarly by readers are treated as similar items.

ibcf = KNNBasic(
    k=10,
    sim_options={
        "name": "cosine",
        "user_based": False
    },
    verbose=False
)

ibcf.fit(full_trainset)

print("Top 5 book recommendations from Item-Based Collaborative Filtering:")
top_n_for_user(ibcf, user_id=1, top_n=5)

"""##Evaluation

###Precision and Recall @ K
"""

# Top-N precision and recall for recommendation quality:
#   - a book is considered "relevant" if its true rating is >= threshold (4.0 here),
#   - for each user we rank predicted book ratings from highest to lowest,
#   - precision = fraction of recommended books in the top-N that were actually relevant,
#   - recall    = fraction of all relevant books that were captured in the top-N recommendations.
#
# top_n = number of recommendations evaluated
# k     = neighborhood size used by the collaborative filtering model
#
# Returns average precision and recall across all users.

def precision_recall_at_k(predictions, top_n=10, threshold=4.0):
    user_data = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:
        user_data[uid].append((est, true_r))

    precisions, recalls = [], []

    for items in user_data.values():
        items.sort(key=lambda x: x[0], reverse=True)

        n_relevant = sum(1 for _, t in items if t >= threshold)
        n_hits = sum(1 for _, t in items[:top_n] if t >= threshold)

        precisions.append(n_hits / top_n)

        if n_relevant > 0:
            recalls.append(n_hits / n_relevant)

    return np.mean(precisions), np.mean(recalls)

"""###Train/Test Split"""

# Build a 90/10 train/test split so we can evaluate recommendations
# against held-out Goodreads ratings.
# random_state ensures the split is reproducible across runs.

trainset, testset = train_test_split(
    data,
    test_size=0.10,
    random_state=6604
)

print(
    f"Training set: {trainset.n_ratings:,} book ratings | "
    f"Test set: {len(testset):,} book ratings"
)

"""###

###Model Comparison
"""

# In practice, you would not hand-pick k or the similarity metric.
# Hyperparameters can be tuned using cross-validation
# (GridSearchCV or RandomizedSearchCV in Surprise).
#
# For a book recommendation system, evaluation should focus on ranking quality
# metrics such as Precision@K, Recall@K, or NDCG@K rather than RMSE alone.
#
# Production recommendation systems often use time-based train/test splits
# and validate improvements through online A/B testing.

models = {
    "Baseline": BaselineOnly(verbose=False),

    "User-Based CF (Pearson)": KNNBasic(
        k=10,
        sim_options={"name": "pearson", "user_based": True},
        verbose=False
    ),

    "Item-Based CF (Cosine)": KNNBasic(
        k=10,
        sim_options={"name": "cosine", "user_based": False},
        verbose=False
    ),
}

results = []

for name, model in models.items():

    model.fit(trainset)

    preds = model.test(testset)

    precision, recall = precision_recall_at_k(
        preds,
        top_n=10,
        threshold=4.0
    )

    results.append({
        "Model": name,
        "RMSE": accuracy.rmse(preds, verbose=False),
        "Precision@10": precision,
        "Recall@10": recall
    })

results_df = pd.DataFrame(results).round(4)

results_df

# Identify best models by each metric
best_rmse = results_df.sort_values("RMSE", ascending=True).iloc[0]
best_precision = results_df.sort_values("Precision@10", ascending=False).iloc[0]
best_recall = results_df.sort_values("Recall@10", ascending=False).iloc[0]

print(f"Best RMSE model: {best_rmse['Model']} with RMSE = {best_rmse['RMSE']}")
print(f"Best Precision@10 model: {best_precision['Model']} with Precision@10 = {best_precision['Precision@10']}")
print(f"Best Recall@10 model: {best_recall['Model']} with Recall@10 = {best_recall['Recall@10']}")

"""The best model depends on the evaluation goal. RMSE measures how close predicted ratings are to actual ratings, while Precision@10 and Recall@10 evaluate the quality of the top-10 recommendation list. For a Goodreads recommendation system, ranking metrics are especially important because the goal is to recommend books a user is likely to enjoy, not just predict exact ratings.

If the collaborative filtering models outperform the baseline, this suggests that user-book interaction patterns provide useful personalized information beyond average rating behavior. If the baseline performs as well as or better than the collaborative filtering models, this may indicate that the Goodreads rating matrix is sparse, that many users/books have limited rating history, or that the baseline is strong because popular/highly rated books are broadly appealing. In that case, the simpler baseline may be preferred unless the collaborative filtering model provides better personalization or better top-N ranking performance.

# AI Re-Ranking Layer
---
---
"""

client = genai.Client(api_key=getpass.getpass("Enter your Gemini API key: "))
MODEL = "gemini-2.5-flash-lite"

class BookPick(BaseModel):
    title: str = Field(description="Exact book title from the candidate list.")
    authors: str = Field(description="Book author(s) as listed in the candidate list.")
    reason: str = Field(description="One sentence on why this book fits the preference.")


def rerank_with_gemini(user_id=1, preference="I want a warm, hopeful read with a strong emotional payoff.", top_n=5, model_name=ibcf):
    """Re-rank the collaborative-filtering top-N results with Gemini."""
    candidates = top_n_for_user(model_name, user_id=user_id, top_n=top_n)
    catalog = "\n".join(
        f"{i + 1}. {title} by {authors} (CF score: {score:.2f})"
        for i, (title, authors, score) in enumerate(candidates)
    )

    system_instruction = (
        "You are a book concierge. Re-rank the candidate books for the viewer’s preference, "
        "returning the best matches first and a short reason for each pick. "
        "Use the exact titles and author names from the candidate list."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=(
            f"Viewer preference: {preference}\n\n"
            f"Candidate books (already ranked by collaborative filtering):\n{catalog}"
        ),
        config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": list[BookPick],
        },
    )

    return response.parsed


user_preference = "I want a warm, hopeful read with a strong emotional payoff."
llm_ranked = rerank_with_gemini(user_id=1, preference=user_preference, top_n=5, model_name=ibcf)

for pick in llm_ranked:
    print(f"{pick.title} by {pick.authors} -> {pick.reason}")

# ============================================================
# STEP 4 – STREAMLIT APP
# Run from your terminal with:  streamlit run project_2.py
# Before running, comment out the `!pip install surprise` line
# at the top (Jupyter magic commands don't work in Streamlit).
# ============================================================
import streamlit as st
import google.genai as genai  # pip install google-genai

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
st.title("📚 Book Recommender")
st.caption("Collaborative Filtering + Gemini AI Re-Ranking")

# ---- Sidebar: configuration ----
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your key here — never stored",
    )
    model_choice = st.radio(
        "CF Model",
        ["Item-Based CF (Cosine)", "User-Based CF (Pearson)"],
    )
    top_n_slider = st.slider("Top-N Recommendations", min_value=3, max_value=10, value=5)

# ---- Data & model loading (cached so Streamlit doesn't retrain on every widget change) ----
@st.cache_data
def load_datasets():
    b = pd.read_csv(DATA_DIR / "Books.csv")
    r = pd.read_csv(DATA_DIR / "Ratings.csv")
    return b, r

@st.cache_resource
def train_cf_models(_ratings):
    rdr = Reader(rating_scale=(1, 5))
    ds = Dataset.load_from_df(_ratings[["user_id", "book_id", "rating"]], rdr)
    full_ts = ds.build_full_trainset()
    _ubcf = KNNBasic(k=10, sim_options={"name": "pearson", "user_based": True}, verbose=False)
    _ibcf = KNNBasic(k=10, sim_options={"name": "cosine", "user_based": False}, verbose=False)
    _ubcf.fit(full_ts)
    _ibcf.fit(full_ts)
    return _ubcf, _ibcf

with st.spinner("Loading data and training models (first run only)…"):
    books_st, ratings_st = load_datasets()
    ubcf_st, ibcf_st = train_cf_models(ratings_st)

_min_ratings_st = 20
_counts_st = ratings_st["book_id"].value_counts()
_popular_st = set(_counts_st[_counts_st >= _min_ratings_st].index)
_lookup_st = books_st.set_index("book_id")[["title", "authors"]].to_dict("index")


def get_cf_recs(model, uid, n=5):
    seen = set(ratings_st.loc[ratings_st["user_id"] == uid, "book_id"])
    scored = [
        (_lookup_st[b]["title"], _lookup_st[b]["authors"], model.predict(uid, b).est)
        for b in books_st["book_id"]
        if b not in seen and b in _popular_st and b in _lookup_st
    ]
    return sorted(scored, key=lambda x: -x[2])[:n]


# ---- User selector ----
user_ids_st = sorted(ratings_st["user_id"].unique())
selected_user = st.selectbox("Select a User ID", user_ids_st)

# ---- CF Recommendations ----
st.subheader(f"Top-{top_n_slider} {model_choice} Recommendations — User {selected_user}")

active_model = ibcf_st if model_choice.startswith("Item") else ubcf_st

with st.spinner("Generating collaborative-filtering recommendations…"):
    cf_results = get_cf_recs(active_model, selected_user, top_n_slider)

if cf_results:
    rec_df = pd.DataFrame(cf_results, columns=["Title", "Authors", "Predicted Rating"])
    rec_df.index = rec_df.index + 1
    rec_df["Predicted Rating"] = rec_df["Predicted Rating"].round(2)
    st.dataframe(rec_df, use_container_width=True)
else:
    st.warning("No recommendations found for this user (too few ratings to generate candidates).")
    st.stop()

# ---- LLM Re-Ranking ----
st.divider()
st.subheader("AI Re-Ranking with Gemini")
st.caption("Enter a reading preference and Gemini will re-order the CF candidates to best match it.")

preference_input = st.text_area(
    "Your reading mood or preference",
    placeholder="e.g., I want a warm, hopeful read with a strong emotional payoff.",
    height=80,
)

run_llm = st.button(
    "Re-rank with Gemini",
    disabled=not (api_key and preference_input),
    help="Requires a Gemini API key and a preference description.",
)

if run_llm:
    with st.spinner("Asking Gemini to re-rank…"):
        try:
            _gemini_client = genai.Client(api_key=api_key)

            catalog = "\n".join(
                f"{i+1}. {title} by {authors} (CF score: {score:.2f})"
                for i, (title, authors, score) in enumerate(cf_results)
            )
            system_instruction = (
                "You are a book concierge. Re-rank the candidate books to best match "
                "the viewer's stated preference, returning the best picks first. "
                "Provide a one-sentence reason for each pick. "
                "Use the exact titles and author names from the candidate list."
            )
            response = _gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=(
                    f"Viewer preference: {preference_input}\n\n"
                    f"Candidate books (pre-ranked by collaborative filtering):\n{catalog}"
                ),
                config={
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json",
                    "response_schema": list[BookPick],
                },
            )
            picks = response.parsed

            st.markdown("### Gemini's Re-Ranked Picks")
            for i, pick in enumerate(picks, 1):
                with st.expander(f"{i}. **{pick.title}** — {pick.authors}", expanded=True):
                    st.write(pick.reason)

        except Exception as exc:
            st.error(f"Gemini error: {exc}")