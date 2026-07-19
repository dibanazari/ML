# Hybrid Movie Recommendation System

A movie recommendation engine built on the MovieLens dataset, combining two complementary approaches into a single hybrid model:

- **Collaborative Filtering** — Matrix Factorization (Funk-SVD) trained with Stochastic Gradient Descent, learning latent user/movie factors from rating behavior.
- **Content-Based Filtering** — TF-IDF vectorization of movie genres combined with Cosine Similarity against a per-user taste profile.

The two models are combined with a tunable weighted average. The blend weight (alpha) is selected on a held-out **validation** set, and final performance is reported on an untouched **test** set — keeping model selection and evaluation strictly separate.

## Results

Final performance on the held-out test set:

| Model | RMSE | Precision@10 |
|---|---|---|
| Content-Based | 1.0145 | 0.0184 |
| Collaborative Filtering | 0.8896 | 0.0265 |
| **Hybrid (α = 0.8)** | **0.8741** | **0.0327** |

The hybrid model outperforms both individual approaches on rating-prediction accuracy (RMSE) *and* recommendation ranking quality (Precision@10) — showing that behavioral signals (collaborative) and content signals (genre similarity) capture complementary information that neither method captures alone.

## Dataset

[MovieLens](https://grouplens.org/datasets/movielens/) — `movies.csv` (9,742 movies) and `ratings.csv` (100,836 ratings from 610 users).

## Approach

### 1. Data splitting
Ratings are split **per user** into train / validation / test sets (80% / rest split twice), ensuring every user has ratings in all three sets and avoiding cold-start bias during evaluation.

### 2. Collaborative Filtering (Matrix Factorization)
A Funk-SVD model learns a latent factor vector per user and per movie, plus per-user and per-movie bias terms, trained via SGD on observed ratings only:

```
prediction = global_mean + user_bias + item_bias + user_factors · item_factors
```

- Latent factors: 30
- Learning rate: 0.01
- Regularization: 0.05
- Epochs: 15

Training converges from an initial RMSE of ~0.95 to ~0.75 over 15 epochs.

### 3. Content-Based Filtering (TF-IDF + Cosine Similarity)
Each movie is represented as a TF-IDF vector over its genre tags. A per-user taste profile is built as the rating-weighted average of the genre vectors of movies the user has rated, then new movies are scored by cosine similarity to that profile.

### 4. Hybrid Model
Combines both models with a weighted average:

```
prediction = alpha * CF_prediction + (1 - alpha) * CB_prediction
```

`alpha` is swept from 0.0 to 1.0 on the validation set; the best value found is **α = 0.8**.

### 5. Evaluation
- **RMSE** — rating-prediction accuracy on the test set.
- **Precision@10** — of the top-10 recommended (unseen) movies, the fraction the user actually rated ≥ 4.0 in the test set. Measures ranking quality, not just prediction error.

## Tech stack

- Python
- NumPy
- pandas
- scikit-learn (`TfidfVectorizer`, `cosine_similarity`)

## Project structure

```
.
├── Movie_rec_system.ipynb   # full pipeline: data loading, models, tuning, evaluation
├── movies.csv                # MovieLens movie metadata
├── ratings.csv                # MovieLens ratings
└── README.md
```

## Getting started

```bash
pip install numpy pandas scikit-learn jupyter
jupyter notebook Movie_rec_system.ipynb
```

Run the notebook top to bottom — it loads the data, trains both models, tunes the hybrid weight on the validation set, and reports final metrics on the test set.

### Generating recommendations

```python
recommend_for_user(hybrid, user_id=1, movies=movies, train_ratings=train_full, n=10)
```

Returns the top-N unseen movies for a given user, ranked by predicted rating.

## Key takeaways

- The hybrid model beats both individual methods on RMSE and Precision@10, confirming that behavioral and content signals are complementary rather than redundant.
- The best blend weights collaborative filtering more heavily than content-based (α = 0.8), consistent with collaborative filtering's stronger standalone RMSE (0.8896 vs. 1.0145).
- Per-user train/validation/test splitting and validation-only alpha tuning avoid leakage and give a reliable estimate of real-world ranking quality.

## License

This project is for educational/portfolio purposes. MovieLens data is provided by [GroupLens](https://grouplens.org/) under their own [usage license](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).
