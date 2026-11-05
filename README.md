# Movie Recommendation System

Recommendation systems are among the most impactful applications of machine learning, powering content discovery across streaming platforms, e-commerce, and social media. At their core they solve a single problem: given a large catalogue and a user with unknown preferences over most of it, surface the items most likely to be relevant.

This project builds a movie recommendation engine from scratch on the [MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest` dataset. Four algorithms are implemented behind one class interface and compared with four simple baselines on the same held-out ratings. The models are served through a FastAPI layer, and a live demo runs on Streamlit Cloud.

**The main result:** UV decomposition is the most accurate model, with an RMSE of 0.770 against 0.850 for a baseline that only knows user and movie averages. But when each model has to pick 10 movies from the whole catalogue, none of the four beats a plain most-popular list. [Results](#results) covers both findings and why they differ.

---

## The Four Approaches

**Content-Based Filtering** builds a profile for each user from the metadata of movies they have rated. Genres and genome tags with relevance ≥ 0.5 form a per-movie corpus, vectorised with TF-IDF. A user profile is the sum of the vectors of the movies they rated, weighted by each rating minus that user's mean, so liked movies pull the profile toward their content and disliked ones push it away. A prediction is the user's mean rating plus the cosine similarity between profile and movie, scaled by the standard deviation of ratings. The model uses the user's own ratings but nothing from other users.

**User-Based Collaborative Filtering** predicts from users with similar taste. Each rating is first reduced to its residual: what remains after subtracting the global mean and damped user and movie offsets. Users are compared by cosine similarity on these residuals, shrunk when two users share few movies. A prediction is the baseline plus a similarity-weighted average of the residuals of up to 50 neighbours who rated the movie.

**Item-Based Collaborative Filtering** applies the same idea to movies. Adjusted cosine similarity is computed on residuals, shrunk when few users rated both movies, and each movie keeps its 200 strongest neighbours. A prediction is the baseline plus a similarity-weighted average of the user's residuals on neighbouring movies. Similarities are computed in blocks, so the full item-item matrix never has to fit in memory.

**UV Decomposition (Matrix Factorisation)** factorises the rating matrix into a user factor matrix P and an item factor matrix Q, each with 50 latent dimensions. Predicted ratings are μ + b_u + b_i + P·Q. All parameters are learned with mini-batch SGD and L2 regularisation for 43 epochs, the stopping point chosen on the validation split.

**Baselines** give the four models something to beat: random order, popularity (most-rated movies), item mean, and a damped user and movie bias model.

---

## Dataset

- **Source**: [MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest`, GroupLens Research, generated July 20, 2023
- **Raw scale**: 33.8M ratings, 86K movies, 331K users
- **After co-filtering**: 25.9M ratings, 11,943 movies, 81,612 users

GroupLens describes `ml-latest` as a development dataset that may change over time, so a later download may not reproduce these numbers exactly.

An iterative co-filtering procedure retains only users and movies with at least 100 ratings, applied alternately until the dataset stabilises. A single pass is insufficient because removing low-activity users can drop a movie below the threshold and vice versa. The filter keeps 77% of all ratings. All results below therefore describe active users and established movies, not the full population.

The filtered utility matrix is still 97.34% sparse, which is the fundamental challenge all four models must overcome.

**Split**: Per-user chronological 80/20. The model sees each user's earlier ratings and is evaluated on their later ones, so no user's own future ratings are used to predict their past. This is not a global cut-off in time: training can include ratings other users made after a given user's test ratings, and co-filtering runs before the split, so it sees test-period activity. A single date cut-off would be a stricter test.

| Set | Ratings |
|---|---|
| Train | 20,721,823 |
| Test | 5,220,954 |

Hyperparameters were chosen by `tune.py` on a validation split, the last eighth of each user's training ratings. The test set played no part. Grids ran on 8,000 sampled users to fit modest hardware. The UV stopping epoch and the item similarity shrinkage were then rechecked on all users, because both depend on data size. The user-based settings were not, so they may be less well tuned than the others.

---

## Repository Structure

```
movie-recommendation-system/
├── train.py                     # Trains, evaluates and writes all artifacts
├── tune.py                      # Hyperparameter search on the validation split
├── requirements.txt             # Pinned dependencies for training and tests
├── data/
│   └── ReadMe.md                # Dataset version, license and citation
├── notebooks/
│   └── Recommender Report.ipynb # EDA, model walkthrough and results
├── src/
│   ├── base_model.py            # Abstract base class (fit / predict / recommend)
│   ├── baselines.py             # Random, popularity, item mean and bias baselines
│   ├── config.py                # Tuned hyperparameters
│   ├── content_model.py         # Content-based filtering
│   ├── data.py                  # Download, filtering and the temporal split
│   ├── evaluation.py            # Re-ranking and full-catalogue evaluation
│   ├── fold_in.py               # Places a new user in the UV model from a few ratings
│   ├── item_model.py            # Item-based collaborative filtering
│   ├── user_model.py            # User-based collaborative filtering
│   └── uv_model.py              # UV decomposition
├── results/
│   ├── metrics.json             # Every number in the Results section
│   └── tuning.json              # Validation results behind src/config.py
├── api/
│   ├── api.py                   # FastAPI application
│   ├── models/                  # Slim model artifacts, holding no user data
│   └── requirements.txt
├── streamlit/
│   ├── app.py                   # Streamlit demo
│   ├── recommender.py           # Search and recommendation logic
│   ├── links.csv
│   ├── requirements.txt
│   └── uv_model_deploy.joblib   # Lightweight deploy artifact
├── tests/                       # pytest suite, run by GitHub Actions
└── presentation/
    └── Recommender Presentation.pptx # Project slides
```

---

## Running It

Python 3.12 or later is required by the pinned dependencies. The project was developed on Python 3.13.

```bash
pip install -r requirements.txt
pytest tests           # about 10 seconds, no dataset needed
python train.py        # downloads the data, trains, evaluates, writes artifacts
python tune.py         # optional: repeats the hyperparameter search
```

The default `train.py` run took about 2 hours 15 minutes on a 2-core laptop with 4 GB of RAM. UV training and user-based evaluation account for nearly all of it.

The results below use the default sample. To compute rating metrics on all 81,612 test users:

```bash
python train.py --eval-users 0
```

Evaluating every user mostly costs user-based CF time, which grows with the number of users: about an hour for 5,000 users on the laptop above. `--eval-users 20000` is a middle ground, and `--models uv item` limits a run to selected models. Artifacts should be built with the pinned versions, because joblib files can fail to load under different library versions.

---

## Evaluation

All models are scored on the same fixed sample of 5,000 test users (318,303 held-out ratings). Full-catalogue metrics use 1,000 of those users, because they score every movie for each user. `results/metrics.json` records the sample with the results. These are sample estimates from one run without confidence intervals, so small gaps between models should not be over-read.

**Rating accuracy** measures how closely predicted ratings match actual ratings. RMSE penalises large errors more heavily, and MAE is the average absolute error in rating units.

**Re-ranking quality** asks how well a model orders the movies a user actually rated in the test period. Precision@10, Recall@10 and nDCG@10 use a relevance threshold of rating ≥ 3.5. These candidates are already movies the user chose to watch, and most of them are relevant, so scores start high: random order reaches a Precision@10 of 0.643. Read these columns against the baselines, not as absolute quality.

**Full-catalogue quality** is the harder task a real recommender faces. The model ranks every movie the user has not rated in training, and a top-10 pick counts as a hit only if the user rated it ≥ 3.5 in the test period. This also gives two measures of what is being recommended:

- **Coverage**: the share of the catalogue that appears in at least one user's top 10.
- **Popularity percentile**: how mainstream the recommended movies are on average, where 100 is the most-rated movie.

A low full-catalogue score does not mean the picks are bad. It means the user has not rated them in the test period, and they may simply never have seen them.

---

## Results

**Rating accuracy and re-ranking** (5,000 users):

| Model | RMSE | MAE | Precision@10 | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|
| Random | – | – | 0.643 | 0.252 | 0.772 |
| Popularity | – | – | 0.705 | 0.279 | 0.814 |
| Item mean | 0.942 | 0.724 | 0.781 | 0.317 | 0.855 |
| Bias (user + item) | 0.850 | 0.643 | 0.781 | 0.317 | 0.855 |
| Content-Based | 0.910 | 0.680 | 0.796 | 0.322 | 0.866 |
| User-Based CF | 0.790 | 0.591 | 0.824 | 0.336 | 0.885 |
| Item-Based CF | 0.779 | 0.581 | 0.827 | 0.339 | 0.887 |
| **UV Decomposition** | **0.770** | **0.579** | **0.835** | **0.342** | **0.892** |

Random and popularity produce rankings, not rating estimates, so they have no RMSE or MAE. Item mean and bias tie on ranking because a user offset cannot change the order of one user's movies.

**Full catalogue** (1,000 users):

| Model | Precision@10 | Recall@10 | nDCG@10 | Coverage | Popularity percentile |
|---|---|---|---|---|---|
| Random | 0.004 | 0.001 | 0.004 | 56.6% | 48.7 |
| **Popularity** | **0.116** | **0.034** | **0.127** | 1.2% | 99.9 |
| Item mean | 0.019 | 0.005 | 0.017 | 0.2% | 59.3 |
| Bias (user + item) | 0.029 | 0.009 | 0.026 | 0.2% | 79.9 |
| Content-Based | 0.056 | 0.017 | 0.060 | 15.3% | 80.2 |
| User-Based CF | 0.052 | 0.016 | 0.055 | 4.7% | 69.4 |
| Item-Based CF | 0.039 | 0.011 | 0.040 | 10.0% | 77.9 |
| UV Decomposition | 0.041 | 0.013 | 0.043 | 9.1% | 53.8 |

**What the numbers show:**

- **On the rating task, every collaborative model clearly beats the baselines.** UV leads on all five metrics and cuts RMSE from 0.850 to 0.770 against the bias model. The three collaborative models are close: UV and item-based CF differ by 0.009 RMSE. User-based and item-based CF share a baseline and a residual approach here, so this comparison does not show that one kind of similarity is inherently better.
- **On the full catalogue, none of the four models beats popularity.** Popularity scores about twice the best of them, and that gap is much larger than the differences between the models themselves. The likely reason is that a hit requires the user to have rated the movie, and what people watch next depends heavily on what is widely seen. These models are trained to estimate a rating, not to predict what someone will watch. This explanation is reasoned from the results, not tested separately.
- **Popularity's score comes at the cost of variety.** It recommends the same few movies to nearly everyone, reaching 1.2% of the catalogue. UV reaches 9.1% and recommends far less mainstream movies, with a popularity percentile of 53.8 against 99.9. So the personalised models do different work, but this evaluation cannot show whether users would prefer their picks, because it only sees movies users rated.
- **Good rating accuracy does not imply good top-10 selection.** Content-based has the worst RMSE of the four models but the best full-catalogue precision among them. With 1,000 users the order among the personalised models is indicative rather than settled.
- **Neighbour coverage is close to complete.** Item-based CF fell back to the baseline for 0.74% of test pairs and user-based CF for less than 0.01%. These fallbacks are counted in the metrics, not skipped.

The natural next step is a model trained for ranking on implicit feedback, such as BPR or weighted ALS, which learns what users choose to watch and not only how they score it. Blending popularity into the personalised scores is a simpler option.

### New users

The API and demo serve people who are not in the training data. A new user is placed in the UV model by a ridge regression of their ratings on the fixed item factors, with no retraining. That regression gives the predicted rating. Choosing the list uses the same calculation with two changes. No user offset is removed first, because people tend to rate only what they love, and removing it leaves little taste signal. And only the 2,000 most-rated movies can be recommended, because thinly rated movies have unreliable factors and otherwise win on noise. To test this, each of the 1,000 users was rebuilt from only their last few training ratings and evaluated on their test ratings:

| Ratings given | RMSE | RMSE from item averages alone | Precision@10 | Popularity Precision@10 |
|---|---|---|---|---|
| 5 | 0.857 | 0.942 | 0.040 | 0.044 |
| 20 | 0.829 | 0.942 | 0.044 | 0.045 |

Five ratings are enough to learn a lot about how someone will score movies, and the top-10 lists come close to a most-popular list while staying personal. Most of that precision comes from the 2,000-movie limit: choosing from every movie with the plain regression gave 0.017 and 0.022. The cost is that the demo cannot recommend lesser-known movies. The limit was set once and not tuned, but these same users were used to compare the options, so treat these two precision values as indicative.

The precision columns are not comparable with the full-catalogue table. This replay hides the user's older ratings, so movies they had already rated stay in the candidate list and can never count as hits. Both columns are affected, and popularity probably more, because popular movies are the most likely to have been rated already. The comparison within this table is like for like.

---

## API

The FastAPI layer serves three recommendation endpoints and three utility endpoints. Models are loaded once at startup, and requests only read them, so concurrent requests cannot interfere with each other. Artifacts hold no user data.

**Run locally:**
```bash
pip install -r api/requirements.txt
cd api
uvicorn api:app --reload
```

Interactive docs are available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/genres` | All available genres |
| `GET` | `/tags` | All single-word terms in the TF-IDF vocabulary |
| `GET` | `/movies/search?query=` | Plain-text title search, first 10 matches |
| `POST` | `/recommend/content` | Content-based recommendations |
| `POST` | `/recommend/item` | Item-based CF recommendations |
| `POST` | `/recommend/uv` | Matrix factorisation recommendations |

Recommendation endpoints return a list of `{"movieId", "title"}` objects, with `n` between 1 and 100. Ratings must be between 0.5 and 5.0. A request whose genres, tags or movies are all unknown to the model gets a 422 response explaining why.

**Content-based example:**
```json
POST /recommend/content
{
  "genres": ["Action", "Thriller"],
  "tags": ["suspense", "twist ending"],
  "n": 10
}
```

**Item-based example:**
```json
POST /recommend/item
{
  "movieIds": [318, 296, 593],
  "n": 10
}
```

**UV example:**
```json
POST /recommend/uv
{
  "rated_movies": [
    {"movieId": 318, "rating": 5.0},
    {"movieId": 296, "rating": 4.5}
  ],
  "n": 10
}
```

---

## Streamlit Demo

An interactive demo of the UV model runs on Streamlit Cloud. Users search for movies the model knows, rate at least 5, and receive 10 personalised recommendations. The ratings are turned into a temporary user as described under [New users](#new-users), so no retraining or stored history is involved. The deployed artifact is a lightweight dict of plain arrays: Q, the item biases, the fold-in setting, the candidate movies, and the movie IDs and titles.

**Run locally:**
```bash
pip install -r streamlit/requirements.txt
streamlit run streamlit/app.py
```

Posters are optional. To show them, add `TMDB_API_KEY = "..."` to `.streamlit/secrets.toml`. On Streamlit Cloud, choose Python 3.12 or later in the app settings.

**Live demo:** *[Movie Recommendation System](https://movie-recommender-billygeros.streamlit.app/)*

---

## Key Design Decisions

- **Iterative co-filtering** rather than a single-pass threshold, because removing users can cascade into movie removals and vice versa.
- **Per-user temporal split** rather than random, so each user is tested on ratings that came after the ones the model saw.
- **Baselines and full-catalogue metrics** alongside the usual table, because re-ranking scores alone made every model look strong.
- **Hyperparameters chosen on a validation split**, keeping the test set out of every decision.
- **Residuals for both CF models**, so similarity reflects shared taste rather than how generously people rate or how many movies they share.
- **Sparse matrix storage** (CSR/CSC) throughout, to keep 26M ratings in memory without materialising a dense 81K × 12K matrix.
- **Batched evaluation** and block-wise similarity computation, which let the whole pipeline run in 4 GB of RAM.
- **Stateless serving**: new users are folded in with a read-only calculation, so requests never change a shared model.
- **Slim artifacts** that drop training data, keeping all models small enough for the repository and free of user ratings.

---

## License and Data

The source code is under the [MIT License](LICENSE). The MovieLens data and files derived from it, including `streamlit/links.csv` and the model artifacts, stay under the MovieLens terms, which rule out commercial use without permission. See [data/ReadMe.md](data/ReadMe.md).

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

Movie posters come from the TMDB API. This product uses the TMDB API but is not endorsed or certified by TMDB.
