# Dataset

This project uses the **MovieLens Latest Dataset** provided by GroupLens Research.

Dataset source:
https://grouplens.org/datasets/movielens/

## Data Acquisition

The dataset is downloaded automatically during execution of the main notebook using the official GroupLens URL.

The pipeline retrieves and extracts the following files directly from the compressed MovieLens archive:

- `ratings.csv`
- `movies.csv`
- `genome-tags.csv`
- `genome-scores.csv`

No manual dataset download or local data preparation is required.

## Dataset Overview

The original MovieLens Latest Dataset contains:

- ~33.8 million ratings
- ~330,000 users
- ~86,000 movies

After iterative co-filtering, retaining only users and movies with at least 100 ratings, the dataset used for modelling contains approximately:

- ~26 million ratings
- ~81,000 users
- ~12,000 movies

## Files Description

### ratings.csv
Explicit user ratings.

Columns:
- `userId`
- `movieId`
- `rating`
- `timestamp`

Used for:
- temporal train/test splitting
- collaborative filtering
- matrix factorisation
- model evaluation

### movies.csv
Movie metadata.

Columns:
- `movieId`
- `title`
- `genres`

Used for:
- content-based recommendations
- displaying recommendation results

### genome-tags.csv
Semantic movie tags.

Columns:
- `tagId`
- `tag`

Used to enrich movie representations.

### genome-scores.csv
Movie-tag relevance scores.

Columns:
- `movieId`
- `tagId`
- `relevance`

Used to select high-relevance tags for the TF-IDF content representation.