# Dataset

This project uses the **MovieLens Latest Dataset** provided by GroupLens Research.

Dataset source:
https://grouplens.org/datasets/movielens/

## Data Acquisition

`python train.py` downloads the dataset from the official GroupLens URL into this folder on first run, then reuses it. These files are ignored by git.

The pipeline uses these files from the MovieLens archive:

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

## Dataset Version

This is `ml-latest`, generated on July 20, 2023. GroupLens describes it as a development dataset that may change over time and is not appropriate for shared research results, so a later download may not reproduce these numbers exactly. The stable alternatives are the GroupLens benchmark datasets.

## License and Citation

The data belongs to GroupLens Research and is not covered by this repository's MIT license. Its terms apply to the data and to files derived from it, including `streamlit/links.csv` and the model artifacts in `api/models/` and `streamlit/`. In summary:

- Do not state or imply endorsement by the University of Minnesota or GroupLens.
- Acknowledge the data set in publications.
- Redistribution, including transformations, must be under the same license conditions.
- No commercial or revenue-bearing use without first getting permission from GroupLens.

The full terms are in the `README.txt` inside the dataset archive.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
