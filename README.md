# Recommender System (English Edition)

This repository contains notebooks for a news recommendation project. The primary task is to predict the next article a user is likely to click based on historical click behavior.

## Project Structure

- `notebook/`
  - `Data_Analysis.ipynb`: data exploration and preprocessing flow.
  - `Model.ipynb`: recall/modeling baseline and related experiments.
- `data/`
  - `data_raw/`: place raw dataset files here (ignored by Git).
  - `temp_results/`: generated intermediate/result files (ignored by Git).

## Required Data Files

Put the following files under `data/data_raw/` before running notebooks:

- `train_click_log.csv`
- `testA_click_log.csv`
- `testB_click_log.csv`
- `articles.csv`
- `articles_emb.csv`

## Run

1. Create a Python environment (3.9+ recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open notebooks from the repository root:

   - `notebook/Data_Analysis.ipynb`
   - `notebook/Model.ipynb`

## Notes

- This repo keeps source notebooks and code only.
- Large raw data files and generated results are intentionally excluded from version control.
