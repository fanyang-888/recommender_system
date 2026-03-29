# News Recommender System (30-second recruiter view)

Built a news recommender system to predict users' next-clicked article from historical click logs using candidate generation and ranking baselines.

## Results First

### Dataset Size
- Train clicks: **1,112,623** rows (**200,000** users)
- TestA clicks: **518,010** rows (**50,000** users)
- TestB clicks: **776,657** rows (**50,000** users)
- Articles: **364,047**

### Task Definition
Given each user's historical click sequence, predict the most likely next-clicked article.

### Offline Metrics (summary)
- Evaluation protocol: Recall-style candidate evaluation and top-K ranking quality checks.
- Core metrics: **Recall@20**, **MRR@20**, **HitRate@20**
- Current status: **metrics table will be filled from notebook evaluation export**

### Best Model vs Baseline
- Placeholder (to be filled after metric extraction): **TBD**

### Top Business Takeaway
- Even simple retrieval + ranking decomposition gives a practical path from prototype to deployable recommendation systems.

## Project Structure
- `notebook/`
  - `Data_Analysis.ipynb`: data exploration and preprocessing flow.
  - `Model.ipynb`: candidate generation and ranking prototype.
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
- This repository tracks source notebooks and code only.
- Large raw data files and generated outputs are excluded from version control.
