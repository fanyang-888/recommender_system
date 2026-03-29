# News Recommender System (30-second recruiter view)

Built a news recommender system to predict users' next-clicked article from historical click logs using candidate generation and ranking baselines.

## 30-Second Snapshot

- **Dataset**: 2.41M click rows across train/test files, 364K articles.
- **Task**: next-click prediction from user historical behavior.
- **Offline setup**: leave-one-out validation on 5,000 sampled users.
- **Best baseline**: ItemCF reaches **Recall@20 = 0.4962**, **MRR@20 = 0.2118**.
- **Practical takeaway**: retrieval + ranking decomposition is effective and production-friendly.

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
- Evaluation setup: leave-one-out validation on **5,000 sampled training users**.

### Best Model vs Baseline
- Best Recall@20: **ItemCF (0.4962)** vs **Popularity baseline (0.4326)**, absolute gain **+0.0636**.
- Best MRR@20: **ItemCF (0.2118)** vs **Popularity baseline (0.1418)**, absolute gain **+0.0700**.

### Top Business Takeaway
- Even simple retrieval + ranking decomposition gives a practical path from prototype to deployable recommendation systems.

## Method Framework

```mermaid
flowchart LR
  rawClickLogs[RawClickLogs] --> preprocessing[Preprocessing]
  preprocessing --> candidateGeneration[CandidateGeneration]
  candidateGeneration --> featureEngineering[FeatureEngineering]
  featureEngineering --> rankingModel[RankingModel]
  rankingModel --> offlineEvaluation[OfflineEvaluation]
```

- **Candidate generation** narrows the search space from all articles to a relevant top-K set.
- **Feature engineering + ranking** estimates fine-grained click likelihood among candidates.
- **Offline evaluation** quantifies retrieval and ranking quality before any online deployment.

## Experiment Results

| Model | Recall@20 | MRR@20 | HitRate@20 |
| :-- | --: | --: | --: |
| Popularity baseline | 0.4326 | 0.1418 | 0.4326 |
| ItemCF | 0.4962 | 0.2118 | 0.4962 |
| Embedding recall | 0.2458 | 0.0566 | 0.2458 |
| Ranker (hybrid) | 0.4574 | 0.1921 | 0.4574 |

Metric source: exported from `notebook/Model.ipynb` offline evaluation section.

## Recommendation Output Example

Example user (illustrative format from the notebook inference flow):

- History clicks: `[article_1024, article_877, article_65021, article_9012]`
- Retrieved Top-K candidates (K=10): `[article_9012, article_441, article_11870, article_3002, article_7744, article_811, article_52300, article_9872, article_22017, article_611]`
- Re-ranked Top-5 recommendations: `[article_441, article_11870, article_7744, article_3002, article_811]`

The output shape follows the repository submission format (`user_id + article_1...article_5`).

## Conclusions and Limitations

- Cold-start remains a practical bottleneck, especially for fresh articles with sparse interactions.
- Offline metrics are useful for iteration speed, but do not directly guarantee online CTR uplift.
- The current implementation is notebook-first; engineering next step is modular pipeline packaging.

## Next Steps

- Split retrieval and ranking into reusable Python modules.
- Add experiment tracking for versioned metrics and parameter sweeps.
- Prepare an online serving interface and A/B-test-ready evaluation loop.

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
