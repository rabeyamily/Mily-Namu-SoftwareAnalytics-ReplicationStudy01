# Replication Study: Impact of CI on Pull Request Delivery Time

**Course:** CS-UH 3260 Software Analytics, NYUAD

---

## 1. Project Title and Overview

**Paper Title:** Studying the Impact of Adopting Continuous Integration on the Delivery Time of Pull Requests

**Authors:** João Helis Bernardo, Daniel Alencar da Costa, and Uirá Kulesza

**Replication Team:** Rabeya Zahan Mily & Namu Go

**Brief Description:**

The original paper analyzes 162,653 pull requests from 87 GitHub projects to study how adopting Continuous Integration (CI) affects the time to deliver merged pull requests. The authors address two research questions: (RQ1) whether merged PRs are released more quickly with CI, and (RQ2) whether increased development activity after CI adoption increases delivery time. They use Mann–Whitney U tests and Cliff’s Delta for RQ1, and Wilcoxon signed-rank tests for RQ2.

This replication reproduces RQ1 and RQ2 on the provided dataset of 87 projects. We implemented analysis scripts that run the same statistical tests (Mann–Whitney, Wilcoxon, Cliff’s Delta) on the shared PR and release meta-data, and we report results and figures consistent with the paper (e.g., ~51% of projects with faster delivery time after CI; majority with increased merge time after CI; increased PR submission and churn after CI).

---

## 2. Repository Structure

```
Replication Study 01/
├── README.md                          # This file – documentation for the repository
├── requirements.txt                   # Python package dependencies and versions
├── replication.py                     # Main replication script (RQ1 + RQ2 analysis)
├── pull_requests_meta_data.csv        # PR meta-data (from paper artifact; see Setup)
├── releases_meta_data.csv             # Release meta-data (from paper artifact; see Setup)
├── results/                           # Generated result files (outputs)
│   ├── rq1_delivery_time_results.csv  # Per-project RQ1 results for delivery_time
│   ├── rq1_merge_time_results.csv    # Per-project RQ1 results for merge_time
│   ├── rq1_lifetime_results.csv      # Per-project RQ1 results for lifetime
│   ├── rq1_summary.csv                # RQ1 summary table
│   ├── rq2_aggregate_results.csv     # RQ2 aggregate statistics table
│   ├── rq2_created_pull_requests_results.csv
│   ├── rq2_merged_pull_requests_results.csv
│   ├── rq2_released_pull_requests_results.csv
│   ├── rq2_sum_submitted_pr_churn_results.csv
│   └── rq2_release_frequency_results.csv
└── figures/                           # Generated visualizations (outputs)
    ├── rq1_summary.png               # RQ1 boxplots: delivery time, merge time, lifetime
    └── rq2_summary.png               # RQ2 boxplots: PR and churn metrics per release
```

| Item | Description |
|------|-------------|
| **replication.py** | Single script that loads data, runs RQ1 and RQ2 analyses (Mann–Whitney, Wilcoxon, Cliff’s Delta), writes CSVs to `results/` and PNGs to `figures/`. |
| **requirements.txt** | Lists Python packages and minimum versions needed to run `replication.py`. |
| **pull_requests_meta_data.csv** | Full PR dataset from the paper’s artifact (162,653 PRs, 87 projects). |
| **releases_meta_data.csv** | Full release dataset from the paper’s artifact (7,440 releases). |
| **results/** | All CSV outputs produced by `replication.py`. |
| **figures/** | All PNG figures produced by `replication.py`. |

---

## 3. Setup Instructions

### Prerequisites

- **OS:** Any (tested on macOS; should work on Linux/Windows with Python 3).
- **Python:** 3.8 or higher.
- **Required packages (see `requirements.txt`):**
  - pandas ≥ 1.3.0  
  - numpy ≥ 1.21.0  
  - scipy ≥ 1.7.0  
  - matplotlib ≥ 3.4.0  
  - seaborn ≥ 0.11.0  

No other dependencies or environment variables are required.

### Installation Steps

1. **Clone or download this repository**  
   Ensure `replication.py` and `requirements.txt` are in the same directory.

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Obtain the datasets**  
   Place the two CSV files in the **same directory as `replication.py`**:
   - `pull_requests_meta_data.csv`
   - `releases_meta_data.csv`  

   They are included in the repo, but they can be downloaded from the paper’s artifact as well:
   - [Pull requests meta-data](https://prdeliverydelay.github.io/data/pull_requests_meta_data.csv)  
   - [Releases meta-data](https://prdeliverydelay.github.io/data/releases_meta_data.csv)  

   If downloaded from paper's articraft then save them into the project root.

5. **Run the replication:**
   ```bash
   python replication.py
   ```

   The script will create `results/` and `figures/` if they do not exist, then write all CSVs and PNGs. No path or config changes are needed if the CSVs are in the same folder as `replication.py`.

---

## 4. GenAI Usage

**Tools used:** Claude

**How they were used:** Claude was used to generate code for producing the boxplots in a clean and readable manner, and to fix certain bugs in the replication scripts.

**Brief description:** GenAI assisted with visualization code (matplotlib/seaborn boxplots for RQ1 and RQ2 summary figures) and debugging. For example, a bug was fixed where NaN values in group-wise statistical tests caused runtime errors when some projects had empty before- or after-CI groups; the fix added proper NaN filtering before calling Mann–Whitney and Wilcoxon tests. The core analysis logic, statistical tests, and data processing were implemented by the replication team. 

## Acknowledgement

This README follows the CS-UH 3260 Software Analytics replication repository template. The replication study reproduces the analysis of Bernardo et al. (2018) for RQ1 and RQ2 using the artifact data and methodology.