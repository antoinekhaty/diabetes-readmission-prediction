# Predicting 30-Day Diabetes Readmission | Tree Ensembles

**Applied machine learning portfolio project | Antoine Khati, MD, MS**

This repository documents a collaborative University of Connecticut data-mining project on early readmission after diabetes-related hospital encounters. I contributed to preprocessing, exploratory analysis, model comparison, interpretation, and model selection. The course team used JMP; this repository provides a **separate, reproducible Python companion demonstration**.

## Original course project: verified results

| Detail | Original coursework |
| --- | --- |
| Dataset | 101,766 admissions at 130 U.S. hospitals, 1999–2008 |
| Outcome | Readmission within 30 days of discharge |
| Compared models | Logistic regression, decision tree, boosted tree, bootstrap forest |
| Selected model | Boosted tree |
| Reported test AUC | **0.620** |
| Reported test lift | **1.856**, according to the group's own lift definition |

The team cleaned and filtered the dataset before modeling, considered class imbalance, and favored the boosted tree for its minority-class performance relative to alternatives. A test AUC of 0.620 indicates **modest discrimination**. High accuracy in an imbalanced readmission task should be interpreted cautiously.

## Reproducible companion implementation

`src/train.py` demonstrates a new Python workflow using:

- First-encounter patient deduplication and adult-age filtering for the public dataset
- Training-fitted imputation, scaling, and one-hot encoding in scikit-learn `Pipeline` components
- Logistic regression, random forest, and XGBoost comparison
- Stratified train/test splitting and evaluation using ROC-AUC, average precision, precision, recall, and top-decile lift

**The Python demo does not reproduce the original JMP results.** Demo mode generates synthetic records with fabricated predictors. Its results only verify that the example pipeline runs.

## Run

Python 3.10+:

```bash
python -m pip install -r requirements.txt
python src/train.py --demo
python -m pytest -q
```

To explore the public [Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008), download `diabetic_data.csv` locally and run:

```bash
python src/train.py --csv /path/to/diabetic_data.csv
```

No original data file, class report, or graded assignment is included.

## Interpreting performance

The code's **top-decile lift** is different from the lift reported in the original course project. These figures must not be compared directly. The code also reports average precision because the positive outcome is uncommon. Results would require independent external validation and additional fairness and calibration assessment before any practical application.

## Repository files

| Path | Purpose |
| --- | --- |
| `src/train.py` | Dataset preparation, models, and evaluation |
| `tests/test_smoke.py` | Synthetic-data tests |
| `requirements.txt` | Python dependencies |
| `data/README.md` | Data source and handling notes |

**Attribution:** Original course project completed collaboratively with Sai Nithya Reddy Chandrapu, Dharani Dharan Kanakaraju Jayanthi, and Alexis Lohrey at UConn. Original metrics come from the group report. The public Python code is a separate educational implementation.

**Research/education only. Do not use these models for clinical decision making or patient resource allocation.**
