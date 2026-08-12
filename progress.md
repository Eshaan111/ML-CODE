		# ML Learning Progress

## Current Status

- **Long-term goal:** Top-tier, job-ready generalist ML engineer
- **Current roadmap node:** 2 — Titanic Survival Prediction
- **Current phase:** Beginner foundations
- **Status:** In progress
- **Started tracking:** 2026-07-12

## Mastery Scale

Use these levels when reporting progress:

| Level             | Meaning                                           |
| ----------------- | ------------------------------------------------- |
| 0 — Not started   | No meaningful exposure yet                        |
| 1 — Exposure      | Watched, read, or followed along                  |
| 2 — Understanding | Can explain the central idea in your own words    |
| 3 — Application   | Can use the idea in a project with limited help   |
| 4 — Mastery       | Can rebuild, debug, modify, compare, and teach it |

## Roadmap Progress

| # | Project | Status | Mastery | Evidence / notes |
|---:|---|---|---:|---|
| 1 | Linear Regression | Completed or previously studied | To assess | Add implementation, evaluation results, and explanation |
| 2 | Titanic Survival Prediction | In progress | To assess | Current project |
| 3 | Housing Price Prediction | Not started | 0 | — |
| 4 | CNN Cats vs Dogs | Not started | 0 | — |
| 5 | Sentiment Analysis | Not started | 0 | — |
| 6 | Product-Based Sentiment System | Not started | 0 | — |
| 7 | Customer Churn Predictor | Not started | 0 | — |
| 8 | Stock-Price Forecasting Experiment | Not started | 0 | — |
| 9 | Neural Network From Scratch | Not started | 0 | — |
| 10 | Real-Time Face Recognition / Verification | Not started | 0 | — |
| 11 | Recommendation System | Not started | 0 | — |
| 12 | Automated ML Pipeline | Not started | 0 | — |
| 13 | Language Model From Scratch | Not started | 0 | — |
| 14 | A/B-Testing Framework | Not started | 0 | — |
| 15 | Image-Generation System | Not started | 0 | — |
| 16 | Multilingual NLP Pipeline | Not started | 0 | — |
| 17 | Reinforcement-Learning Game AI | Not started | 0 | — |
| 18 | Real-Time Fraud-Detection System | Not started | 0 | — |
| 19 | 10x Capstone | Not started | 0 | — |

## Initial Titanic Skill Checklist (superseded by current assessment below)

| Skill | Status | Mastery | Evidence / question to answer |
|---|---|---:|---|
| Load and inspect data | To assess | — | Can I explain every column and identify the target? |
| Exploratory data analysis | To assess | — | What patterns and possible biases did I find? |
| Missing-value handling | To assess | — | Why did I choose each imputation method? |
| Categorical encoding | To assess | — | Why does the model require encoding? |
| Train/validation splitting | To assess | — | What information must remain unseen during training? |
| Baseline model | To assess | — | What simple result must my model beat? |
| Logistic regression | To assess | — | What does its output represent? |
| Decision tree | To assess | — | How can it overfit? |
| Classification metrics | To assess | — | When are accuracy, precision, recall, and F1 useful? |
| Cross-validation | To assess | — | Why is one split potentially misleading? |
| Leakage prevention | To assess | — | Did any preprocessing learn from validation data? |
| Error analysis | To assess | — | Which passengers does the model get wrong, and why? |
| Reproducible pipeline | To assess | — | Can training be rerun consistently from raw data? |
| Explanation and README | To assess | — | Can someone understand the project without opening every file? |

## Current Titanic Skill Checklist

| Skill                      | Status      | Mastery | Current evidence / next requirement                                                                                        |
| -------------------------- | ----------- | ------: | -------------------------------------------------------------------------------------------------------------------------- |
| Load and inspect data      | In progress |       3 | Loaded the CSV and used `describe()`, `info()`, value counts, and correlations; full column explanation remains to verify. |
| Exploratory data analysis  | In progress |       2 | Inspected correlations and missingness; needs survival-rate plots and bias-aware interpretation.                           |
| Missing-value handling     | In progress |       2 | Correctly explained why the test mean must not be learned; implementation must store fitted state in `fit()`.              |
| Categorical encoding       | In progress |       3 | Built a reusable [[OneHotEncoder]] and `ColumnTransformer` wrapper with unknown-category handling.                         |
| Train/validation splitting | In progress |       2 | Used a stratified 80/20 split and understands that test information must remain unseen during fitting.                     |
| Baseline model             | Not started |       0 | Create a majority-class baseline before comparing trained models.                                                          |
| [[Logistic regression]]    | Not started |       0 | Add the first learned classification baseline and interpret its probability output.                                        |
| Decision tree              | Not started |       0 | Compare training and validation performance to observe overfitting.                                                        |
| Classification metrics     | In progress |       1 | Used [[accuracy]]; [[precision]], [[Recall]], [[F1 Score]], and [[Confusion Matrix]] remain to learn and apply.            |
| Cross-validation           | In progress |       2 | Used [[GridSearchCV]] ; preprocessing must move inside the searched pipeline to prevent fold leakage.                      |
| Leakage prevention         | In progress |       2 | Can explain that `fit` learns, `transform` applies, and test-set feature statistics cause leakage.                         |
| Error analysis             | Not started |       0 | Identify false positives, false negatives, and passenger groups with recurring errors.                                     |
| Reproducible pipeline      | In progress |       2 | Built custom transformers and a pipeline; fitted-state conventions and model integration need correction.                  |
| Explanation and README     | Not started |       0 | Document the final workflow, results, limitations, and reproduction steps.                                                 |

## Session Log

Add one row after each meaningful study or building session.

| Date | Time spent | What I worked on | What I learned | What confused me | Evidence produced | Next action |
|---|---:|---|---|---|---|---|
| 2026-07-12 | — | Established roadmap and current Titanic position | Defined the long-term project ladder and mastery standard | Current Titanic knowledge still needs assessment | `plan.md` and `progress.md` | Record exact Titanic progress |
| 2026-07-15 | Not recorded | Reviewed both Titanic notebooks; studied leakage, `fit` versus `transform`, custom estimators, fitted-state attributes, and GPU suitability | Test features must never be fitted; `BaseEstimator` provides parameter compatibility; `TransformerMixin` supplies transformer conveniences; fitted attributes conventionally end in `_` | A stateless final transformer caused `Pipeline.transform()` to raise `NotFittedError` because it recorded no fitted-state attribute | Explained the leakage mechanism correctly and diagnosed the pipeline traceback | Correct `AgeImputer` and `Dropper`, recreate the pipeline, then verify `fit_transform(train)` followed by `transform(test)` |

## Weekly Review

Complete this at the end of each week:

- **What did I finish?**
- **What can I now explain without notes?**
- **What can I build without copying?**
- **Where did I get stuck?**
- **What evidence did I create?**
- **Was the workload realistic?**
- **What should change next week?**

## Current Blockers and Questions

- The custom `AgeImputer` must learn and store its imputation state during `fit()` rather than fitting inside `transform()`.
- The final stateless `Dropper` must record fitted state so the current scikit-learn pipeline passes its fitted check.
- Preprocessing and model selection are not yet integrated into one leakage-safe cross-validation pipeline.
- Linear Regression mastery and completion evidence need confirmation.

