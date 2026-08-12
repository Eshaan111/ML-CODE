# ML Engineer Roadmap

## Goal

Become a top-tier, job-ready generalist ML engineer who can take an unclear problem from data exploration through modelling, deployment, monitoring, and explanation.

The projects remain the spine of the journey because they make learning enjoyable and force practical problem-solving. Prerequisites should be learned just in time instead of delaying projects indefinitely.

## Project Roadmap

|   # | Project / node                            | Prerequisites                                                                        | Key learning to extract                                                                                                               | Completion evidence                                                                                    |
| --: | ----------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
|   1 | Linear Regression                         | Basic Python, NumPy, pandas, plotting, and algebra                                   | Features and targets, loss functions, gradient-descent intuition, coefficients, residuals, MAE, MSE, RMSE, and train/test splitting   | Implement simple regression manually and with scikit-learn; explain coefficients and errors            |
|   2 | Titanic Survival Prediction               | pandas, data cleaning, classification basics, and linear-regression concepts         | EDA, missing values, categorical encoding, logistic regression, decision trees, classification metrics, cross-validation, and leakage | Build a reproducible pipeline, compare baselines, produce a confusion matrix, and document experiments |
|   3 | Housing Price Prediction                  | Linear regression and Titanic-level preprocessing                                    | Regression pipelines, feature engineering, skewed data, regularization, tree ensembles, and error analysis                            | Compare a baseline, regularized regression, random forest, and boosting with cross-validation          |
|   4 | CNN Cats vs Dogs                          | Python, NumPy, basic linear algebra, neural-network fundamentals, and PyTorch        | Tensors, forward and backward propagation, convolutions, pooling, augmentation, overfitting, and transfer learning                    | Train a basic CNN, diagnose overfitting, and compare it with transfer learning                         |
|   5 | Sentiment Analysis                        | Titanic-level classification, text preprocessing, and evaluation metrics             | Tokenization, bag-of-words, TF-IDF, embeddings, class imbalance, NLP evaluation, and error analysis                                   | Compare a TF-IDF baseline with a neural or transformer model                                           |
|   6 | Product-Based Sentiment System            | Sentiment analysis, data collection, and grouping                                    | Aspect-based sentiment, product-level aggregation, noisy labels, and dashboards or APIs                                               | Produce product sentiment summaries and explain common failures                                        |
|   7 | Customer Churn Predictor                  | Classification, feature engineering, and cross-validation                            | Imbalanced classes, precision/recall trade-offs, probability calibration, thresholds, explainability, and business costs              | Select a threshold using business costs and explain predictions with feature importance or SHAP        |
|   8 | Stock-Price Forecasting Experiment        | Regression, time-indexed data, and basic statistics                                  | Temporal splitting, walk-forward validation, lag features, baselines, non-stationarity, and leakage prevention                        | Honestly test whether the model beats a naive baseline using valid temporal evaluation                 |
|   9 | Neural Network From Scratch               | NumPy, functions, derivatives, matrix multiplication, and linear/logistic regression | Activations, losses, forward propagation, backpropagation, initialization, and optimization                                           | Build and train a small network without framework autograd                                             |
|  10 | Real-Time Face Recognition / Verification | CNNs, embeddings, transfer learning, and basic OpenCV                                | Detection versus recognition, embedding similarity, thresholds, real-time inference, latency, bias, and privacy                       | Build a consent-based controlled verification demo and measure accuracy and latency                    |
|  11 | Recommendation System                     | pandas, linear algebra, and supervised-ML evaluation                                 | Popularity baselines, collaborative filtering, matrix factorization, implicit feedback, ranking, and cold start                       | Compare popularity and personalized models with Precision@K, Recall@K, or NDCG                         |
|  12 | Automated ML Pipeline                     | Several completed ML projects, Git, and modular Python                               | Reusable preprocessing, configuration, experiment tracking, testing, reproducibility, and scheduled training                          | Use one command to train, evaluate, version, and save a model, with tests                              |
|  13 | Language Model From Scratch               | Neural network from scratch, PyTorch, NLP, probability, and attention                | Tokenization, embeddings, self-attention, transformers, next-token prediction, sampling, and scaling limits                           | Implement and train a small transformer and explain every major component                              |
|  14 | A/B-Testing Framework                     | Probability, hypothesis testing, confidence intervals, and basic SQL                 | Experiment design, power, sample size, randomization, p-values, effect sizes, and guardrail metrics                                   | Simulate and analyze experiments while detecting common statistical mistakes                           |
|  15 | Image-Generation System                   | CNNs, PyTorch, probability, and generative-model fundamentals                        | Autoencoders, GANs or diffusion, latent representations, conditioning, sampling, and evaluation limitations                           | Train a small generator or adapt an existing diffusion model with documented experiments               |
|  16 | Multilingual NLP Pipeline                 | Sentiment analysis, transformers, and evaluation design                              | Multilingual tokenization, cross-lingual transfer, translation effects, per-language evaluation, and bias                             | Evaluate each language separately and analyze uneven failures                                          |
|  17 | Reinforcement-Learning Game AI            | Neural networks, probability, optimization, and Markov decision processes            | States, actions, rewards, policies, values, exploration, Q-learning or DQN, and training instability                                  | Train an agent that improves over a random baseline and chart its learning curves                      |
|  18 | Real-Time Fraud-Detection System          | Churn-style classification, pipelines, APIs, and database or streaming basics        | Severe imbalance, cost-sensitive learning, low-latency inference, concept drift, monitoring, and human review                         | Build a simulated end-to-end stream with alerts, latency metrics, and drift monitoring                 |
|  19 | 10x Capstone                              | Multiple completed projects and production ML skills                                 | System design, independent research, ambiguity management, cost/performance trade-offs, and ownership                                 | Design, ship, monitor, and defend a serious end-to-end ML product                                      |

## Supporting Skills

| Skill | Introduce around | Expected outcome |
|---|---|---|
| Git and GitHub | Titanic | Meaningful commits, clean repositories, and useful READMEs |
| SQL | Titanic / Housing | Query, join, aggregate, and prepare modelling data |
| Statistics | Linear Regression onward | Evaluate evidence and uncertainty correctly |
| Clean Python architecture | Churn | Move beyond one large notebook |
| Unit testing | Churn / Automated Pipeline | Test transformations, schemas, and inference |
| Prediction APIs | Churn / Recommendations | Serve predictions through FastAPI or a similar framework |
| Docker | First deployed project | Package an application reproducibly |
| Experiment tracking | CNN / Automated Pipeline | Record parameters, metrics, and model artifacts |
| Cloud basics | After local deployment | Deploy a small service and understand compute and storage |
| Monitoring | Automated Pipeline / Fraud | Track latency, errors, distributions, drift, and model quality |
| ML system design | Recommendations onward | Reason about data, training, serving, and feedback loops |

## Project Mastery Standard

Each project moves through five stages:

1. **Make it work:** complete a functioning version with guidance.
2. **Understand it:** explain the major code and modelling decisions.
3. **Improve it:** run deliberate experiments and interpret their results.
4. **Rebuild it:** reproduce the important pieces with less assistance.
5. **Present it:** publish clean code, results, limitations, and documentation.

A working prediction is not by itself a completed project. Completion requires evidence that the model was evaluated correctly and that its limitations are understood.

## Current Position

**Node 2: Titanic Survival Prediction**

### Current focus

Build a leakage-safe scikit-learn workflow and understand the estimator interface before comparing models.

Immediate sequence:

1. Correct the custom transformers so `fit()` learns from training data and `transform()` only applies learned state.
2. Put preprocessing and the estimator inside one pipeline so cross-validation cannot leak information between folds.
3. Establish a majority-class baseline.
4. Compare logistic regression, a decision tree, and a random forest.
5. Evaluate accuracy, precision, recall, F1, and the confusion matrix.
6. Perform cross-validation and error analysis before treating the project as complete.

GPU acceleration is deferred until a workload benefits from it, especially the CNN project. The current Titanic dataset and scikit-learn random forest should use CPU parallelism.

