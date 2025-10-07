# FX Price-Change Classification with TensorFlow/Keras (22-Class)

End-to-end TensorFlow pipeline that predicts **next-hour price-change bins** for a CAD FX series. The project covers **TFRecord creation**, a **custom Keras imputer**, **temporal embeddings** (weekday/hour/month), **Keras Tuner** hyperparameter search, and **label smoothing** for imbalanced targets.

## Introduction

We formulate short-horizon price movement as a **22-class classification** task by digitizing the fractional change between *next-hour high* and *current-hour close*. The pipeline emphasizes reproducibility and modularity (data → TFRecords → model).

When modeling financial targets, two risks exist:
1. **Overfitting noisy signals** → poor generalization  
2. **Ignoring imbalance** → overconfident yet unhelpful predictions

We mitigate these via careful preprocessing, regularization, label smoothing, and early stopping.

## Business Understanding

Traders and analysts need **probabilistic direction/size** signals to triage opportunities. Rather than a single point forecast, a calibrated class distribution helps combine with **cost/utility** and downstream decision rules.

## Business Objectives

- Build a robust multi-input Keras model for **22-class** next-hour movement  
- Encode **calendar effects** (weekday/hour/month) with embeddings  
- Address **class imbalance** with **label smoothing**  
- Tune architecture/training via **Keras Tuner RandomSearch** and report held-out results

## Analytical Approach

1. **Data Preparation**
   - Read the source pickle (`appml-assignment1-dataset-v2.pkl`)  
   - Compute fractional change: `(next_high − current_close) / current_close`  
   - **Digitize** into **22 bins** to create the classification target  
   - Extract **weekday, hour, month** from timestamps  
   - Serialize examples to **TFRecord** (`dataset.tfrecords`)

2. **Preprocessing & Feature Engineering**
   - **Custom ImputerLayer** replaces NaNs with **feature-wise minima**  
   - **Normalization** on imputed ticker features  
   - **Embeddings** for temporal variables:  
     - weekday (7) → Embedding(2)  
     - hour (24) → Embedding(2)  
     - month (12) → Embedding(2)  
   - Concatenate compressed ticker signal with embeddings

3. **Modeling**
   - Inputs:  
     - `tickers`: ~188-dim float vector  
     - `weekday`, `hour`, `month`: integer indices  
   - Compression: `Dense(64, relu)` on normalized tickers  
   - Fusion: concat(ticker features, temporal embeddings)  
   - Head: 2–4 `Dense` layers (tuned), **output** `Dense(22, softmax)`  
   - **Loss:** categorical cross-entropy with **label smoothing = 0.1**  
   - **Optimizer:** Adam (tuned LR)

4. **Hyperparameter Tuning**
   - **Keras Tuner – RandomSearch** over activation, layers, units, learning rate, batch size  
   - **EarlyStopping** (val loss) to avoid overfitting

5. **Evaluation**
   - Track train/val curves; select best model  
   - Report **test accuracy**, discuss calibration and class imbalance

## Tools & Libraries

- **Python** (TensorFlow/Keras)  
- **Keras Tuner**, **NumPy**, **Pandas**  
- **TFRecord** serialization, **Matplotlib** for plots

## Data

- `appml-assignment1-dataset-v2.pkl` → source features + timestamps  
- Target: fractional change of **next-hour high vs current close**, **digitized into 22 bins**  
- Temporal features derived from datetime: **weekday [0–6]**, **hour [0–23]**, **month [1–12]**

## Key Insights

This workflow surfaces that:
- The label distribution is **highly skewed** (most samples concentrated in higher bins), so **label smoothing** stabilizes training better than naive class weighting.  
- A small **Dense(64)** compression on tickers + **temporal embeddings** improves fusion and downstream accuracy.  
- With tuning and regularization, the model reaches **~25% test accuracy** on a 22-class task and shows **no overfitting** in curves.

*(Exact curves and sweeps are in the report/notebook code paths.)*

## Conclusion & Business Impact

A modular TensorFlow pipeline with **TFRecords, custom imputation, and temporal embeddings** provides a reproducible base for **short-horizon movement classification**. The probabilistic outputs (22 bins) can be post-processed into **directional signals**, **risk bands**, or **cost-aware triggers**.

## Files in this Repository

- `createSavedDataset.py` — Build `dataset.tfrecords` from the pickle (feature extraction, digitization, TFExample writing)  
- `customImputeLayerDefinition.py` — Keras **ImputerLayer** that learns feature-wise minima and replaces NaNs at call-time  
- `buildandtrainmodel.py` — Data pipeline, **Keras Tuner RandomSearch**, model build, training, evaluation, and export  
- `appml-assignment1-dataset-v2.pkl` — Source dataset (pickle)  
- `dataset.tfrecords` — Serialized training dataset (TFRecord)  
- `FX Price Change Classification.pdf` — Project report (setup, experiments, results)

## How to Run

1) **Create TFRecord from the pickle**
```bash
python createSavedDataset.py
# outputs: dataset.tfrecords
```

2) **Train & tune**
  python buildandtrainmodel.py
  # runs RandomSearch, early stopping, prints final test accuracy and saves model

3) **Environment**
   pip install tensorflow keras-tuner pandas numpy matplotlib

## Author

**Aakash Maskara**  
*M.S. Robotics & Autonomy, Drexel University*  
Robotics | Reinforcement Learning | Autonomous Systems

[LinkedIn](https://linkedin.com/in/aakashmaskara) • [GitHub](https://github.com/aakashmaskara)
