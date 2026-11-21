# 📱 Smartphone Price Prediction

### End-to-End Machine Learning Project | Regression | Random Forest | Streamlit

An end-to-end data science project that analyzes smartphone specifications and builds a machine learning model to predict smartphone prices.

The project covers the complete workflow from **data exploration and preprocessing to model development, evaluation, error analysis, and an interactive Streamlit application**.

---

## 🚀 Project Highlights

| | |
|---|---|
| 📊 Dataset | **980 smartphones, 25 features** |
| 🎯 Problem | **Price Prediction — Regression** |
| 🤖 Final Model | **Random Forest Regressor** |
| 📈 R² Score | **0.7292** |
| 💰 MAE | **₹7,494.53** |
| 📉 RMSE | **₹16,105.98** |
| 🧪 Test Set | **196 smartphones** |
| 🌐 Application | **Streamlit** |

---

## 🌐 Live Demo

**[Open the Smartphone Price Prediction App](https://smartphone-price-prediction-rbyfm6otg6aytwj4gxawdf.streamlit.app/)**

---

## 🎯 Business Problem

Smartphone prices vary significantly across brands and specifications.

The goal of this project is to investigate:

- Which smartphone specifications are associated with price?
- How do RAM, storage, processor speed, cameras, battery, and display features vary across price levels?
- Can smartphone price be predicted from technical specifications?
- How does prediction performance change across different price ranges?

The final model uses smartphone specifications to estimate the price of an unseen smartphone.

---

# 📊 Dataset

The dataset contains:

- **980 smartphones**
- **25 columns**
- **46 unique brands**
- Processor, memory, display, camera, battery, connectivity, operating-system, and rating information

The target variable is:

```text
price
```

### Major feature groups

| Category | Features |
|---|---|
| Product | `brand_name`, `model`, `price` |
| Processor | `processor_brand`, `num_cores`, `processor_speed` |
| Memory | `ram_capacity`, `internal_memory`, `extended_memory_available`, `extended_upto` |
| Display | `screen_size`, `refresh_rate`, `resolution` |
| Battery | `battery_capacity`, `fast_charging_available`, `fast_charging` |
| Camera | `num_rear_cameras`, `num_front_cameras`, `primary_camera_rear`, `primary_camera_front` |
| Connectivity | `has_5g`, `has_nfc`, `has_ir_blaster` |
| Other | `rating`, `os` |

---

# 🔎 Exploratory Data Analysis

Before building the model, the dataset was investigated to understand its structure, pricing distribution, missing values, feature relationships, and potential outliers.

## 💰 Price Distribution

Smartphone prices are strongly right-skewed.

| Statistic | Price |
|---|---:|
| Minimum | ₹3,499 |
| Median | ₹19,994.50 |
| Mean | ₹32,520.50 |
| 75th percentile | ₹35,491.50 |
| 90th percentile | ₹69,999 |
| 95th percentile | ₹99,990.45 |
| Maximum | ₹650,000 |

The large difference between the median and mean reflects the presence of a small number of very high-priced smartphones.

---

# 🏷️ Price Segmentation

For exploratory analysis, smartphones were divided into six price segments:

- Budget
- Entry-Mid
- Mid-Range
- Upper-Mid
- Premium
- Luxury

### Price-segment profile

| Segment | Smartphones | Median Price | Median RAM | Median Storage | Median Processor Speed | Median Refresh Rate |
|---|---:|---:|---:|---:|---:|---:|
| Budget | 141 | ₹7,999 | 3 GB | 32 GB | 2.00 GHz | 60 Hz |
| Entry-Mid | 365 | ₹14,999 | 6 GB | 128 GB | 2.20 GHz | 90 Hz |
| Mid-Range | 263 | ₹27,994 | 8 GB | 128 GB | 2.50 GHz | 120 Hz |
| Upper-Mid | 114 | ₹53,045 | 8 GB | 128 GB | 3.05 GHz | 120 Hz |
| Premium | 49 | ₹82,990 | 8 GB | 256 GB | 3.05 GHz | 120 Hz |
| Luxury | 48 | ₹129,990 | 8 GB | 256 GB | 3.00 GHz | 120 Hz |

### Key observation

Several specifications increase across lower price segments and then begin to plateau.

```text
RAM:
3 GB → 6 GB → 8 GB → plateau

Storage:
32 GB → 128 GB → 256 GB

Refresh rate:
60 Hz → 90 Hz → 120 Hz

Processor speed:
2.00 GHz → 2.20 GHz → 2.50 GHz → ~3.05 GHz
```

These patterns describe associations in the dataset and should not be interpreted as causal effects.

---

# 🧹 Missing-Value Analysis

The dataset contains missing values in several columns.

| Feature | Missing Values |
|---|---:|
| `extended_upto` | 480 |
| `fast_charging` | 211 |
| `rating` | 101 |
| `processor_speed` | 42 |
| `processor_brand` | 20 |
| `os` | 14 |
| `battery_capacity` | 11 |
| `num_cores` | 6 |
| `primary_camera_front` | 5 |
| `num_front_cameras` | 4 |

Missing values were handled inside the machine learning pipeline rather than manually filling the dataset beforehand.

This keeps preprocessing reproducible between training and prediction.

---

# 📈 Feature Relationships

Correlation analysis was performed for numerical variables.

### Correlation with price

| Feature | Correlation |
|---|---:|
| Internal Memory | **0.557** |
| Processor Speed | **0.474** |
| RAM Capacity | **0.386** |
| Rating | **0.284** |
| Fast Charging | **0.278** |
| Refresh Rate | **0.244** |
| Primary Front Camera | 0.163 |
| Number of Rear Cameras | 0.125 |
| Number of Front Cameras | 0.115 |
| Screen Size | 0.113 |
| Primary Rear Camera | 0.092 |
| Extended Storage | 0.092 |
| Number of Cores | -0.049 |
| Battery Capacity | -0.159 |

These are **associations within this dataset**, not causal effects.

---

# 🤖 Machine Learning Approach

## Problem Type

**Supervised Regression**

### Target variable

```text
price
```

The model predicts a continuous numerical value rather than a class.

---

# 🧩 Feature Selection

The final model uses **23 predictor features**.

### Numerical features

```text
rating
has_5g
has_nfc
has_ir_blaster
num_cores
processor_speed
battery_capacity
fast_charging_available
fast_charging
ram_capacity
internal_memory
screen_size
refresh_rate
num_rear_cameras
num_front_cameras
primary_camera_rear
primary_camera_front
extended_memory_available
extended_upto
```

### Categorical features

```text
brand_name
processor_brand
os
resolution
```

---

# 🚫 Data Leakage Prevention

Target-derived features were deliberately excluded from the final model.

The following exploratory features were **not used as model inputs**:

```text
price_segment
is_premium
premium_status
```

The `model` column was also excluded because it contains highly specific product identifiers and has **980 unique values for 980 smartphones**.

This prevents the model from relying on information derived from the target or memorizing individual product names.

---

# ⚙️ Preprocessing Pipeline

The final model uses a Scikit-learn `Pipeline` with a `ColumnTransformer`.

### Numerical preprocessing

```text
Median Imputation
```

### Categorical preprocessing

```text
Most-Frequent Imputation
```

### Categorical encoding

```text
One-Hot Encoding
```

Unknown categories during prediction are handled using:

```text
handle_unknown="ignore"
```

The preprocessing and model are stored together so the same transformation process is applied during both training and inference.

---

# 🌲 Model Development

A Random Forest Regressor was used as the final algorithm.

```python
RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
```

The dataset was split into:

```text
80% Training → 784 smartphones
20% Testing  → 196 smartphones
```

The test set was kept separate for final evaluation.

---

# 🧪 Model Experiments

The project included multiple modeling experiments.

### 1. Numerical-only Random Forest

| Metric | Result |
|---|---:|
| MAE | ₹8,524.55 |
| RMSE | ₹17,054.43 |
| R² | 0.6963 |

### 2. Hyperparameter tuning

Grid search with cross-validation was performed over:

- Number of trees
- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples per leaf

The best cross-validation configuration was:

```text
n_estimators = 500
max_depth = 20
min_samples_split = 2
min_samples_leaf = 2
```

This tuned configuration did not outperform the original numerical-only model on the held-out test set.

### 3. Final model

The final model incorporated:

```text
brand_name
processor_brand
os
resolution
```

along with the numerical features.

This produced the strongest held-out test performance among the evaluated approaches.

---

# 🏆 Final Model Performance

### Test-set results

| Metric | Score |
|---|---:|
| **MAE** | **₹7,494.53** |
| **RMSE** | **₹16,105.98** |
| **R²** | **0.7292** |

### What does this mean?

**MAE — ₹7,494.53**

On average, the absolute difference between the actual and predicted prices was approximately ₹7,495 on the held-out test set.

**RMSE — ₹16,105.98**

RMSE is higher than MAE because larger prediction errors receive greater weight.

**R² — 0.7292**

The model explains approximately 72.9% of the variance in smartphone prices in this held-out test set.

---

# 🔍 Feature Importance

Feature importance was investigated using both:

- Random Forest impurity-based importance
- Permutation importance

Important model features included:

- Internal memory
- Processor speed
- Primary front camera specification
- Battery capacity
- NFC
- Rating
- RAM capacity
- 5G support
- Screen size

Feature importance represents **model behavior and predictive usefulness**, not causation.

---

# 📉 Error Analysis

The model was evaluated across different price ranges.

| Price Range | Test Smartphones | MAE |
|---|---:|---:|
| Under ₹10K | 29 | ₹2,063.24 |
| ₹10K–₹20K | 70 | ₹1,911.50 |
| ₹20K–₹50K | 63 | ₹6,653.76 |
| ₹50K–₹1L | 24 | ₹21,718.59 |
| Above ₹1L | 10 | ₹33,485.60 |

### Important observation

Prediction errors are considerably smaller in the lower and mainstream price ranges and increase substantially for higher-priced smartphones.

This is an important limitation of the current model.

Possible factors that are **not fully represented by the available features** may contribute to this pattern, such as brand positioning, product exclusivity, design, or other market considerations.

These are possible explanations rather than causal conclusions established by the model.

---

# 🚨 Outlier Strategy

The dataset contains legitimate high-price smartphones, including observations reaching ₹650,000.

These observations were **retained rather than automatically removed or capped**.

> An extreme price is not automatically an erroneous observation.

Removing legitimate high-priced products could distort the price distribution the model is intended to learn.

The current model therefore trains on the original price distribution.

### Future experiment

A useful next experiment would be comparing the current raw-price model against a model trained using:

```python
log1p(price)
```

Both approaches should be evaluated on the same held-out test set before selecting one for production.

---

# 🌐 Streamlit Application

The project includes an interactive Streamlit application for price prediction.

Users can enter smartphone specifications such as:

- Brand
- Processor
- Operating system
- RAM
- Internal storage
- Processor speed
- Battery capacity
- Refresh rate
- 5G
- NFC
- Fast charging
- Cameras
- Display resolution
- Expandable storage

The application passes the submitted specifications through the saved preprocessing + Random Forest pipeline and returns an estimated smartphone price.

### Application design

The Streamlit interface is organized into:

- Basic Information
- Performance
- Memory & Storage
- Display
- Battery & Charging
- Camera
- Connectivity & Expansion

The application also displays:

- Predicted price
- Model performance
- Submitted specifications
- Dataset information
- Outlier-handling explanation

---

# 🔄 Dataset-Consistent Input Interface

Many categorical and discrete input options in the Streamlit application are generated directly from the dataset.

For example:

```python
brand_options = sorted(
    df["brand_name"].dropna().unique().tolist()
)
```

Similar dataset-driven options are used for processor brands, operating systems, resolutions, RAM, storage, refresh rates, camera counts, charging values, and other discrete inputs.

This reduces the risk of the application accepting arbitrary categories that were never represented in the training data.

---

# 🛠️ Technology Stack

### Programming
- Python

### Data Analysis
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- Random Forest
- Pipeline
- ColumnTransformer
- SimpleImputer
- OneHotEncoder
- Train/Test Split
- Cross-Validation
- Permutation Importance

### Application
- Streamlit

### Version Control
- Git
- GitHub

---

# 📁 Project Structure

```text
smartphone-price-prediction/
│
├── app.py
├── README.md
├── requirement.txt
├── smartphone_cleaned_v5.csv
├── smartphone_price_model.pkl
├── .gitignore
│
└── notebooks/
    └── smartphone_price_prediction.ipynb
```

> Update this structure before publishing if the final repository uses different filenames or folders.

---

# ▶️ Run the Project Locally

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/smartphone-price-prediction.git
```

```bash
cd smartphone-price-prediction
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Start the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

# 📌 Key Takeaways

This project demonstrates an end-to-end machine learning workflow:

```text
Raw Data
   ↓
Data Understanding
   ↓
EDA & Visualization
   ↓
Missing-Value Analysis
   ↓
Feature Analysis
   ↓
Price Segmentation
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ↓
Random Forest Regression
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Error Analysis
   ↓
Streamlit Application
```

The project goes beyond simply training a model by examining **data quality, leakage, model experiments, feature importance, price-range errors, and an interactive application**.

---

# ⚠️ Limitations

- The dataset contains 980 smartphones, which limits the amount of training data available.
- Smartphone prices are highly skewed.
- High-priced smartphones represent a relatively small portion of the dataset.
- Some features contain missing values.
- The available features do not capture every factor that influences smartphone pricing.
- Prediction error increases substantially in higher price ranges.
- The model should be interpreted as a data-driven estimate based on the provided dataset, not as a real-time market price.

---

# 🔮 Future Improvements

### Machine Learning
- Compare Random Forest with Gradient Boosting models.
- Experiment with XGBoost or other boosting algorithms.
- Test `log1p(price)` target transformation.
- Perform additional feature engineering.
- Compare multiple regression algorithms systematically.

### Explainability
- Add SHAP-based explanations.
- Display feature contributions for individual predictions.

### Application
- Add interactive EDA dashboards.
- Add actual-vs-predicted visualizations.
- Add model explanation for each prediction.
- Improve validation for unusual combinations of specifications.

### Deployment
- Deploy the Streamlit application publicly.
- Add a live demo link to the GitHub README.
- Add screenshots/GIFs of the application.

---

# 👨‍💻 Author

**Sarthak Dubey**

Data Science & Machine Learning Portfolio Project

- GitHub: [View my GitHub profile] (https://github.com/sarthakdubey0708-ship-it)
- LinkedIn: [Connect with me on LinkedIn] (https://www.linkedin.com/in/sarthak-dubey-865829393/)

---

## ⭐ Project Summary

**Smartphone Price Prediction** is an end-to-end machine learning project demonstrating how structured product data can be transformed into an interactive predictive application.

The project combines:

**EDA + Feature Analysis + Data Preprocessing + Regression + Model Evaluation + Error Analysis + Streamlit**

with a final Random Forest model achieving:

> **R²: 0.7292 | MAE: ₹7,494.53 | RMSE: ₹16,105.98**

on a held-out test set of 196 smartphones.
