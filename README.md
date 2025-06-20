# Air Quality Prediction

Predicting Air Quality Index (AQI) using advanced machine learning techniques.

---

## 🚀 Project Overview
This project leverages real-world air quality data to build and evaluate regression models for predicting AQI. The workflow includes data cleaning, visualization, feature engineering, model training (Linear Regression & Random Forest), hyperparameter tuning, and model evaluation.

## ✨ Features
- Data cleaning & preprocessing
- Exploratory data analysis (EDA) with visualizations
- Regression modeling: Linear Regression & Random Forest
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Model evaluation: MSE, RMSE, MAE, R²
- Model saving & loading (joblib)

## 📁 Project Structure
```
Air-Quality-Prediction/
├── air-quality-prediction.ipynb  # Main Jupyter notebook
├── requirements.txt              # Project dependencies
├── random_forest_model.pkl       # Saved Random Forest model
├── README.md                     # Project documentation
└── .gitignore
```

## 🛠️ Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Air-Quality-Prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage
1. **Download the dataset:**
   - The dataset is fetched from KaggleHub in the notebook. No manual download required.
2. **Run the notebook:**
   ```bash
   jupyter notebook air-quality-prediction.ipynb
   ```
3. **Follow the notebook cells:**
   - Data exploration, cleaning, modeling, and evaluation are all step-by-step in the notebook.

## 🧠 Model Details
- **Linear Regression:** Baseline model for AQI prediction.
- **Random Forest Regressor:** Advanced model with hyperparameter tuning for improved accuracy.
- **Evaluation Metrics:**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score
- **Model Saving:** Trained models are saved as `.pkl` files for reuse.

## 📈 Results
- The Random Forest model outperformed Linear Regression in all evaluation metrics.
- For detailed results and visualizations, see the notebook's conclusion section.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repo and submit a pull request.

## 👤 Author
Developed by a data science enthusiast and passionate about AI and ML Engineer.