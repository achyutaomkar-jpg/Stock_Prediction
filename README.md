# 📈 Stock Price Prediction using Deep Learning

## 🔍 Project Overview

This project focuses on building a deep learning–based stock price prediction system using historical stock market data. The application trains time-series models (LSTM-based) to learn price trends and forecast future stock prices. The final model is deployed as an interactive **Streamlit web application** for real-time predictions and visualization.

---

## 🌐 Live Streamlit App

👉 **Streamlit App Link:** https://stockprediction-e2hwwnrjmqv6fm5adrjr7o.streamlit.app/

---

## 📂 Repository Structure & Purpose

```
Stock_Prediction/
│
├── app.py                 # Main Streamlit application file (entry point)
├── requirements.txt       # Python dependencies for Streamlit Cloud deployment
├── runtime.txt            # (Optional) Python runtime specification (may be ignored by Streamlit Cloud)
├── Stock_Prediction.ipynb # Jupyter notebook for EDA, feature engineering, and model training
│
├── data/                  # Contains datasets and processed data files
│
├── models/                # Saved trained deep learning models (e.g., LSTM .h5/.keras files)
│
├── notebooks/             # Additional experiments or exploratory notebooks
│
├── report/                # Project report, findings, and documentation
│
├── .gitignore             # Files and folders ignored by Git
└── README.md              # Project documentation
```

---

## 📊 Dataset

* **Source:** *Add dataset source here (e.g., Yahoo Finance – TSLA historical data)*
* **Type:** Time-series stock price data (Open, High, Low, Close, Volume)
* **Usage:** Used for training and evaluating deep learning models for price prediction

---

## 🧰 Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * TensorFlow / Keras
  * NumPy
  * Pandas
  * Scikit-learn
  * Matplotlib
* **Web Framework:** Streamlit
* **Version Control:** Git & GitHub
* **Deployment:** Streamlit Community Cloud

---

## ▶️ How to Run the Notebook Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/achyutaomkar-jpg/Stock_Prediction.git
   cd Stock_Prediction
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   venv\\Scripts\\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter notebook:

   ```bash
   jupyter notebook Stock_Prediction.ipynb
   ```

5. Run all cells sequentially to reproduce preprocessing, training, and evaluation.

---

## 📈 Results Summary

* Deep learning models successfully captured temporal patterns in historical stock prices
* LSTM-based architecture provided strong performance for time-series forecasting
* Model predictions closely followed actual trends with acceptable error margins
* The trained model was integrated into a Streamlit app for interactive inference

---

## 🚀 Future Improvements

* Add multi-stock support
* Incorporate technical indicators (RSI, MACD, moving averages)
* Improve forecasting horizon
* Add model explainability and confidence intervals

---

## 👤 Author

**Akash Jalapati**

---

⭐ If you find this project useful, feel free to star the repository!
