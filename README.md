# ExoLab Dashboard — NASA Space Apps Challenge 2025 🌌
**Project presented by team Astro404 as part of the NASA Space Apps Challenge 2025.**

## Description
ExoLab is an interactive dashboard built with **Streamlit** to explore, analyze, and classify exoplanets using Machine Learning, data science, and visualization. It allows loading a dataset (for example, from NASA KOI/TOI), obtaining statistics, exploring advanced graphics, training models, and making predictions about planetary disposition.

## Main Technologies and Dependencies
- **Python 100%**
- **Frameworks and libraries:**
  - Streamlit (web interface)
  - Flask (legacy support and migration)
  - Pandas, NumPy (data processing)
  - Scikit-learn, XGBoost (ML)
  - Plotly, Matplotlib, Seaborn (visualization)
  - Joblib (ML models)
  - Others: PyArrow, Altair, Watchdog, GitPython, etc.

_Check the requirements.txt file for the complete list of dependencies._

## Repository Structure
```
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── README.md             # This document
├── .streamlit/           # Streamlit configuration
├── data/                 # Datasets (current_data.csv, uploaded.csv)
├── models/               # Trained models (.joblib)
├── src/                  
│   ├── modules/          # Functional modules (ML, data, upload, download, etc.)
│   ├── utils.py          # Utilities
│   ├── ...               # Other associated scripts
└── assets/               # Images, icons (ExoLab.png, IconEL.png)
```

## Main Features 🚀
- **CSV Upload** (own or from NASA)
- **Exploration and statistical analysis:**
  - Key metrics by columns
  - Interactive visualizations (distribution, correlation, scatter, histograms)
- **ML model training:**
  - RandomForest and XGBoost
  - Metrics: Accuracy, Precision, Recall, F1, confusion matrix, feature importance
  - 5-fold cross-validation
- **Interactive prediction:**
  - Manual feature input for disposition prediction
  - Class probability/confidence
- **Specialized transit analysis:**
  - Duration vs. depth scatter plot
  - Analysis by planetary type and scatter plot

## Installation and Execution
```bash
# Clone the repository
git clone https://github.com/Ainy04/ExoLab-NasaSpacesApp2025.git
cd ExoLab-NasaSpacesApp2025

# Install dependencies
pip install -r requirements.txt

# Run the dashboard (Streamlit)
streamlit run app.py
```

Access through your browser, upload a CSV, and discover exoplanets like never before.

## Team
**Astro404**
- Coordination, design, data science, ML, and development: Ainy Contreras Mendoza, Daniel Tornero Solano, Juan Daniel Gonzales Reyes, Jorge Alberto Ruiz Rodriguez, Juan Carlos Cruz Hernandez and Josue Robledo Zuñiga.

## Useful Resources
- Example datasets (/data folder)
- Custom images and icons
- Ready-to-download models
- Modular structure to easily grow the project.

## License
Open source project presented for NASA Space Apps Challenge 2025.  
Developed by Astro404. All rights reserved.

