# AgroML-Hub

AgroML-Hub is an open-source platform for comparative machine learning analysis on key agricultural datasets. The project features multiple datasets—mushroom type prediction, forest cover prediction, and seed classification—and benchmarks several machine learning models (Random Forest, XGBoost, TensorFlow, and TensorFlow Lite) to provide actionable insights and performance comparisons. The platform is designed for extensibility, with plans to integrate a robust backend (Spring Boot, FastAPI), modern frontend (ReactJS), and MongoDB for data management.

## Features

- **Three Core Datasets**:  
  - **Mushroom Type Prediction**  
  - **Forest Cover Prediction**  
  - **Seed Type Prediction**
- **Model Implementations**:  
  - Random Forest  
  - XGBoost  
  - TensorFlow  
  - TensorFlow Lite
- **Comparative Analysis**:  
  - Evaluate and compare classical ML and deep learning approaches on real agricultural problems.
- **Planned Full-Stack Integration**:  
  - Backend: Spring Boot (Java) & FastAPI (Python)  
  - Frontend: ReactJS  
  - Database: MongoDB

## Project Structure

```
models/           # Trained model files and scripts
  notebooks/        # Jupyter notebooks for EDA, training, and analysis
  datasets/         # Contains mushroom, forest, seed datasets
  server/           # Server Code, consist of Fastapi code for routers
backend/          # Spring Boot 
frontend/         # (Planned) ReactJS frontend code
README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/PUSHPAK-JAISWAL/AgroML-Hub.git
cd AgroML-Hub
```

### 2. Setup Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Explore Datasets & Notebooks

- See the `notebooks/` directory for exploratory data analysis and training scripts for each dataset.
- Modify and experiment with different models and hyperparameters.

### 4. Model Training and Evaluation

- Scripts in the `models/` directory demonstrate training and comparing Random Forest, XGBoost, TensorFlow, and TensorFlow Lite models on all three datasets.
- Example usage:

```bash
python models/train_random_forest.py --dataset datasets/mushroom.csv
python models/train_xgboost.py --dataset datasets/forest.csv
python models/train_tensorflow.py --dataset datasets/seeds.csv
```

### 5. (Planned) Web Application

- The repository will soon include a backend (`backend/`) powered by Spring Boot and FastAPI and a frontend (`frontend/`) built with ReactJS.
- Data will be stored and managed via MongoDB.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under [The Unlicense](LICENSE).

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/)
- [TensorFlow](https://www.tensorflow.org/)
- Agricultural datasets collected from open sources

---

> For questions or collaboration, please contact [PUSHPAK-JAISWAL](https://github.com/PUSHPAK-JAISWAL).
