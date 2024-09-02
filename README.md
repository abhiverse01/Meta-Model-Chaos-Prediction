# Meta-Model Approach for Chaos Prediction Using the Titanic Dataset.

## Project Overview

This project explores the use of a meta-model approach to predict chaotic outcomes—such as survival in the Titanic disaster—by combining the strengths of multiple machine learning and deep learning models. Chaos prediction involves identifying situations where small changes in input can lead to unpredictable and highly variable outcomes. The goal is to create a robust, interpretable model to identify such chaotic behaviour.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Data Preparation](#data-preparation)
   - [Data Ingestion](#data-ingestion)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Data Preprocessing](#data-preprocessing)
4. [Model Building](#model-building)
   - [Machine Learning Models](#machine-learning-models)
   - [Deep Learning Models](#deep-learning-models)
5. [Meta-Model Approach](#meta-model-approach)
6. [Chaos Prediction Enhancements](#chaos-prediction-enhancements)
7. [Results](#results)
8. [Discussion and Future Work](#discussion-and-future-work)
9. [Requirements](#requirements)
10. [How to Run](#how-to-run)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction

Chaos theory is a field of study in mathematics that deals with systems that are highly sensitive to initial conditions—commonly referred to as the "butterfly effect." In this project, we apply the concept of chaos prediction to the Titanic dataset by developing a meta-model that aggregates predictions from multiple models to better capture the complexity and non-linearity of the data.

### Why the Titanic Dataset?

The Titanic dataset is a well-known dataset in the data science community, often used for binary classification problems. However, by viewing survival prediction as a chaotic process influenced by many interacting factors (age, gender, class, etc.), we can explore how small changes in input features might lead to vastly different outcomes.

### What is a Meta-Model?

A meta-model, also known as a stacked model, is a model that combines the outputs of multiple base models to make a final prediction. This approach leverages the strengths of various models, allowing for a more accurate and robust prediction, especially in systems exhibiting chaotic behavior.

## Project Structure

```
chaos-prediction-titanic/
│
├── data/                     # Raw and processed data files
│   ├── titanic.csv           # Original Titanic dataset
│   ├── titanic_preprocessed.csv
│   └── meta_features.csv
│
├── notebooks/                # Jupyter notebooks for EDA and prototyping
│   └── eda.ipynb
│
├── src/                      # Source code for data processing and model building
│   ├── data_ingestion.py     # Load and explore the dataset
│   ├── data_preprocessing.py # Preprocess the data (cleaning, feature engineering)
│   ├── model_training.py     # Train individual machine learning models
│   ├── meta_model_training.py# Aggregate predictions and train the meta-model
│   ├── model_evaluation.py   # Evaluate models and meta-model
│   └── visualizations.py     # Generate visualizations for interpretability
│
├── logs/                     # Logs for tracking progress and errors
│   └── training.log
│
├── models/                   # Saved trained models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── meta_model.pkl
│
├── requirements.txt          # Python packages required
└── README.md                 # Project documentation
```

## Data Preparation

### Data Ingestion

We begin by loading the Titanic dataset, which contains information on 891 passengers, including features like `Age`, `Sex`, `Pclass`, and whether they `Survived` the disaster. The dataset is first explored to identify any missing values, distributions of key features, and potential outliers.

### Exploratory Data Analysis (EDA)

EDA is conducted to uncover relationships between the features and the target variable (`Survived`). Visualizations such as histograms, box plots, and correlation matrices help in understanding these relationships. For instance, we might observe that younger passengers had a higher survival rate or that first-class passengers were more likely to survive.

### Data Preprocessing

Data preprocessing includes the following steps:
1. **Handling Missing Values:** Imputation strategies are used to fill missing values, such as using the median age to fill missing age entries.
2. **Encoding Categorical Variables:** Categorical variables like `Sex` and `Embarked` are converted into numerical values using one-hot encoding or label encoding.
3. **Feature Engineering:** New features are created to capture interactions between existing features. For example, combining `Pclass` and `Sex` might create a more informative feature.
4. **Scaling and Normalization:** Numerical features are scaled to ensure that models that rely on distance metrics (e.g., SVM) perform optimally.

## Model Building

### Machine Learning Models

We build several machine learning models, each with its strengths in capturing different patterns in the data:

1. **Logistic Regression:** Serves as our baseline model, providing a linear approach to classification.
2. **Decision Tree:** Captures non-linear interactions by recursively splitting the data based on feature values.
3. **Random Forest:** An ensemble method that reduces overfitting by averaging multiple decision trees.
4. **Support Vector Machine (SVM):** Finds the optimal hyperplane that separates the classes in high-dimensional space.
5. **XGBoost:** A gradient boosting method that focuses on difficult-to-predict cases, enhancing overall model performance.

### Deep Learning Models

Deep learning models are used to capture more complex patterns in the data:

1. **Artificial Neural Network (ANN):** A feedforward network that models non-linear relationships using multiple layers of neurons.
2. **Long Short-Term Memory (LSTM):** A type of recurrent neural network (RNN) designed to handle sequential data and long-term dependencies, potentially useful in chaotic systems.

## Meta-Model Approach

### Aggregating Predictions

Once the base models are trained, their predictions and probabilities are combined to create a meta-dataset. This dataset is then used to train a final meta-model (e.g., Random Forest or Logistic Regression) that predicts survival by leveraging the strengths of all individual models.

### Meta-Model Training

The meta-model is trained on the aggregated predictions from the base models. This model aims to capture the complex interactions and non-linear relationships that individual models might miss.

## Chaos Prediction Enhancements

### Advanced Feature Engineering

We enhance our feature set by introducing interaction terms, polynomial features, and embeddings for categorical variables, allowing the model to capture more complex relationships.

### Ensemble and Stacking

Stacking multiple models can significantly improve prediction accuracy. We explore stacking methods where base model predictions are combined and fed into a meta-learner, which makes the final prediction.

### Bayesian Neural Networks

Bayesian Neural Networks are used to quantify uncertainty in predictions. This is particularly important in chaotic systems, where uncertainty can be a critical factor in determining outcomes.

### Chaos Theory Metrics

Metrics like Lyapunov Exponents and Fractal Dimension are integrated into the model to better quantify and predict chaotic behavior in the system.

## Results

### Performance Metrics

We evaluate each model and the meta-model using accuracy, precision, recall, F1-score, and ROC-AUC. The meta-model typically outperforms individual models by effectively combining their strengths.
- <b> Sample Results. </b>

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.78     | 0.76      | 0.71   | 0.74     | 0.80    |
| Decision Tree        | 0.81     | 0.79      | 0.76   | 0.77     | 0.82    |
| Random Forest        | 0.84     | 0.82      | 0.79   | 0.81     | 0.86    |
| SVM                  | 0.82     | 0.80      | 0.77   | 0.78     | 0.83    |
| XGBoost              | 0.85     | 0.83      | 0.81   | 0.82     | 0.88    |
| ANN                  | 0.83     | 0.81      | 0.78   | 0.79     | 0.85    |
| LSTM                 | 0.82     | 0.80      | 0.77   | 0.78     | 0.84    |
| Meta-Model           | 0.87     | 0.85      | 0.83   | 0.84     | 0.90    |

## Discussion and Future Work

### Key Insights

- **Meta-Model Performance:** The meta-model consistently outperforms individual models, indicating the effectiveness of aggregating diverse model predictions.
- **Chaos Theory Integration:** Incorporating chaos theory metrics enhances the model's ability to predict unpredictable outcomes.

### Future Directions

- **Cross-Domain Application:** Apply the meta-model approach to other domains such as financial markets, weather forecasting, and social media dynamics where chaos

 is prevalent.
- **Advanced Architectures:** Explore the use of Transformer models for better handling of sequential and chaotic data.
- **Uncertainty Quantification:** Further develop methods to quantify and incorporate uncertainty in the prediction process, especially in chaotic systems.

## Requirements

To run this project, you'll need the following Python packages:

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- XGBoost
- SHAP (for interpretability)
- TensorFlow Probability (for Bayesian Neural Networks)
- nolds (for chaos theory metrics)

Install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

## How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/chaos-prediction-titanic.git
cd chaos-prediction-titanic
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the data preprocessing script:**

```bash
python src/data_preprocessing.py
```

4. **Train the base models:**

```bash
python src/model_training.py
```

5. **Train the meta-model:**

```bash
python src/meta_model_training.py
```

6. **Evaluate the model:**

```bash
python src/model_evaluation.py
```

7. **Generate visualizations and interpret results:**

```bash
python src/visualizations.py
```

## Contributing

We welcome contributions! Please fork this repository, create a new branch, and submit a pull request with your changes. Be sure to update the documentation as needed.

## License

This project is licensed under the MIT License. Please look at the [LICENSE](LICENSE) file for details.

