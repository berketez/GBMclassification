# GRB (Gamma-Ray Burst) Classification Project

This project is a machine learning endeavor that aims to classify Gamma-Ray Bursts (GRBs) into short, long, and ultra-long categories using data from various GRB data sources. The project utilizes a range of machine learning algorithms to achieve this classification goal.

## Table of Contents

-   [Project Purpose](#project-purpose)
-   [Technologies](#technologies)
-   [Data Sources](#data-sources)
-   [Setup](#setup)
-   [Usage](#usage)
-   [Project Details](#project-details)
    -   [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    -   [Feature Engineering](#feature-engineering)
    -   [Model Training](#model-training)
    -   [Ensemble Model](#ensemble-model)
    -   [Results Visualization](#results-visualization)
    -   [Saving Classification Results](#saving-classification-results)
-   [Contributing](#contributing)
-   [License](#license)

## Project Purpose

The project's purpose is to classify GRBs into short, long, and ultra-long categories based on their duration, using data from different satellites (Swift BAT, Fermi GBM, CGRO BATSE). This aims to help better understand the properties and nature of GRBs in the field of astrophysics.

## Technologies

This project utilizes the following technologies:

-   **Python 3.8+**
-   **NumPy:** For numerical operations.
-   **Pandas:** For data manipulation and analysis.
-   **XGBoost:** Gradient boosting algorithm.
-   **LightGBM:** Fast gradient boosting algorithm.
-   **PyTorch TabNet:** Deep learning model.
-   **CatBoost:** Gradient boosting algorithm.
-   **Scikit-learn:** Machine learning tools.
-   **Uproot:** For reading ROOT files.
-   **Imbalanced-learn:** For handling imbalanced datasets (SMOTE).
-   **Matplotlib and Seaborn:** For data visualization.
-   **SciPy:** For statistical calculations.

## Data Sources

The project uses data from the following satellites:

-   **Swift BAT:** `filtered_grb_table_with_epeak.root` file.
-   **Fermi GBM:** `GRB_Data_with_Hardness_Ratio.root` file.
-   **CGRO BATSE:** `output_data.root` file.

## Setup

1.  **Install Python:** Install Python 3.8 or higher from [Python](https://www.python.org/downloads/).
2.  **Install required libraries:**

    ```bash
    pip install numpy pandas xgboost lightgbm pytorch-tabnet catboost scikit-learn uproot imbalanced-learn matplotlib seaborn scipy
    ```
3.  **Place the dataset files in the appropriate directory.** Place the dataset files in the `data` folder within the directory where the project is executed. The file names and formats should be compatible with the code.

## Usage

1.  Clone or download the project.
2.  Navigate to the project directory.
3.  Run the `main.py` file:

    ```bash
    python main.py
    ```

The project will load the data, perform preprocessing, train models, and visualize the results. Additionally, it will save the classification results for each satellite into separate CSV files.

## Project Details

### Data Loading and Preprocessing

-   The `load_data_from_directories` function loads `.root` extension files from the specified directory.
-   Each satellite's data is processed separately, and T90 (GRB duration) distributions are visualized.
-   Numerical features in the datasets are normalized.
-   Missing values are filled using the median value per satellite.
-   Duplicate columns are removed.

### Feature Engineering

-   The `prepare_features` function cleans duplicate columns from the dataset.
-   The `transform_to_three_classes` function classifies GRBs into short, long, and ultra-long categories based on their T90 durations.
-   The `preprocess_data` function cleans numerical columns in the dataset and fills `NaN` values.

### Model Training

The project trains the following machine learning models:

-   **XGBoost:** Trained using `xgb.XGBClassifier` for multi-class classification.
-   **LightGBM:** Trained using `lgb.train` for multi-class classification.
-   **TabNet:** Trained using `TabNetClassifier`.
-   **CatBoost:** Trained using `CatBoostClassifier`.

SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the data.

### Ensemble Model

-   An ensemble model is created using the `get_ensemble_predictions` function by combining model predictions and averaging the probabilities for each class.
-   For multi-classification results, the outputs of a model ensemble are combined using a majority vote method using predicted class values.
-   Various visualizations are used to assess the classification performance of the ensemble model.

### Results Visualization

-   The class distributions, accuracies, confusion matrices, and classification reports of model predictions are visualized.

### Saving Classification Results

-   The `save_classification_to_csv` function saves the classification results for each satellite into separate CSV files.

## Contributing

If you would like to contribute to the development of the project, please follow these steps:

1.  Fork the project.
2.  Create a new branch (`git checkout -b feature/new-feature`).
3.  Make your changes.
4.  Commit your changes (`git commit -am 'Added a new feature'`).
5.  Push your branch (`git push origin feature/new-feature`).
6.  Create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

