# Medical Insurance Cost Prediction

## Project Overview
This machine learning project predicts estimated medical insurance charges from personal and lifestyle information. It is a **regression** problem because the target is a continuous numerical value: `charges`.

## Problem Statement
Insurance costs can vary according to factors such as age, BMI, smoking status, number of children, gender, and residential region. The goal is to train a regression model that estimates insurance charges from these input features.

## Dataset
- **Name:** Medical Cost Personal Dataset
- **Format:** CSV
- **Source:** [Medical Cost Personal Dataset](https://github.com/stedy/Machine-Learning-with-R-datasets)
- **Target variable:** `charges`
- **Input features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Dataset size:** 1,338 rows and 7 columns

The Streamlit app reads the public CSV dataset when it starts, so a separate CSV upload is not required.

## Data Preprocessing
1. Loaded the CSV dataset with Pandas.
2. Checked the dataset structure and data types.
3. Converted categorical columns (`sex`, `smoker`, and `region`) into numeric dummy variables using `pandas.get_dummies()`.
4. Separated input features (`X`) from the target (`y = charges`).
5. Split the data into 80% training and 20% testing sets using `random_state=42`.

## Exploratory Data Analysis
Exploratory analysis examined relationships between insurance charges and age, BMI, smoking status, and feature correlations. The visualizations and detailed observations should be added from the project's Phase 4 notebook screenshots.

## Model
- **Algorithm:** Linear Regression
- **Task:** Regression
- **Training:** Fit on the training split
- **Prediction:** Predict charges for the held-out test set and new user inputs

## Evaluation Results
The Phase 6 evaluation produced the following results:

| Metric | Result |
|---|---:|
| MAE | 4181.19 |
| MSE | 33596915.85 |
| RMSE | 5796.28 |
| R² Score | 0.7836 |

The R² score indicates that the model explains approximately 78.36% of the variation in insurance charges on the test split. MAE and RMSE describe prediction error in the target's units; lower values are generally better.

## Prediction Application
The Streamlit application accepts:
- Age
- BMI
- Number of children
- Gender
- Smoking status
- Region

It applies the same dummy-encoding approach and aligns the input columns with the model's training columns before generating a numerical estimate.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Screenshots
Add your own screenshots in this section after deployment:
- `screenshots/application-interface.png`
- `screenshots/test-case-1.png`
- `screenshots/test-case-2.png`
- `screenshots/github-repository.png`
- `screenshots/deployed-app.png`

## Limitations
- Linear Regression may not capture every complex relationship in insurance costs.
- Predictions are estimates and should not be treated as official insurance quotes.
- Results depend on the dataset and its representation of the population.
- The app needs an internet connection to load the public dataset.

## Project Structure
```text
medical-insurance-cost-prediction/
├── app.py
├── requirements.txt
├── README.md
└── screenshots/
```

## Author
Add your name and internship/course details here.
