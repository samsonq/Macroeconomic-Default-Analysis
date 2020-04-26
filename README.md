# Reassessment of P2P Credit Risk Modeling with Macroeconomic Factors

Evaluating the influence of macroeconomic factors, along with personal factors, on peer-to-peer loan defaulting. Implement models with different features and perform statistical tests to determine the most important features contributing to default.

## Usage Instructions

* Install packages outlined in `requirements.txt`:

  ```
  pip3 install -r requirements.txt
  ```
  
## Overview
P2P Lending has enabled people to lend and receive loans without the mediation of a bank, making lending more accessible. But, a problem in this system is defaulting: the borrower’s failure to pay back. Before lending occurs, platforms like LendingClub evaluate the risk of investment for lenders and decide a suitable interest rate by assessing data of the borrower like their credit score, income, and education. However, there are inconsistencies in this risk modeling that may misclassify high-risk borrowers. External factors of the economy, like unemployment, GDP, and CPI can also impact the decision to default: it’s easier to pay back in prospering economies, while difficult in declining ones. So, people with good credit scores in declining areas like the Rust Belt may default. In this project, I propose methods of data integration, machine learning, and statistics to compare and evaluate the impact of macroeconomic factors to loan defaulting, versus primarily factors of the borrower.

## Project Description & Methods
A traditional risk model can be built with lending data from LendingClub containing information about borrowers and loan dates. To evaluate the impact of macroeconomic conditions, a second model will be built by merging data of unemployment, GDP, CPI, and other recession indicators from public sources like FRED to the lending data based on loan date. Defaulting may occur more often during layoffs and less during job booms, so I will categorize borrowers based on their job title and use the unemployment rate of that industry. I will also use economic factors based on the state of residence since Rust Belt states in recovery may have more defaulters than recovered states. Both models classify that a borrower pays or defaults. Different models will be implemented and compared, including logistic regression, XGBoost, and DNNs. The same borrowers will be used for both models in the training and testing set; the difference will be the economic features used in the second model. The accuracy and predictions made by both models on the test set will be compared to see if their differences are statistically significant. By comparing the results of both models, we can address the limitations of traditional risk models by exploring not only what, but how external economic factors influence defaulting. With a better understanding of defaulting behavior, P2P Lending can become more widely used as risk of investment is mitigated, providing people with more access to financial opportunities.

## Description of Contents

The project consists of these sections:
```
Table_Chart_Graph_Segmentation
├── README.md
├── .gitignore
├── data
│   ├── Economy
│   └── LendingClub
├── src
│   ├── __init__.py
│   ├── modeling
│   ├── preprocessing
│   └── visualization
├── notebooks
│   ├── EDA & Data Visualization.ipynb
│   ├── Missingness & Imputation.ipynb
│   ├── Preprocessing.ipynb
│   └── Modeling.ipynb
├── references
│   ├── Previous Research
├── requirements.txt
└── LICENSE
```

### `data`

* `Economy`: Contains public macroeconomic data gathered from federal sources
 
* `LendingClub`: Contains loan data from LendingClub including default status

### `src`

* `modeling`: Contains algorithms to train, evaluate, and optimize implemented models

* `preprocessing`: Utility python files to clean all data used

* `visualization`: Functions to quickly plot different charts in the data

### `notebooks`

* `EDA & Data Visualization`: Uni and multivaraite exploration of features in the data to discover associations/trends
  
* `Missingness & Imputation`: Analyze and handle all missing values in the datasets

* `Preprocessing`: Reformatting, scaling, encoding, data to prepare for modeling

* `Modeling`: Build and evaluate models to predict loan default

### `references`

* Data Dictionaries, references to external sources
