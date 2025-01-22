# Deep Learning vs Tree-Based Machine Learning for Tabular Data Prediction

## Overview
This project aims to compare the performance of deep learning and tree-based machine learning models in predicting tabular data. Four models are used for each category:

- **Deep Learning (DL) Models**: NODE, SAINT, TabNet, TabTransformer
- **Tree-Based Machine Learning (ML) Models**: Random Forest, XGBoost, LightGBM, CatBoost

The comparison is based on three metrics:
1. **Accuracy**
2. **F1 Score**
3. **Running Time**

## Datasets
The following datasets were used for this project:
1. **Bank Churn**: Dataset that purposed to predict whether a customer is continued use the service or not, by a certain feature like Age, Income, etc.
2. **Keep It Dry**: Dataset that purposed to predict whether a product is suitable for sale or not, by a certain measurement result test.
3. **Horse Health**: Dataset that purposed to predict outcome of horse healt based on some health conditions.

## Models
### Deep Learning Models
1. **NODE (Neural Oblivious Decision Ensembles)**: A model designed for structured data by combining decision tree principles with neural networks.
2. **SAINT (Self-Attention and Intersample Attention Transformer)**: A transformer-based model optimized for tabular data using self-attention and intersample attention mechanisms.
3. **TabNet**: A deep learning model leveraging sequential attention to select features dynamically for tabular data.
4. **TabTransformer**: A transformer-based approach to learn embeddings for categorical features alongside standard tabular processing.

### Tree-Based Machine Learning Models
1. **Random Forest (RF)**: An ensemble method using multiple decision trees for robust predictions.
2. **XGBoost (XGB)**: An optimized gradient boosting framework.
3. **LightGBM (LGBM)**: A gradient boosting framework focused on speed and memory efficiency.
4. **CatBoost (CB)**: A gradient boosting framework designed to handle categorical features effectively.

## Results
The following tables summarize the comparison across accuracy, F1 score, and running time for each dataset:

### Bank Churn
| Model             | Accuracy | F1 Score | Running Time (s) |
|-------------------|--------------|--------------|------------------|
| NODE             | 0.839        | 0.837      | 890.75           |
| SAINT            | 0.861        | 0.846        | 1090.98           |
| TabNet           | 0.862        | 0.858        | 222.55           |
| TabTransformer   | 0.865        | 0.857        | 655.67           |
| Random Forest    | 0.858        | 0.850        | 16.23           |
| XGBoost          | 0.866        | 0.859        | 0.61           |
| LightGBM         | **0.868**    | **0.861**    | **0.51**           |
| CatBoost         | **0.868**        | 0.860        | 11.29          |

### Keep It Dry
| Model             | Accuracy | F1 Score | Running Time (s) |
|-------------------|--------------|--------------|------------------|
| NODE             | 0.699        | 0.681       | 245.54           |
| SAINT            | 0.786        | 0.695        | 157.70           |
| TabNet           | **0.787**        | 0.694        | 45.98           |
| TabTransformer   | 0.767        | 0.698        | 96.62           |
| Random Forest    | **0.787**        | 0.694        | 8.24           |
| XGBoost          | 0.772        | **0.700**        | 0.18           |
| LightGBM         | 0.786        | 0.694        | **0.15**           |
| CatBoost         | **0.787**        | 0.695        | 6.79           |

### Horse Health
| Model             | Accuracy | F1 Score | Running Time (s) |
|-------------------|--------------|--------------|------------------|
| NODE             | 0.500        | 0.500        | 2.65           |
| SAINT            | 0.641        | 0.641        | 2.19           |
| TabNet           | 0.336        | 0.336        | **0.09**           |
| TabTransformer   | 0.660        | 0.657        | 2.64           |
| Random Forest    | 0.692        | 0.692        | 0.18          |
| XGBoost          | 0.676        | 0.675      | 0.49           |
| LightGBM         | 0.696        | 0.697        | 0.25           |
| CatBoost         | **0.708**        | 0.708        | 2.24           |

## Contact
For any questions or feedback, please contact [fadhilahhilmi04@gmail.com].