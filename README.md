# IJC437 Coursework Project  
## Predicting Billboard Hot-100 Chart Success Using Audio Features

This repository contains the coursework project for **IJC437 – Introduction to Data Science**.

The project investigates whether Spotify audio features can be used to predict whether a song reaches the **Top 50 of the Billboard Hot-100 chart**, using statistical analysis and machine learning methods implemented in **R**.

---

## Data

The analysis uses datasets provided in the coursework description, combining:
- Billboard Hot-100 chart data  
- Spotify audio features  

After preprocessing, **486 songs** with complete audio feature data were retained.  
Chart position was converted into a binary outcome: **Top 50** vs **Lower 50 (51–100)**.

---

## Methods

- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Train–test split (70% / 30%)  
- Logistic Regression  
- Random Forest  

Model performance was evaluated using accuracy and ROC/AUC.

---

## Key Findings

- Audio features alone provide **limited predictive power** for chart success.  
- Logistic Regression achieved the best performance with **61.38% test accuracy**.  
- Random Forest showed strong overfitting and poor generalisation.  
- Results suggest that non-audio factors (e.g. marketing and artist popularity) play a major role in Billboard chart success.

---

## Repository Contents

- `code.R` – Main R script for data preprocessing, analysis, and modelling  
- `README.md` – Project overview and documentation  

> The dataset is not uploaded due to file size and coursework requirements.  
> The code assumes the data file is placed in the same directory as `analysis.R`.

---
