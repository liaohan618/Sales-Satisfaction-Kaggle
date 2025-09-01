# Analysis of Sales and Satisfaction Scores for 2024

## Table of Contents

1. [Abstract](#abstract)
2. [Acknowledgments](#acknowledgments)
   - [Team Acknowledgment](#team-acknowledgment)
   - [Data Acknowledgment](#data-acknowledgment)
3. [Background](#background)
4. [Data Description](#data-description)
   - [Variables](#variables)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Modeling](#modeling)
7. [Conclusion](#conclusion)
   - [Insights](#insights)
   - [Limitations](#limitations)
   - [Future Plans](#future-plans)
8. [Dependencies](#dependencies)
9. [Project Folder Structure](#project-folder-structure)
10. [References](#references)
11. [Contact Information](#contact-information)

## Abstract

This project aims to investigate what factors impact sales and satisfaction, more specifically whether introducing an intervention leads to an increase in either of these variables. The dataset used for this part of the analysis contains missing data, by which we introduce imputation as a way to work with the missing data. We also deploy data science techiques including hypothesis testing and machine learning (both supervised regression and classification) for investigating factors of sales and satisfaction. We provide the following key takeaways: (1) The **intervention increased sales**, but **NOT customer satisfaction scores**. (2) **Imputation can replace numerical distributions somewhat safely**, but can **poorly imbalance categorical classes**. (3) Based on regression model insights, we determine that **Sales & Customer Satisfaction Scores Prior to Intervention lead to increased Sales & Customer Satisfaction Scores After Intervention**. We also see that those in the **High Value Customer Segment** have **increased Sales & Customer Satisfaction Scores**. (4) Based on the classification model insights, we **cannot** confidently assert that the variables provided in this dataset are predictive of whether a purchase was made or not. This component of the project illustrates the approaches one can take to perform analysis even with missing data.

## Acknowledgments

### Team Acknowledgment

I would like to extend my gratitude towards the following team member for her contributions:

- Liaohan Wang

### Data Acknowledgment

The [data](https://www.kaggle.com/datasets/matinmahmoudi/sales-and-satisfaction) used in this project was provided by MATIN MAHMOUDI.

- Columns including **Group**, **Customer Segment**, **Sales (Before & After)**, **Customer Satisfaction Scores (Before & After)**, and **Purchase Made (Y/N)**. 

## Background

This part of the README is a continuation (or an off-shoot) of the main part of the project. The context is largely the same, where synthetic data is provided, giving demographic and numerical data regarding sales and customer satisfaction scores with and without introducing an intervention. The primary difference of this project focuses on what happens when there is missing data; there is the potential of bias being introduced as less data is available for analysis. There are a few approaches that one can take to deal with this

## Methodology

Our approach consists of the following steps:
1. We will conduct an initial EDA of the data to scope out trends and patterns within the data.
2. We will apply approaches (mainly imputation) to deal with missing data.
3. We will apply machine learning models to assess which factors impact sales and customer satisfaction scores, as well as if a purchase was made.
4. We will conduct hypothesis testing for assessing the intervention's impact on sales and customer satisfaction scores.

## Data Description

One dataset contains missing values (NaNs) and the other does not. These datasets contain information on sales and customer satisfaction before and after an intervention, as well as purchase data for control and treatment groups. The dataset is synthetic and was created for use in statistical analysis. This is an original dataset. The following version of the dataset is used:

- `Sales_without_NaNs.csv`

![Dataset](EDA/Dataset.png)

**Figure 1: Dataset**

### Variables

Below is a summary of the key variables in the dataset:

## Exploratory Data Analysis

An EDA investigation was done for the following predictor variables: **Age**, **Education**, **Gender**, **Race**, **States**, **Weeks**. By looking into how the data is distributed for these variables, we can gain greater insight into patterns and trends that correspond with increases in anxiety indicators.

Our main response variable will be the **Value** variable, which is the score value indicative of anxiety disorder.

Note: In the Jupyter Notebook containing analysis on [Preliminary Analysis](code/(CHRIS)%20Anxiety%20Prelim%20Analysis%20+%20Gender.ipynb), there is an explanation about hypothesis testing (both 1-sided and 2-sided t-tests). For comparing two groups in isolation, this form of statistical testing will enable us to determine significant differences between the mean value of these groups.

### Age

![Fig 2](EDA/Age/box.png)

**Figure 2: Boxplot of Score Value Distributions by Age.** The IQR (Interquartile Range) for each box per age group differs in length. Younger age groups, especially 18-29, seem to have higher levels of anxiety compared to the rest of the cohort, and their data is skewed. A noticeable gap between age groups 50-59 and 60-69 can be seen. Factors such as generational trauma or a competitive job market possibly cause this disparity between younger and older generations.

![Fig 25](EDA/Week/stats.png)

**Figure 25: Statistics of Score Value by Week.** The mean anxiety score value across these weeks were relatively comparable.

## Modeling

Supervised machine learning models were used to analyze the relationships between variables. The modeling section will be broken down into two sections: 

**(1) A Simple Linear Regression Model for each variable**

**(2) A Multivariate Linear Regression Model for all variables** 

The Jupyter Notebook for [multivariate linear regression](code/(CHRIS)%20MLR%20model%20based%20on%20Anxiety.ipynb) contains explanations of the statistical theory behind the multivariate linear regression model.

### Simple Linear Regression Models

#### Age

![Fig 26](Model%20Building/SLR/Age/model_summary.png)

**Figure 26: Model Summary for Simple Linear Regression Model for Age.** The model accuracy was 97.6% (or 97.2% adjusted), with the probability of the F-statistic being low enough to suggest evidence for differences in mean anxiety score among ages. Although this model accuracy is the highest for all of the simple linear regression models for all predictor variables, the magnitude of the accuracy raises concern for possibly overfitting the data, suggesting that hyperparameter tuning or optimization would be needed before firmly asserting the idea of mean difference among ages as suggested by the model.

## Conclusion

### Insights

The following factors impact Sales and Customer Satisfaction Scores:

### Limitations

Due to a lack of temporal data, we are unable to account for seasonal impacts on sales and customer satisfaction scores. With our current predictors, we cannot assert which factors impact purchases made. It could also help to get information on what products are being sold as well as what intervention is being introduced to contextualize the experiment better.

### Future Plans

We hope to assess if temporal data (i.e. seasonal, monthly, etc.) has an impact on sales and customer satisfaction scores, which would involve a time-series analysis. We also hope to investigate into further factors that could be impacting purchases made, provided that our current predictors fail to predict this specific response variable.

## Dependencies

- pandas
- matplotlib
- scipy
- statsmodels
- seaborn
- numpy
- sklearn

## Project Folder Structure

```plaintext
Sales-Satisfaction-Kaggle/
|
├── models/       # Folder containing pictures of model
|
├── plots/        # Folder containing EDA plots
|
├── results/      # Folder containing model results
|
├── tables/       # Folder containing statistical results
|
├── LICENSE       # License information
└── README.md     # Project documentation
```

## References

1. Kaggle - Sales and Satisfaction (https://www.kaggle.com/datasets/matinmahmoudi/sales-and-satisfaction)

## Contact Information

For any questions, please contact:

- Name: Christopher Fu
- Email: christopherfuwas@gmail.com
