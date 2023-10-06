# CS4641 Project Proposal

## Team Members
- Azhan Khan
- Eric Vela
- Felipe Mairhofer
- Hardik Goel
- Nicholas Arribasplata

## Introduction

Heart disease is a relentless reaper of lives. It ranks as the leading cause of death in the United States, claiming over 650,000 lives annually, and affects one in every five Americans [1]. In the time it takes to read this paragraph, someone in the US will have had another heart attack. We find this to be unconscionable. Health is wealth, and in a transformative era of rapid technological improvement and increasing wealth disparity, the least we can do is improve the methodologies for early detection of this cruel disease.

There is a personal element to this project as well; many of our team members have close family members who suffer from this disease, and because of its genetic component, it is of dire personal interest to ensure our heart health well into the future. Thus, our project will focus on the detection of heart disease and potential heart failure.

## Problem Definition

Early forays into machine learning solutions for heart disease have become popular as a result of hardware and deep learning advances [2]. But we believe they do not go far enough. Underlying factors such as diabetes, blood enzyme content, and blood pressure can serve as an effective indicator of the potential for heart disease in one’s future. We intend to explore all of these options throughout the course of this project, first focusing on the contents of this dataset: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/data

## Methods

We plan to use a number of supervised and unsupervised Machine Learning algorithms in order to achieve this, namely Decision Trees, Random Forest, Support Vector Machines, Linear Regression, and Naive Bayes. Python’s `scikit-learn` library offers robust implementations of these, thus giving us opportunities in exploring more of the machine learning/data science side of this problem.

## Discussion

As far as potential results, we will look at Accuracy and Precision as two key metrics to keep in mind, as the ramifications of a false positive or negative diagnosis can be potentially severe. We will conduct this project with the same respect as if this were a system created to aid doctors in the real world with their diagnostics, carefully observing best practices for data processing and outliers. Efficacy of such practices can often vary [3], but we will do our best in following the ethical care for our data.

## Project Timeline

Our expected contribution table is as follows:

| Name                | Assigned Portion       |
|---------------------|------------------------|
| Azhan Khan          | Decision Trees         |
| Hardik Goel         | Random Forest          |
| Eric Vela           | Support Vector Machines|
| Nicholas Arribasplata | Linear Regression     |
| Felipe Mairhofer    | Naive Bayes            |

Here is the Gantt Chart detailing the forecast for our development: [Upload and Insert Excel]

![Gantt Chart](https://i.imgur.com/yqUN22h.png)
[Download the Excel File](https://github.com/znatri/HeartFailurePredictor-CS4641/blob/eric-proposal-addGranttChart/assets/GanttChart.xlsx)


## Proposal Presentation

[Inset Video Link]

## References

1. [CDC - Heart Disease Facts](https://www.cdc.gov/heartdisease/facts.htm#:~:text=About%20695%2C000%20people%20in%20the,Coronary%20Artery%20Disease)
2. Javed Azmi et al. “A Systematic Review on Machine Learning Approaches for Cardiovascular Disease Prediction Using Medical Big Data.” Medical Engineering & Physics, Elsevier, 27 May 2022, [Link](www.sciencedirect.com/science/article/abs/pii/S1350453322000741#:~:text=These%20research%20articles%20have%20been,recognition%2C%20can%20predict%20cardiovascular%20problems.)
3. Zuo Z, Watson M, Budgen D, Hall R, Kennelly C, Al Moubayed N. Data Anonymization for Pervasive Health Care: Systematic Literature Mapping Study. JMIR Med Inform. 2021 Oct 15;9(10):e29871. doi: 10.2196/29871. PMID: 34652278; PMCID: PMC8556642.
