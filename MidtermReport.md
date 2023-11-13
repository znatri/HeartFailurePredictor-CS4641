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

Here is the [Gantt Chart](https://github.com/znatri/HeartFailurePredictor-CS4641/blob/eric-proposal-addGranttChart/assets/GanttChart.xlsx) detailing the forecast for our development:

![Gantt Chart](https://i.imgur.com/yqUN22h.png)


## Proposal Presentation
Here is the [Presentation Video](https://www.youtube.com/watch?v=cpREG1BGDAM)
## References

1. [CDC - Heart Disease Facts](https://www.cdc.gov/heartdisease/facts.htm#:~:text=About%20695%2C000%20people%20in%20the,Coronary%20Artery%20Disease)
2. Javed Azmi et al. “A Systematic Review on Machine Learning Approaches for Cardiovascular Disease Prediction Using Medical Big Data.” Medical Engineering & Physics, Elsevier, 27 May 2022, [Link](www.sciencedirect.com/science/article/abs/pii/S1350453322000741#:~:text=These%20research%20articles%20have%20been,recognition%2C%20can%20predict%20cardiovascular%20problems.)
3. Zuo Z, Watson M, Budgen D, Hall R, Kennelly C, Al Moubayed N. Data Anonymization for Pervasive Health Care: Systematic Literature Mapping Study. JMIR Med Inform. 2021 Oct 15;9(10):e29871. doi: 10.2196/29871. PMID: 34652278; PMCID: PMC8556642.

As far as results, we were focusing on Accuracy and Precision as our two key metrics.

Based on the insights derived from the 'Feature Importance Bar Chart,' it is evident that the variable 'Time,' 
representing the follow-up period in days, plays a pivotal role in assessing the risk of heart failure. 
The feature importance analysis reveals that 'Time' holds a significant weight of approximately 30%, making it the 
most influential factor in the predictive model.

In comparison, other critical indicators such as 'serum creatinine' contribute around 15%, while factors like 
'ejection fraction' and 'age' each hold a 12% importance. Notably, all remaining features exhibit an importance value 
of less than or equal to 10%.

The interpretation of these findings suggest that patients with more frequent doctor visits, indicated by shorter 
durations in the 'Time' variable, are at a heightened risk of experiencing heart failure. This observation aligns 
with the intuitive notion that individuals with a history of recurrent hospitalizations and persistent critical heart 
conditions may face an elevated susceptibility to heart failure. The emphasis placed on the temporal aspect ('Time') 
underscores the significance of consistent medical monitoring and underscores the potential correlation between 
shorter follow-up periods and increased risk of heart-related complications.

Employing the random forest classifier allowed us to construct a confusion matrix, revealing insights into the 
model's performance. We observed a higher accuracy in predicting true negatives compared to true positives. The 
combined count of true negatives and true positives constituted a substantial portion of our sample, indicating a 
satisfactory predictive capability.

Despite this overall effectiveness, the model did exhibit instances of false negatives and false positives, which 
had a discernible impact on the accuracy metric. The occurrence of false negatives introduces a challenge, as these 
instances represent cases where the model failed to identify positive outcomes when they were, in fact, present. It 
is essential to address and mitigate false negatives to enhance the model's ability to improve the overall accuracy 
and reliability of the predictive results, but, more importantly, minimize the risk of overlooking the need for 
immediate medical attention.

The graph above illustrates the robust performance of the random forest classifier, achieving 100% training accuracy 
with a limited number of trees (12). Testing accuracy exhibited a positive trend with additional trees, signifying 
effective generalization to new data. The absence of overfitting, demonstrated by the convergence of training and testing 
accuracy, highlights the model's balanced learning. Despite minor variations in accuracy across runs, the model 
consistently maintained a commendable testing accuracy range of approximately 0.82 to 0.87. Overall, these results 
validate the classifier's reliability and effectiveness in heart failure prediction.
