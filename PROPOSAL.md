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

## Results

As far as results, we were focusing on Accuracy and Precision as our two key metrics.

In our initial data pre-processing phase, we employed the `StandardScaler` from the scikit-learn library to standardize the features in our dataset. Standardization involves transforming the data such that it has a mean of 0 and a standard deviation of 1. This step is crucial, especially when working with the random forest algorithm, which is sensitive to the scale of input features.

To assess the performance of our model, we partitioned our dataset into two subsets: 85% for training and 15% for testing. This division ensures that the model is trained on a majority of the data, allowing it to learn patterns and relationships. The remaining 15% is reserved for testing, providing an independent dataset to evaluate how well the model generalizes to new, unseen data. This approach helps us gauge the model's performance on real-world scenarios beyond the training set, allowing for a more robust assessment of its predictive capabilities.

![Feature Importance](https://i.imgur.com/h7toqm5.png)

Based on the insights derived from the 'Feature Importance Bar Chart,' it is evident that the variable 'Time,' representing the follow-up period in days, plays a pivotal role in assessing the risk of heart failure. The feature importance analysis reveals that 'Time' holds a significant weight of approximately 30%, making it the most influential factor in the predictive model.

In comparison, other critical indicators such as 'serum creatinine' contribute around 15%, while factors like 'ejection fraction' and 'age' each hold a 12% importance. Notably, all remaining features exhibit an importance value of less than or equal to 10%.

The interpretation of these findings suggest that patients with more frequent doctor visits, indicated by shorter durations in the 'Time' variable, are at a heightened risk of experiencing heart failure. This observation aligns with the intuitive notion that individuals with a history of recurrent hospitalizations and persistent critical heart conditions may face an elevated susceptibility to heart failure. The emphasis placed on the temporal aspect ('Time') underscores the significance of consistent medical monitoring and underscores the potential correlation between shorter follow-up periods and increased risk of heart-related complications.

![Confusion Matrix](https://i.imgur.com/Wzgr8d6.png)

Employing the random forest classifier allowed us to construct a confusion matrix, revealing insights into the model's performance. We observed a higher accuracy in predicting true negatives compared to true positives. The combined count of true negatives and true positives constituted a substantial portion of our sample, indicating a satisfactory predictive capability.

Despite this overall effectiveness, the model did exhibit instances of false negatives and false positives, which had a discernible impact on the accuracy metric. The occurrence of false negatives introduces a challenge, as these instances represent cases where the model failed to identify positive outcomes when they were, in fact, present. It is essential to address and mitigate false negatives to enhance the model's ability to improve the overall accuracy and reliability of the predictive results, but, more importantly, minimize the risk of overlooking the need for immediate medical attention.

![Random Forest Classifier - Accuracy Trends](https://i.imgur.com/ptEsVlS.png)

The graph above illustrates the robust performance of the random forest classifier, achieving 100% training accuracy with a limited number of trees (12). Testing accuracy exhibited a positive trend with additional trees, signifying effective generalization to new data. The absence of overfitting, demonstrated by the convergence of training and testing accuracy, highlights the model's balanced learning. Despite minor variations in accuracy across runs, the model consistently maintained a commendable testing accuracy range of approximately 0.82 to 0.87. Overall, these results validate the classifier's reliability and effectiveness in heart failure prediction.


### Enhanced Preprocessing and Its Impact on Model Performance
To further refine our predictive model, we introduced additional preprocessing steps aimed at enhancing the robustness and accuracy of our predictions. These steps included outlier handling, class imbalance correction, and hyperparameter tuning, each contributing to a more sophisticated model training process.

### Outlier Handling
We first addressed outliers in the numerical features: 'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', and 'serum_sodium'. By applying the Interquartile Range (IQR) method, we capped extreme values, ensuring that our model was not skewed by anomalous data. This step was crucial in maintaining the integrity of the dataset, allowing the model to focus on representative trends and patterns.

### Class Imbalance Correction
Recognizing the potential bias introduced by imbalanced classes, we utilized the Synthetic Minority Over-sampling Technique (SMOTE). This approach balanced our target classes, enabling the model to learn from a more equitable representation of both outcomes. This adjustment was particularly important in ensuring that our model did not favor the majority class and could identify nuances in the minority class more effectively.

### Hyperparameter Tuning
We employed GridSearchCV for systematic hyperparameter tuning of our RandomForestClassifier. This method rigorously searched through combinations of parameters, allowing us to identify the optimal configuration for our model. The best parameters identified were: n_estimators, max_depth, and min_samples_split. This fine-tuning played a pivotal role in enhancing the model's predictive accuracy.

With these additional preprocessing steps, we observed a notable improvement in our model's performance. The accuracy increased from approximately 80% to around 87%, Peaking at about 90%. This jump in accuracy underscores the effectiveness of our preprocessing enhancements.

![Updated Random Forest Classifier - Accuracy Trends](https://i.imgur.com/nKZJHJN.png)

The updated accuracy trends graph further illustrates the impact of our preprocessing strategies. The graph depicts a consistent improvement in both training and testing accuracy, demonstrating the model's robustness and ability to generalize.

![Updated Confusion Matrix](https://i.imgur.com/SwnzMAQ.png)

The new confusion matrix provides a clearer picture of the model's predictive capabilities. Compared to our initial results, we observed a more balanced distribution of true positives and negatives, along with a reduction in false positives and negatives. This balance is indicative of the model's improved sensitivity and specificity, crucial for reliable heart failure prediction.

In summary, the integration of outlier handling, class imbalance correction, and hyperparameter tuning significantly boosted our model's accuracy and predictive reliability. These enhancements allowed for a more nuanced understanding of the factors influencing heart failure, leading to a more dependable tool for medical professionals in their diagnostic processes.

### Naive Bayes

Naive Bayes is another machine learning models we chose for predicting heart failure. As a supervised learning algorithm, Naive Bayes operates on the principles of Bayes Theorem to classify data based on the probability of certain events occurring given prior knowledge. Naive Bayes assumes independence among features, which is not very common in real world scenarios, especially heart failures. Despite this oversimplified assumption, Naive Bayes is particularly good at handling a large number of features relative to the dataset size and is computationally efficient, making it suitable for real-time predictions.

![Naive Bayes Confusion Matrix](https://i.imgur.com/znNMKMn.png) 

Running the Naive Bayes model allowed us to construct a confusion matrix, revealing insights into the model's performance. We observed a higher accuracy in predicting true negatives compared to true positives. Together, the count of true negatives and true positives formed a substantial portion of our sample, indicating a satisfactory predictive capability. However, compared to other models we used, such as Random forest classifier, this model didnt not perform as well.

Despite this overall effectiveness, the model did exhibit instances of false negatives and false positives. The occurrence of false negatives introduces a challenge, as these instances represent cases where the model failed to identify positive outcomes when they were, in fact, present. It is essential to address and mitigate false negatives to enhance the model's ability to improve the overall accuracy and reliability of the predictive results, but, more importantly, minimize the risk of overlooking the need for immediate medical attention.

![Naive Bayes Accuracy, Recall, F1-score, Precision](https://i.imgur.com/SxZ0hAT.png)

![True Values vs Predictions](https://i.imgur.com/iSicq16.png)

## Logistic Regression
![LR Confusion Matrix](https://i.imgur.com/BgpBGRm.png)

## Project Timeline

Our expected contribution table is as follows:

| Name                | Assigned Portion       |
|---------------------|------------------------|
| Azhan Khan          | Decision Trees         |
| Hardik Goel         | Random Forest          |
| Eric Vela           | Support Vector Machines|
| Nicholas Arribasplata | Logistic Regression  |
| Felipe Mairhofer    | Naive Bayes            |

Here is the [Gantt Chart](https://github.com/znatri/HeartFailurePredictor-CS4641/blob/eric-proposal-addGranttChart/assets/GanttChart.xlsx) detailing the forecast for our development:

![Gantt Chart]([https://i.imgur.com/yqUN22h](https://imgur.com/gallery/xM3hD6U).png)


## Proposal Presentation
Here is the [Presentation Video](https://www.youtube.com/watch?v=cpREG1BGDAM)
## References

1. [CDC - Heart Disease Facts](https://www.cdc.gov/heartdisease/facts.htm#:~:text=About%20695%2C000%20people%20in%20the,Coronary%20Artery%20Disease)
2. Javed Azmi et al. “A Systematic Review on Machine Learning Approaches for Cardiovascular Disease Prediction Using Medical Big Data.” Medical Engineering & Physics, Elsevier, 27 May 2022, [Link](www.sciencedirect.com/science/article/abs/pii/S1350453322000741#:~:text=These%20research%20articles%20have%20been,recognition%2C%20can%20predict%20cardiovascular%20problems.)
3. Zuo Z, Watson M, Budgen D, Hall R, Kennelly C, Al Moubayed N. Data Anonymization for Pervasive Health Care: Systematic Literature Mapping Study. JMIR Med Inform. 2021 Oct 15;9(10):e29871. doi: 10.2196/29871. PMID: 34652278; PMCID: PMC8556642.
