****abstract/****

-We have data which classified if patients have heart disease or not according to features in it. We will try to use this data to create a model which tries to predict if a patient has this disease or not.

***-Datasets and Inputs:***

The data used for training and testing is the Heart Disease UCI downloaded from Kaggle (<https://www.kaggle.com/ronitf/heart-disease-uci>). This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "target" field refers to the presence of heart disease in the patient.

***-about the dataset:***

It's clean ( doesn't have any missing values "nan" ), it only has  numerical data (there are categorical variables in it but they already encoding) so no data needs to be processed (cleaned or encoded) before start to work with the model and it is easy to understand it  .  it consists of 303 rows and 14 columns .we will define each column in the following sentence.

1-age: The person's age in years.

2-sex: The person's sex (1 = male, 0 = female)

3-cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-angina pain, Value 4: asymptomatic)

4-trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)

5-chol: The person's cholesterol measurement in mg/dl

6-fbs: The person's fasting blood sugar (&gt; 120 mg/dl, 1 = true; 0 = false)

7-restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)

8-thalach: The person's maximum heart rate achieved

9-exang: Exercise induced angina (1 = yes; 0 = no)

10-ldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)

11-slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

12-ca: The number of major vessels (0-3)

13-thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

14-target: Heart disease (0 = no, 1 = yes)

***Data Visualization***

Data visualization is the discipline of trying to understand data by placing it in a visual context so that patterns, trends and correlations that might not otherwise be detected can be exposed.

***Bar chart:***

we could either use the bar chart to measure the frequency of each feature with additional comparing between having or not having diseases. 

![image](https://user-images.githubusercontent.com/72630461/131226289-620121e5-907c-47c4-9ae3-2f130ba71743.png)

***Histogram :***

It is the best way to represent the data and to know its distribution .Where Histograms are column-charts, which each column represents a range of the values, and the height of a column corresponds to how many values are in that range.

![image](https://user-images.githubusercontent.com/72630461/131226311-d367f0bf-f82f-4b73-ac88-2dd9460a23e1.png)

From the histogram we notice that:

1-age: has different discrete values from around 20 to around 80. To get the accurate domain we could use (df[‘age’].unique()) . and its mean is around 50

2-ca: has values(0,1,2,3,4) ,max of patients( more than 180) in the dataset have ca=0 , min of patients (less than 10) in the dataset have ca=4.

3-chol:has different discrete values from around 120 to around 560. And its mean is  around 250.

4-cp: has values(0,1,2,3) ,max of patients( more than 140) in the dataset have cp=0 , min of patients (25) in the dataset have cp=3.

5-exang:has values (0,1) , around 200 of patients have exang=0, and around 100 of patients have exang=1.

6-fbs:has values (0,1) , more 250 of patients have fbs=0, and less than 50 of patients have fbs=1.

7-oldpeak:has different discrete values from 0 to around 6.and its mean is around 1

8-rectangle:has values(0,1,2),max of patients( more than 150) in the dataset have rectceg=1, min of patients (less than 5) in the dataset have rectceg=2.

9-sex:has values(0,1), more 200 of patients have sex=1, and less than 100 of patients have sex=0.

10-slope:has values(0,1,2),max of patients( more than 125) in the dataset have slope=2, min of patients (less than 25) in the dataset have slope=0.

11-target:has values(0,1), more 160 of patients have target=1, and less than 140 of patients have target=0.

12-thal:has values(0,1,2,3) ,max of patients( more than 150) in the dataset have thal=2 , min of patients (5) in the dataset have thal=0.

13-thalach:has different discrete values from around 700 to around 200. and its mean is around 150.

14-trestbps:has different discrete values from around 70 to around 200. and its mean is around 90.

***scatter plot:***

-shows the relationship between two variables as dots in two dimensions, one axis for each attribute .we could get the accurate value of correlation using corr() method. 

-Now we will show the correlation between each feature and the target.

-We will not measure the correlation for the categorical variables(ca,cp,sex,exang,fbs,restecg,slope,thal).

![image](https://user-images.githubusercontent.com/72630461/131226329-2fafdadb-6f0e-4b6b-a08b-ce74ad54cfd0.png)

![image](https://user-images.githubusercontent.com/72630461/131226344-b46d4dca-3f59-4b28-b192-46ab44a979bb.png)

![image](https://user-images.githubusercontent.com/72630461/131226356-8fb7f4de-030c-4602-8ac3-b6b6977d21c9.png)

![image](https://user-images.githubusercontent.com/72630461/131226369-c1f02e42-4d21-4f89-ad73-b4825e21f82d.png)

![image](https://user-images.githubusercontent.com/72630461/131226376-77aba270-c3b2-43ff-83f8-3b4b57943576.png)

-Scatter plots that consist of all data points forming either a vertical or horizontal line indicate that the linear correlation is undefined. As we saw the following scatter is vertical so it is undefined. We will use corr()method 

1-age: there is no correlation.

2-trestbps:there is no correlation.

3-chol:there is no correlation.

4-thalach:there is weak negative.

5-oldpeak:there is a weak negative.

As we saw , there is no correlation with target 

-then we draw the scatter of different features with each other .

![image](https://user-images.githubusercontent.com/72630461/131226211-a73d6a86-9f3f-42f1-a971-981da8e6b471.png)

![image](https://user-images.githubusercontent.com/72630461/131226225-17a6f28f-690d-4e39-b89e-e0648c370d0a.png)

![image](https://user-images.githubusercontent.com/72630461/131226234-72a46c26-d884-4edc-b606-600fbd8b6713.png)

we notice that the correlation with different features is week or even no correlation . so we will use all features in the dataset.

***Line chart:***

I use a line chart in knns classifier to measure which neighbour has the maximum accuracy.

![image](https://user-images.githubusercontent.com/72630461/131226246-bb9b6bf8-ee54-464c-b597-cb28e7033b47.png)


***Machine learning model, a technical discussion.***

we will use many techniques :

***1-K-Nearest Neighbour (KNN) Model Classification***

***Description :***

KNN is an easy and supervised studying set of rules and it could keep all of the whole dataset consequently there may be no studying required.

KNN is used in statistical estimation and this classification is solely dependent on votes of the neighbours.

***Advantages :*** No assumptions about data — useful, for example, for nonlinear data

***Applications :*** Using KNN Algorithm Researchers are using data mining techniques in the medical diagnosis of several diseases such as diabetes ,stroke , cancer , and heart disease respectively.

First, when we use neighbours=2 the accuracy =83.61%,but after we use different values of neighbours we get accuracy =90.16% at neighbour=8.

***2-Decision Tree Model***

***Description :***

Decision tree is used to resolve the complex troubles. It resembles a tree-like shape.

The essential additives of the choice tree are selection node, nodes, and root.

Algorithms particularly used by selection tree are ID3, CART, CY3, C5.Zero and J48.These algorithms used by selection tree are used to analyse the dataset

***Strengths of Decision Tree*** :Decision Tree Deals With :

Categorical attributes with many distinct values

Variables with nonlinear effect on outcome

By Implementing Decision Tree We got Accuracy Rate : 75.41%

***3-Support Vector Machine (SVM) Algorithm:***

***Description :***

SVM is a supervised machine learning algorithm which can be used for classification or regression problems. It uses a technique called the kernel trick to transform your data and then based on these transformations it finds an optimal boundary between the possible outputs. Simply put, it does some extremely complex data transformations, then figures out how to separate your data based on the labels or outputs you've defined.

***Advantages :*** you can capture much more complex relationships between your data points without having to perform difficult transformations on your own.

***Disadvantages :*** the training time is much longer as it's much more computationally intensive.

Test Accuracy of SVM Algorithm: 83.61%

***Conclusion***

We just performed a survey associated with prediction of coronary heart illnesses; the use of records mining strategies and evaluation had been achieved in step with the accuracy and the records mining techniques are KNN, Decision Tree, SVM respectively.

By Taking The Predicted Accuracies into Considerations ,We Concluded that

KNNs model Which shows 90.16% Accuracy rate respectively.

Decision Tree model Shown 75.41% Accuracy rate respectively.

svm  model Which Shown 83.16% Accuracy rate respectively.




























