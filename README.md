# Capstone 2: Detecting Heavy Drinking with Smartphone Accelerometer Data
For Springboard Data Science Career Track

Alcohol has been found to be the third leading preventable cause of death in the US, preceded only by tobacco (first leading preventable cause) and poor diet and physical inactivity (second leading preventable cause). As smartphones become increasingly commonplace in the US, and heavy drinking remains an ongoing problem, it is desirable to be able to detect and prevent incidences of heavy drinking using an easily accessible metric, such as a mobile phone’s accelerometer data. This would allow for a variety of applications, such as just-in-time adaptive interventions (JITAIs). This method sends prompts or cues in a strategically timed manner to the user (sent “just-in-time”) to alter user behavior and change habits.

By using a dataset developed by Killian et al. (2019), I generated new features from the raw data and created a supervised binary classification model that classifies whether a personal is considered legally impaired from drinking alcohol--this is defined as when a person’s blood alcohol content (BAC) is 8% or higher. The model is an XGBoost Classifier, and achieved an Accuracy of 90.9% and True Positive Rate of 93.1% (exceeding the authors’ Accuracy and TPR of 77.5% and 69.8%, respectively).

## Data and Preprocessing

The raw dataset by Killian et al. (2019) contains smartphone accelerometer data from 13 anonymized Ohio State University college students and corresponding Transdermal Alcohol Concentration (TAC) data using SCRAM ankle bracelets, during a bar crawl. The SCRAM bracelets provided 24/7 transdermal alcohol testing, and functions like a breathalyzer for the ankle. SCRAM bracelets show better TAC accuracy at higher alcohol concentrations, making the dataset unsuitable for determining all instances of drinking, but suitable for detecting heavy drinking. 

There is much higher granularity in the accelerometer data (sampled at 40 Hz) than in the TAC data (sampled every 30 minutes). Furthermore, the data is also very noisy, and needs to be smoothed. This resulted in a bifurcation of our data: one dataset which used a triangular-moving-average of window size 3 to smooth all data points (for both the accelerometer data and TAC data), and one dataset which used the scipy interpolation function to fill in the missing TAC data for each row in the accelerometer data.

## EDA and Featurization

The dataset had only 3 independent variables (x, y, and z accelerometer data) to predict our 1 dependent variable (TAC Level). Feeding these 3 columns of data into a machine learning model is essentially meaningless. Features have to be derived from the accelerometer data. I made the following features in 6-fold sets (x, y, and z axis, plus their smoothed counterparts x_tma, y_tma, and z_tma) using the pandas rolling() method over 10-second windows:
 * Mean
 * Standard Deviation
 * Variance
 * Median
 * Max (of raw and of absolute signal)
 * Min (of raw and of absolute signal)
 * Skew
 * Kurtosis
 * Zero Crossing Rate - the number of times the signal changes signs.
 * Gait stretch - the difference between max and min of one stride (of raw signal).
 * Number of steps - the total number of peaks.
 * Step Time - the average time between steps.
 * RMS - Root-mean-square of accelerations, i.e. the average power.
 * Average Resultant Acceleration - Average of the square roots of the sum of the values of each axis squared.
 * Jerk (the gradient of the accelerometer data) and the above features for Jerk.
 * Snap (the gradient of jerk) and the above features for Snap.
 * The standard deviation of all of the above features.

## Modeling and Results:

After featurization, I had 310 columns and 11.28 million rows. To reduce the size of my dataset, I sampled n=400 random rows from each participant/class combination (13 participants, 2 classes). This resulted in 2 dataframes with 10400 rows and 310 columns. I then dropped any columns containing 100 or more NaN values, and then dropped any rows containing NaN or np.inf values. This left me with 2 dataframes completely clean from any NaN or infinite values. My raw/interpolated dataset contained 9107 rows and 293 columns, and my TMA-smoothed dataset contained 9486 rows and 287 columns. I conducted a 70/30 train/test split and scaled the data.

For each data set (raw/interpolated, tma-smoothed), I tested 4 different Decision Tree Classifiers: 
 * Random Forest Classifier
 * AdaBoost Classifier
 * Gradient Boosting Classifier
 * XGBoost Classifier
XGBoost out-of-the-box performed best, followed by RF out-of-the-box. I took both the RF and XGBoost and conducted hyperparameter tuning with 5-fold cross validation. The final and best performing model was XGBoost, with Best Params = {'n_estimators': 1000, 'max_depth': 15, 'learning_rate': 0.1}

### Table of Results
|Model|Data (raw or tma)|Accuracy|F1-Score|Precision|Recall|Avg Precision|ROC AUC|Runtime|
|:-:|:-:|--:|--:|--:|--:|--:|--:|--:|
|RF (paper)|-|0.775|-|0.666|0.698|-|-|-|
|RF|raw|0.753|0.748|0.718|0.873|0.82|0.69|__2.526__|
|RF|tma|0.793|0.791|0.742|0.903|0.88|0.72|2.692|
|AdaBoost|raw|0.680|0.679|0.680|0.741|0.75|0.64|34.953|
|AdaBoost|tma|0.711|0.710|0.694|0.759|0.79|0.65|35.277|
|GradientBoost|raw|0.739|0.737|0.719|0.830|0.81|0.69|85.575|
|GradientBoost|tma|0.762|0.760|0.723|0.853|0.84|0.69|83.256|
|XGBoost|raw|0.864|0.863|0.836|0.922|0.94|0.81|4.655|
|__XGBoost__|tma|__0.894__|__0.893__|__0.875__|__0.920__|__0.96__|__0.84__|3.641|
|RF-optimized|tma|0.768|0.765|0.722|0.874|0.85|0.69|__0.528__|
|__XGBoost-optimized__|tma|__0.909__|__0.909__|__0.891__|__0.931__|__0.97__|__0.86__|32.856|

### Feature Importance:

For the top 20 features of the best-performing models, a significant portion of the top 20 features pertain to the standard deviation of another feature (14 of 20 for XGB-optimized, 16 of 20 for XGB). I similarly looked into the bottom 10 least important features of these models, none of them included the standard deviation of another feature. Computing the standard deviation of every feature seems to provide more predictive power to the model.

## Future Directions:

If this model were to be implemented into a just-in-time adaptive intervention app, there is still room for improvement. Had there been more time, I would have liked to do more hyperparameter tuning on the RF classifiers to improve its predictive power while keeping it light. I also would have liked to do the same with XGB Classifier, tuning it while keeping the n_estimators capped at 100 to avoid making the model heavier.

Alternative models that were not discussed here could also be looked into. Instead of randomly selecting data from each participant when preparing the training data, I could opt to preserve the datetime index and use it in combination with the features to predict blood alcohol content ahead of time, using forecasting methods like ARIMA or Facebook Prophet.

## Credit:

I would like to thank Chris Esposo for being an awesome Springboard mentor.

I would like to thank Jackson A Killian (jkillian '@' g.harvard.edu, Harvard University); Danielle R Madden (University of Southern California); John Clapp (University of Southern California) for uploading this valuable dataset to the UCI Machine Learning Repository.

## Sources:

Killian, J.A., Passino, K.M., Nandi, A., Madden, D.R. and Clapp, J., Learning to Detect Heavy Drinking Episodes Using Smartphone Accelerometer Data. In Proceedings of the 4th International Workshop on Knowledge Discovery in Healthcare Data co-located with the 28th International Joint Conference on Artificial Intelligence (IJCAI 2019) (pp. 35-42). http://ceur-ws.org/Vol-2429/paper6.pdf

Dataset: http://archive.ics.uci.edu/ml/datasets/Bar+Crawl%3A+Detecting+Heavy+Drinking

https://www.niaaa.nih.gov/publications/brochures-and-fact-sheets/alcohol-facts-and-statistics
https://pubs.niaaa.nih.gov/publications/aa74/aa74.htm
https://pubs.niaaa.nih.gov/publications/aa87/aa87.htm
