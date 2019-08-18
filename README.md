# CloudsML
## Predicting the wheather with Python and Scikit


bring in intial libraries we will need
```
 import numpy
 import pandas
 import sklearn
 ```

bring in our data
 ```
 wheather = pandas.read_csv("C:/Users/Samue/Desktop/MyWebsite/CloudsML/seattleWeather_1948-2017Copy.csv", header = 0, sep = ",")
```
 We can specify headers manually by adding `names=["DATE", "PRCP", "TMAX", "TMIN", "RAIN"]` to above

```
>>> print(wheather.head())

      DATE  PRCP  TMAX  TMIN  RAIN
0  19480101  0.47    51    42  True
1  19480102  0.59    45    36  True
2  19480103  0.42    45    35  True
3  19480104  0.31    45    34  True
4  19480105  0.17    45    32  True
```


```
>>> print(wheather.describe())

        DATE          PRCP          TMAX          TMIN
count  2.555100e+04  25548.000000  25551.000000  25551.000000
mean   1.982543e+07      0.106222     59.544206     44.514226
std    2.019306e+05      0.239031     12.772984      8.892836
min    1.948010e+07      0.000000      4.000000      0.000000
25%    1.965063e+07      0.000000     50.000000     38.000000
50%    1.982122e+07      0.000000     58.000000     45.000000
75%    2.000062e+07      0.100000     69.000000     52.000000
max    2.017121e+07      5.020000    103.000000     71.000000
```


```
>>> print(wheather.isnull().values.any())
True
```

 We have a few days without rain data, such as `"1998-06-02","NA",72,52,"NA"`. let's just remove them.

```
 wheather = wheather.dropna()
 ```


 ```
 from sklearn.model_selection import train_test_split
 ```
 We want to predict the RAIN value, so let's break out data into two sets.
 X, the independent data, and Y, the dependent data we want to predict.
 ```
 X = wheather.drop(["PRCP", "RAIN"], axis=1) #all columns except PRCP and RAIN, as both give the answer away.
 y = wheather["PRCP"]
 ```

 Using the train_test_split function, we create the appropriate train/test data for our features ("X_train" and "X_test" respectively) and target data ("Y_train" and "Y_test"). We are specifying our test data to be 20% of the total data (80/20 split model, thanks Pareto).
 We are also providing a defined seed value (42) to be able to reproduce this split if we want to come back to it later.

 ```
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

>>> print(X_train.shape)
(20438, 3) #20438 training rows
>>> print(X_test.shape)
(5110, 3) #5110 testing rows
 ```



 ```
 from sklearn.tree import DecisionTreeRegressor
 from sklearn.ensemble import RandomForestRegressor

 tree_model = DecisionTreeRegressor()
 rf_model = RandomForestRegressor()

 tree_model.fit(X_train, y_train)
 rf_model.fit(X_train, y_train)
 ```



 model evaluation

```
 from sklearn.metrics import mean_squared_error
 from sklearn.metrics import mean_absolute_error


 tree_mse = mean_squared_error(y_train, tree_model.predict(X_train))
 tree_mae = mean_absolute_error(y_train, tree_model.predict(X_train))
 rf_mse = mean_squared_error(y_train, rf_model.predict(X_train))
 rf_mae = mean_absolute_error(y_train, rf_model.predict(X_train))

 from math import sqrt

 print("Decision Tree training mse = ",tree_mse," & mae = ",tree_mae," & rmse = ", sqrt(tree_mse))
 print("Random Forest training mse = ",rf_mse," & mae = ",rf_mae," & rmse = ", sqrt(rf_mse))


 tree_test_mse = mean_squared_error(y_test, tree_model.predict(X_test))
 tree_test_mae = mean_absolute_error(y_test, tree_model.predict(X_test))
 rf_test_mse = mean_squared_error(y_test, rf_model.predict(X_test))
 rf_test_mae = mean_absolute_error(y_test, rf_model.predict(X_test))

 print("Decision Tree test mse = ",tree_test_mse," & mae = ",tree_test_mae," & rmse = ", sqrt(tree_test_mse))
 print("Random Forest test mse = ",rf_test_mse," & mae = ",rf_test_mae," & rmse = ", sqrt(rf_test_mse))
```

 #Random forrest does better than decision tree, with an avg error of +-0.24 inches instead of +-0.29 inches of precipitation. Still, both aren't great.
 #Both are also doing much much better on the training data vs the test data, +-0.03 error vs 0.29 error is a *big* difference in inches.




```
 def display_scores(scores):
     print("Scores:", scores)
     print("Mean:", scores.mean())
     print("Standard deviation:", scores.std())
     print("\n")


 from sklearn.model_selection import cross_val_score

 scores = cross_val_score(tree_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
 tree_rmse_scores = numpy.sqrt(-scores)

 scores = cross_val_score(rf_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
 rf_rmse_scores = numpy.sqrt(-scores)

```

Let's see how our decision tree model preformed vs our random forest
```
>>> display_scores(tree_rmse_scores)
Scores: [0.32785747 0.3120757  0.31520675 0.3118263  0.28814834 0.31632178
 0.31382765 0.32981789 0.30310489 0.31043535]
Mean: 0.31286221219866617
Standard deviation: 0.011154876523176923

>>> display_scores(rf_rmse_scores)
Scores: [0.25463742 0.24316711 0.24797083 0.24010446 0.23350186 0.24951792
 0.24806436 0.23225629 0.23227359 0.25888471]
Mean: 0.24403785441735462
Standard deviation: 0.00893830076030914
```


```
 #provide date in form 19480103
 def predictVsActuals(day):
     precipPredictDay = X.loc[X['DATE'] == day]
     precipPredictValue = round(rf_model.predict(precipPredictDay)[0], 3)


     precipActual = wheather.loc[wheather['DATE'] == day]
     precipActualValue = round(precipActual.iloc[0]["PRCP"], 3)

     print("On", precipActual.iloc[0]["DATE"], "there was a high of", precipActual.iloc[0]["TMAX"], "and a low of", precipActual.iloc[0]["TMIN"])
     print("There were", precipActualValue, "inches of rainfall")
     print("We predicted",precipPredictValue, "inches of rainfall")
     print("we were off by ", round(abs(precipActualValue-precipPredictValue), 3), "inches")
     print("\n")
 ```

```
>>> predictVsActuals(19550302)
on 19550302 there was a high of 40 and a low of 29
There were 0.17 inches of rainfall
We predicted 0.135 inches of rainfall
we were off by  0.035 inches

>>> predictVsActuals(19880502)
on 19880502 there was a high of 47 and a low of 41
There were 0.33 inches of rainfall
We predicted 0.335 inches of rainfall
we were off by  0.005 inches


>>> predictVsActuals(19960815)
On 19960205 there was a high of 54 and a low of 40
There were 0.82 inches of rainfall
We predicted 0.021 inches of rainfall
we were off by  0.799 inches

 ```
