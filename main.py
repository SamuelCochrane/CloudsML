import numpy
import pandas
import sklearn

wheather = pandas.read_csv("C:/Users/Samue/Desktop/MyWebsite/CloudsML/seattleWeather_1948-2017Copy.csv", header = 0, sep = ",")
# can specify headers manually by adding names=["DATE", "PRCP", "TMAX", "TMIN", "RAIN"] to above
print(wheather.head())
print(wheather.describe())


print(wheather.isnull().values.any())
#We have a few days without rain data. let's just remove them.
#"1998-06-02","NA",72,52,"NA"
wheather = wheather.dropna()




from sklearn.model_selection import train_test_split

#We want to predict the RAIN value, so let's break out data into two sets. X, the independent data, and Y, the dependent data we want to predict.

X = wheather.drop(["PRCP", "RAIN"], axis=1) #all columns except PRCP and RAIN, as both give it away.
y = wheather["PRCP"]

#Using the train_test_split function, we create the appropriate train/test data for our features ("X_train" and "X_test" respectively) and target data ("Y_train" and "Y_test"). We are specifying our test data to be 20% of the total data (80/20 split model, thanks Pareto).
#We are also providing a seed (42) to be able to reproduce this split if we want to come back to it later.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape) #20438 training rows
print(X_test.shape) #5110 testing rows



#standardize
#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
#train_scaled = scaler.fit_transform(X_train[["PRCP", "TMAX", "TMIN"]])

#print(train_scaled.head())


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

tree_model = DecisionTreeRegressor()
rf_model = RandomForestRegressor()

tree_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


#model evaluation

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

#Random forrest does better than decision tree, with an avg error of +-0.24 inches instead of +-0.29 inches of precipitation. Still, both aren't great.
#Both are also doing much much better on the training data vs the test data, +-0.03 error vs 0.29 error is a *big* difference in inches.





def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print("\n")


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = numpy.sqrt(-scores)
display_scores(tree_rmse_scores)

scores = cross_val_score(rf_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
rf_rmse_scores = numpy.sqrt(-scores)
display_scores(rf_rmse_scores)








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

#predictVsActuals(19550302)
#predictVsActuals(19880502)
#predictVsActuals(19960815)


def predictFuture(day):
    
    precipPredictValue = round(rf_model.predict(day)[0], 3)


    print("We predicted",precipPredictValue, "inches of rainfall")

    print("\n")
