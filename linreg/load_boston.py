import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()

X = pd.DataFrame(boston.data, columns = boston.feature_names)
y = pd.DataFrame(boston.target)

print(X.shape, y.shape)
print(X.head())  
print("\nAttribute Information:\n")
print("1. CRIM      per capita crime rate by town")
print("2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.")
print("3. INDUS     proportion of non-retail business acres per town")
print("4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) ")
print("5. NOX       nitric oxides concentration (parts per 10 million)")
print("6. RM        average number of rooms per dwelling")
print("7. AGE       proportion of owner-occupied units built prior to 1940")
print("8. DIS       weighted distances to five Boston employment centres")
print("9. RAD       index of accessibility to radial highways")
print("10. TAX      full-value property-tax rate per $10,000")
print("11. PTRATIO  pupil-teacher ratio by town")
print("12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
print("13. LSTAT    % lower status of the population")
print("14. MEDV     Median value of owner-occupied homes in $1000's")

