
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from tkinter import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression,f_classif
print("Let's gooooooooooooooooooooooooo")
print("Buena suerte ")
warnings.filterwarnings("ignore", category=RuntimeWarning)

def Dontat_Circle(names , size):
    print("Circle Here")
    print(f"names is {names}")
    print(f"names size is {size}")
    fig = plt.figure()
    fig.patch.set_facecolor('black')
    plt.rcParams['text.color'] = 'white'
    plt.title("Distribution of Classes")
    my_circle=plt.Circle( (0,0), 0.7, color='black')
    plt.pie(size, labels=names)
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

def BarGraph(train_times ,test_times ,models ,ModelsType):
    plt.figure(figsize=(8, 6))
    bar_width = 0.35
    r1 = np.arange(len(train_times))
    r2 = [x + bar_width for x in r1]
    plt.bar(r1, train_times, color='b', width=bar_width, edgecolor='white', label='Train Time')
    plt.bar(r2, test_times, color='g', width=bar_width, edgecolor='white', label='Test Time')
    plt.xticks([r + bar_width/2 for r in range(len(train_times))], models, rotation=45)
    plt.yscale('log')
    plt.ylabel('Time (log scale)')
    plt.title(f'Train and Test Times for {ModelsType} Models')
    plt.legend()
    plt.show()
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
##########################################################################################
##########################################################################################

data=pd.read_csv('hotel-regression-dataset.csv')
# SPLIT DATA TO XTRAIN XTEST
feature = data.iloc[:, :-1]
target = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=0)
############################################################################################
# print(" sum of nulls",X_train.isna().sum())
# print(" sum of nulls",y_train.isna().sum())
# ###########################################################################################
# # check nulls from x_tst
# print(" sum of nulls",X_test.isna().sum())
# print(" sum of nulls",y_test.isna().sum())
###########################################################################################
# fill nulls with mean in x_train
X_train['lat'].fillna(X_train['lat'].mean(),inplace=True)
X_train['lng'].fillna(X_train['lng'].mean(),inplace=True)
###########################################################################################
#check if duplicted rows
data.drop_duplicates(inplace=True)
##########################################################################################
#handle address column in train
X_train["Countries"]=X_train["Hotel_Address"].apply(lambda X:X.split()[-1])
# X_train["State"]=X_train["Hotel_Address"].apply(lambda X:X.split()[-2])
X_train =X_train.drop(['Hotel_Address'], axis=1)
###########################################################################################
# Encode the 'Countries' column
le = LabelEncoder()
cols = ['Countries', 'Reviewer_Nationality','Hotel_Name','Positive_Review','Negative_Review']
for co in cols:
    X_train[co] = le.fit_transform(X_train[co])

print("Hold a second ")
##########################################################################################
#handle Review_Date column in x_tarin
NewDate = pd.to_datetime(X_train['Review_Date'],errors='ignore',yearfirst=True)
X_train['Review_Date'] = NewDate.dt.month
############################################################################################
# take just num of days in x_tarin
X_train["days_since_review"]=X_train["days_since_review"].apply(lambda X:X.split()[0])
X_train['days_since_review']=pd.to_numeric(X_train['days_since_review'])
#############################################################################################
#handle Tags column in X_train
def fill_new(tags,value):
    if isinstance(tags, str) and value in tags:
        tags_list = tags.split(",")
        for tag in tags_list:
            if value in tag:
                return tag.strip()
    return "nan"

def fill_new2(tags):
    if isinstance(tags, str) and "Couple" in tags or "Family" in tags or "Solo" in tags or "Group" in tags:
        tags_list = tags.split(",")
        for tag in tags_list:
            if "Couple" in tag:
                return tag.strip()
            elif"Family" in tag:
                return tag.strip()
            elif "Solo" in tag:
                return tag.strip()
            elif "Group" in tag:
                return tag.strip()
    return "nan"

print("Hold a second ")
# Apply the functions to create the "with pet" and "Leisure trip" columns
X_train['with pet'] = X_train['Tags'].apply(lambda x: fill_new(x,"pet"))
X_train['kind of trip']= X_train['Tags'].apply(lambda x: fill_new(x,"trip"))
X_train['number of guests'] = X_train['Tags'].apply(lambda x: fill_new2(x))
X_train['rooms'] = X_train['Tags'].apply(lambda x: fill_new(x,"room"))
X_train['nights'] = X_train['Tags'].apply(lambda x: fill_new(x,"night"))
X_train['Submit way'] = X_train['Tags'].apply(lambda x: fill_new(x,"Submitted"))

X_train['with pet'] = X_train['with pet'].str.strip("[]")
X_train['kind of trip'] = X_train['kind of trip'].str.strip("[]")
X_train['number of guests'] = X_train['number of guests'].str.strip("[]")
X_train['rooms'] = X_train['rooms'].str.strip("[]")
X_train['nights'] = X_train['nights'].str.strip("[]")
X_train['Submit way'] = X_train['Submit way'].str.strip("[]")
colo=["with pet", "kind of trip", "number of guests","rooms","nights","Submit way"]
X_train.drop('Tags',inplace=True,axis=1)

for col in colo:
    X_train[col] = le.fit_transform(X_train[col])

print("Here we go ")
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X_train,y_train)
X_train= selector.transform(X_train)
###########################################################################################
scale = StandardScaler(copy=True,with_mean=True,with_std=True)
X_train= scale.fit_transform(X_train)
############################################################################################
poly=PolynomialFeatures(degree=1)
X_train=poly.fit_transform(X_train)

print("I'll Show u")

reg = LinearRegression()

TrainTimeForLinearRegerssion = time.time()
reg.fit(X_train,y_train)
TrainTimeForLinearRegerssion = time.time()-TrainTimeForLinearRegerssion
print(f"Train time for Linear Regerssion Model = {TrainTimeForLinearRegerssion}")

fileLinearRegression= 'linear_regression.sav'
pickle.dump(reg, open(fileLinearRegression, 'wb'))
############################################################################################


lasso = Lasso()
params = {
    'alpha': [0.1, 1.0, 10.0]}
g_lasso = GridSearchCV(lasso, param_grid=params, cv=5)

TrainTimeForlassoRegression = time.time()
g_lasso.fit(X_train, y_train)
TrainTimeForlassoRegression = time.time() - TrainTimeForlassoRegression
print(f"Train time for Lasso Model = {TrainTimeForlassoRegression}")

fileLasso= 'lasso_regression.sav'
g_lasso_best=g_lasso.best_estimator_
pickle.dump(g_lasso_best, open(fileLasso, 'wb'))
print("Best parameters of lasso alpha= ",g_lasso.best_params_["alpha"])

############################################################################################



ridge_model = Ridge()
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
g_ridge= GridSearchCV(ridge_model, param_grid, cv=5)

TrainTimeForRidgeRegerssion = time.time()
g_ridge.fit(X_train, y_train)
TrainTimeForRidgeRegerssion = time.time() -TrainTimeForRidgeRegerssion 
print(f"Train time for Ridge Model = {TrainTimeForRidgeRegerssion}")

fileRidge= 'ridge_regression.sav'
g_ridge_best=g_ridge.best_estimator_
pickle.dump(g_ridge_best, open(fileRidge, 'wb'))
print("Best parameters of Ridge alpha=",g_ridge.best_params_['alpha'])
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
# #fill nulls with mean in test
X_test['lat'].fillna(X_test['lat'].mean(),inplace=True)
X_test['lng'].fillna(X_test['lng'].mean(),inplace=True)
###########################################################################################
# #handle address column in test
X_test["Countries"]=X_test["Hotel_Address"].apply(lambda x1:x1.split()[-1])
# X_test["State"]=X_test["Hotel_Address"].apply(lambda x1:x1.split()[-2])
X_test =X_test.drop(['Hotel_Address'], axis=1)
# # ###########################################################################################
# # #encoded X test
def encoding(data,name):
    data[name].fillna('Unknown', inplace=True)
    le.classes_ = np.append(le.classes_, 'Unknown')
    data[name] = data[name].apply(lambda x: x if x in le.classes_ else 'Unknown')
    data[name] = le.transform(data[name])
colms = ['Countries', 'Reviewer_Nationality','Hotel_Name','Positive_Review','Negative_Review']
for c in colms:
    encoding(X_test,c)

############################################################################################
#handle Review_Date column in x_test
NewDate = pd.to_datetime(X_test['Review_Date'],errors='ignore',yearfirst=True)
X_test['Review_Date'] = NewDate.dt.month
############################################################################################
# take just num of days in x_test
X_test["days_since_review"]=X_test["days_since_review"].apply(lambda X:X.split()[0])
X_test['days_since_review']=pd.to_numeric(X_test['days_since_review'])
#############################################################################################
# handle Tags column in X_test
X_test['with pet'] = X_test['Tags'].apply(lambda x: fill_new(x,"pet"))
X_test['kind of trip']= X_test['Tags'].apply(lambda x: fill_new(x,"trip"))
X_test['number of guests'] = X_test['Tags'].apply(lambda x: fill_new2(x))
X_test['rooms'] = X_test['Tags'].apply(lambda x: fill_new(x,"room"))
X_test['nights'] = X_test['Tags'].apply(lambda x: fill_new(x,"night"))
X_test['Submit way'] = X_test['Tags'].apply(lambda x: fill_new(x,"Submitted"))

X_test['with pet'] = X_test['with pet'].str.strip("[]")
X_test['kind of trip'] = X_test['kind of trip'].str.strip("[]")
X_test['number of guests'] = X_test['number of guests'].str.strip("[]")
X_test['rooms'] = X_test['rooms'].str.strip("[]")
X_test['nights'] = X_test['nights'].str.strip("[]")
X_test['Submit way'] = X_test['Submit way'].str.strip("[]")
colols=["with pet", "kind of trip", "number of guests","rooms","nights","Submit way"]
for cololo in colols:
    encoding(X_test,cololo)

X_test.drop('Tags',inplace=True,axis=1)

###########################################################################################
#select best columns in X_test
X_test= selector.transform(X_test)
###########################################################################################
#scaling the X_test
X_test= scale.transform(X_test)
###########################################################################################
#test score of a liner regression
X_test=poly.transform(X_test)

LinearRegressionmodel = pickle.load(open(fileLinearRegression, 'rb'))

TestTimeForLinearRegression = time.time()
y_pred_reg = LinearRegressionmodel.predict(X_test)
TestTimeForLinearRegression = time.time() - TestTimeForLinearRegression
print(f"Test time for LinearRegression Model = {TestTimeForLinearRegression}")

print('MSE of a liner regression:',mean_squared_error(y_test,y_pred_reg))
print("##########################################################################################")


Lassomodel = pickle.load(open(fileLasso, 'rb'))

TestTimeForLasso = time.time()
y_pred_lasso = Lassomodel.predict(X_test)
TestTimeForLasso = time.time()-TestTimeForLasso
print(f"Test time for Lasso Model = {TestTimeForLasso}")

print('MSE of lasso:', mean_squared_error(y_test,y_pred_lasso))
print("##########################################################################################")


Ridgemodel= pickle.load(open(fileRidge, 'rb'))

TestTimeForRidge = time.time()
y_pred_ridge= Ridgemodel.predict(X_test)
TestTimeForRidge = time.time() - TestTimeForRidge
print(f"Test time for Ridge Model = {TestTimeForRidge}")

print("MSE RIDGE:", mean_squared_error(y_test,y_pred_ridge))
# Evaluate performance using mean squared error
print("##########################################################################################")


TrainTime = [TrainTimeForLinearRegerssion ,TrainTimeForlassoRegression ,TrainTimeForRidgeRegerssion ]
TestTime =  [TestTimeForLinearRegression ,TestTimeForLasso ,TestTimeForRidge ]
modelNames = ['LinearRegerssion','lasso','Ridge']
BarGraph(TrainTime ,TestTime ,modelNames ,"Regerssion")

Regmean_squared_error = mean_squared_error(y_test, y_pred_reg)
Lassomean_squared_error = mean_squared_error(y_test, y_pred_lasso)
Ridgemean_squared_error = mean_squared_error(y_test, y_pred_ridge)
final_data_Accuracy_Regression = pd.DataFrame({'Models': ['LinearReg', 'lasso', 'Ridge'],
                                    'Accuracy': [Regmean_squared_error,
                                                Lassomean_squared_error,
                                                Ridgemean_squared_error]})

print(final_data_Accuracy_Regression)
print("liner regression Score:",reg.score(X_test, y_test))
print("lasso Score:",g_lasso_best.score(X_test, y_test))
print("RIDGE Score:",g_ridge_best.score(X_test, y_test))

#*************************************************************
######################### Old calssification###################################################
######################### Old calssification###################################################
######################### Old calssification###################################################
######################### Old calssification###################################################
######################### Old calssification###################################################
df=pd.read_csv('hotel-classification-dataset.csv')


##########################################################################################
n_feature=df.iloc[:,:-1]
n_target=df.iloc[:,-1]

n_X_train,n_X_test,n_y_train,n_y_test = train_test_split(n_feature, n_target, test_size=0.2, random_state=0)
print(n_X_train.shape)
print(n_X_test.shape)
##########################################################################################
#check nulls from X_train
# #print(n_X_train['lat'].value_counts())
# #print(n_X_train['lat'].mean())
# #print(n_X_train['lng'].value_counts())
# #print(n_X_train['lng'].mean())
# print(" sum of nulls",n_X_train.isna().sum())
# print(" sum of nulls",n_y_train.isna().sum())

############################################################################################
# #check nulls from x_tst
# #print(n_X_test['lat'].value_counts())
# #print(n_X_test['lat'].mean())
# #print(n_X_test['lng'].value_counts())
# #print(n_X_test['lng'].mean())
# print(" sum of nulls",n_X_test.isna().sum())
# print(" sum of nulls",n_y_test.isna().sum())
############################################################################################
#fill nulls with mean in x_train
n_X_train['lat'].fillna(n_X_train['lat'].mean(),inplace=True)
n_X_train['lng'].fillna(n_X_train['lng'].mean(),inplace=True)
# ###########################################################################################
df.drop_duplicates(inplace=True)
##########################################################################################
#handle address column in train
n_X_train["Countries"]=n_X_train["Hotel_Address"].apply(lambda X:X.split()[-1])
# n_X_train["State"]=n_X_train["Hotel_Address"].apply(lambda X:X.split()[-2])
n_X_train =n_X_train.drop(['Hotel_Address'], axis=1)

Dontat_Circle(n_y_train.unique(),n_y_train.value_counts())
Dontat_Circle(n_y_test.unique(),n_y_test.value_counts())

# sns.countplot(n_y_train)
# plt.show()

# sns.countplot(n_y_test)
# plt.show()

###########################################################################################
le_clasific =  LabelEncoder()
le_clasification = LabelEncoder()
n_y_train=le_clasific.fit_transform(n_y_train)
cols = ['Countries', 'Reviewer_Nationality','Hotel_Name','Positive_Review','Negative_Review']
for co in cols:
    n_X_train[co] = le_clasification.fit_transform(n_X_train[co])
##########################################################################################
#handle Review_Date column in x_tarin
NewDate = pd.to_datetime(n_X_train['Review_Date'],errors='ignore',yearfirst=True)
n_X_train['Review_Date'] = NewDate.dt.month
############################################################################################
# take just num of days in x_tarin
n_X_train["days_since_review"]=n_X_train["days_since_review"].apply(lambda X:X.split()[0])
n_X_train['days_since_review']=pd.to_numeric(n_X_train['days_since_review'])
#############################################################################################
#handle Tags column in X_train

# Apply the functions to create the "with pet" and "Leisure trip" columns
n_X_train['with pet'] = n_X_train['Tags'].apply(lambda x: fill_new(x,"pet"))
n_X_train['kind of trip']= n_X_train['Tags'].apply(lambda x: fill_new(x,"trip"))
n_X_train['number of guests'] = n_X_train['Tags'].apply(lambda x: fill_new2(x))
n_X_train['rooms'] = n_X_train['Tags'].apply(lambda x: fill_new(x,"room"))
n_X_train['nights'] = n_X_train['Tags'].apply(lambda x: fill_new(x,"night"))
n_X_train['Submit way'] = n_X_train['Tags'].apply(lambda x: fill_new(x,"Submitted"))

n_X_train['with pet'] = n_X_train['with pet'].str.strip("[]")
n_X_train['kind of trip'] = n_X_train['kind of trip'].str.strip("[]")
n_X_train['number of guests'] = n_X_train['number of guests'].str.strip("[]")
n_X_train['rooms'] = n_X_train['rooms'].str.strip("[]")
n_X_train['nights'] = n_X_train['nights'].str.strip("[]")
n_X_train['Submit way'] = n_X_train['Submit way'].str.strip("[]")
colo=["with pet", "kind of trip", "number of guests","rooms","nights","Submit way"]
n_X_train.drop('Tags',inplace=True,axis=1)
for col in colo:
    n_X_train[col] = le_clasification.fit_transform(n_X_train[col])

selector1= SelectKBest(score_func=f_classif, k=5)
selector1.fit(n_X_train,n_y_train)
n_X_train= selector1.transform(n_X_train)
###########################################################################################
n_X_train= scale.fit_transform(n_X_train)
###########################################################################################
c=0.1
# c=1.0
# c=10.0

LogRegression = LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=100 ,multi_class='ovr')

TrainTimeForLogisticRegression = time.time()
LogRegression.fit(n_X_train,n_y_train)
TrainTimeForLogisticRegression = time.time() - TrainTimeForLogisticRegression
print(f"Train time for Logistic calssification Model = {TrainTimeForLogisticRegression}")

fileLogRegression = 'Logistic_Regression_classifier.sav'
pickle.dump(LogRegression, open(fileLogRegression, 'wb'))
###########################################################################################
print("OH here we again Line (470) ")

min=1
# min=2
# min=3
dtc = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2, min_samples_leaf=min)

TrainTimeForDecisionTreeClassifier = time.time()
dtc.fit(n_X_train,n_y_train)
TrainTimeForDecisionTreeClassifier = time.time() - TrainTimeForDecisionTreeClassifier
print(f"Train time for DecisionTreeClassifier calssification Model = {TrainTimeForDecisionTreeClassifier}")

fileDecisionTree = 'decision_tree_classifier.sav'
pickle.dump(dtc, open(fileDecisionTree, 'wb'))
# #############################################################################################

#maxi=10
# maxi=100
maxi=500
rfc = RandomForestClassifier(n_estimators=100, max_depth=maxi, max_features='sqrt')

TrainTimeForRandomForestClassifier = time.time()
rfc.fit(n_X_train,n_y_train)
TrainTimeForRandomForestClassifier = time.time() - TrainTimeForRandomForestClassifier
print(f"Train time for RandomForestClassifier calssification Model = {TrainTimeForRandomForestClassifier}")

fileRandomForest= 'Random_Forest_classifier.sav'
pickle.dump(rfc, open(fileRandomForest, 'wb'))
#############################################################################################
# #fill nulls with mean in test
n_X_test['lat'].fillna(n_X_test['lat'].mean(),inplace=True)
n_X_test['lng'].fillna(n_X_test['lng'].mean(),inplace=True)
###########################################################################################
# #handle address column in test
n_X_test["Countries"]=n_X_test["Hotel_Address"].apply(lambda x1:x1.split()[-1])
# n_X_test["State"]=n_X_test["Hotel_Address"].apply(lambda x1:x1.split()[-2])
n_X_test =n_X_test.drop(['Hotel_Address'], axis=1)
###########################################################################################
#encoded X test
n_y_test=le_clasific.fit_transform(n_y_test)
colms = ['Countries', 'Reviewer_Nationality','Hotel_Name','Positive_Review','Negative_Review']
for c in colms:
    encoding(n_X_test,c)
############################################################################################
#handle Review_Date column in x_test
NewDate = pd.to_datetime(n_X_test['Review_Date'],errors='ignore',yearfirst=True)
n_X_test['Review_Date'] = NewDate.dt.month
# ############################################################################################
# take just num of days in x_test
n_X_test["days_since_review"]=n_X_test["days_since_review"].apply(lambda X:X.split()[0])
n_X_test['days_since_review']=pd.to_numeric(n_X_test['days_since_review'])
#############################################################################################

# handle Tags column in X_test
n_X_test['with pet'] = n_X_test['Tags'].apply(lambda x: fill_new(x,"pet"))
n_X_test['kind of trip']= n_X_test['Tags'].apply(lambda x: fill_new(x,"trip"))
n_X_test['number of guests'] = n_X_test['Tags'].apply(lambda x: fill_new2(x))
n_X_test['rooms'] = n_X_test['Tags'].apply(lambda x: fill_new(x,"room"))
n_X_test['nights'] = n_X_test['Tags'].apply(lambda x: fill_new(x,"night"))
n_X_test['Submit way'] = n_X_test['Tags'].apply(lambda x: fill_new(x,"Submitted"))

n_X_test['with pet'] = n_X_test['with pet'].str.strip("[]")
n_X_test['kind of trip'] = n_X_test['kind of trip'].str.strip("[]")
n_X_test['number of guests'] = n_X_test['number of guests'].str.strip("[]")
n_X_test['rooms'] = n_X_test['rooms'].str.strip("[]")
n_X_test['nights'] = n_X_test['nights'].str.strip("[]")
n_X_test['Submit way'] = n_X_test['Submit way'].str.strip("[]")
colols=["with pet", "kind of trip", "number of guests","rooms","nights","Submit way"]

def encodingClassification(data,name):
    data[name].fillna('Unknown', inplace=True)
    le_clasification.classes_ = np.append(le_clasification.classes_, 'Unknown')
    data[name] = data[name].apply(lambda x: x if x in le_clasification.classes_ else 'Unknown')
    data[name] = le_clasification.transform(data[name])
colms = ['Countries', 'Reviewer_Nationality','Hotel_Name','Positive_Review','Negative_Review']
for cololo in colols:
    encodingClassification(n_X_test,cololo)

n_X_test.drop('Tags',inplace=True,axis=1)
###########################################################################################
#select best columns in X_test
n_X_test= selector1.transform(n_X_test)
###########################################################################################
#scaling the X_test
n_X_test= scale.transform(n_X_test)
###########################################################################################

fileLogRegressionmodel = pickle.load(open(fileLogRegression, 'rb'))

TestTimeForLogisticRegression = time.time()
lr_y_pred = fileLogRegressionmodel.predict(n_X_test)
TestTimeForLogisticRegression = time.time() - TestTimeForLogisticRegression
print(f"Test time for Prediction Using Logistig Classifier Model = {TestTimeForLogisticRegression}")

print("LOGISTIG REG Score:",LogRegression.score(n_X_test,n_y_test))
print("MSE LOGISTIG REG:", mean_squared_error(n_y_test,lr_y_pred))
print("###########################################################################################")

RandomForestmodel = pickle.load(open(fileRandomForest, 'rb'))

TestTimeForTestRandomForestClassifier = time.time()
rfc_y_pred = RandomForestmodel.predict(n_X_test)
TestTimeForTestRandomForestClassifier = time.time() - TestTimeForTestRandomForestClassifier
print(f"Test time for Prediction Using RandomForestClassifier Model = {TestTimeForTestRandomForestClassifier}")

# print("Random Forest Score:",rfc.score(n_X_test,n_y_test))
print("MSE Random Forest:", mean_squared_error(n_y_test,rfc_y_pred))
print("###########################################################################################")

DescisionTreemodel = pickle.load(open(fileDecisionTree, 'rb'))

TestTimeForTestDescisionTree = time.time()
dtc_y_pred = DescisionTreemodel.predict(n_X_test)
TestTimeForTestDescisionTree = time.time() - TestTimeForTestDescisionTree
print(f"Test time for Prediction Using DescisionTree Classification Model = {TestTimeForTestDescisionTree}")

# print("Descision Tree Score:",dtc.score(n_X_test,n_y_test))
print("MSE Descision Tree:", mean_squared_error(n_y_test,dtc_y_pred))
print("###########################################################################################")
print("Line(405)")


Logaccuracy = accuracy_score(n_y_test, lr_y_pred)
RfCaccuracy = accuracy_score(n_y_test, rfc_y_pred)
DTCaccuracy = accuracy_score(n_y_test, dtc_y_pred)

final_data_Accuracy_Classification = pd.DataFrame({'Models': ['LogisticReg', 'RandomFT', 'DecisionTree'],
                                    'Accuracy': [Logaccuracy,
                                                RfCaccuracy,
                                                DTCaccuracy]})

print("**********************************************************")
print("**********************************************************")
print("**********************************************************")
print("**********************************************************")
print("**********************************************************")
print(final_data_Accuracy_Classification)

TrainTime = [TrainTimeForLogisticRegression ,TrainTimeForDecisionTreeClassifier ,TrainTimeForRandomForestClassifier ]
TestTime =  [TestTimeForLogisticRegression ,TestTimeForTestDescisionTree ,TestTimeForTestRandomForestClassifier ]
modelNames = ['Logistic','DecisionTreeClassifier','RandomForest']
BarGraph(TrainTime ,TestTime ,modelNames ,"Classification")
print(final_data_Accuracy_Classification)
print(f"Train time for Logistic calssification Model = {TrainTimeForLogisticRegression}")
print(f"Train time for DecisionTreeClassifier calssification Model = {TrainTimeForDecisionTreeClassifier}")
print(f"Train time for RandomForestClassifier calssification Model = {TrainTimeForRandomForestClassifier}")
print(f"Test time for Prediction Using Logistig Classifier Model = {TestTimeForLogisticRegression}")
print(f"Test time for Prediction Using RandomForestClassifier Model = {TestTimeForTestRandomForestClassifier}")
print(f"Test time for Prediction Using DescisionTree Classification Model = {TestTimeForTestDescisionTree}")
################################## New Regression ******************************************
################################## New Regression ************************************
################################## New Regression *********************************
################################## New Regression *****************************
################################## New Regression ***********************
################################## New Regression ********************
################################## New Regression *****************
################################## New Regression ***************
################################## New Regression *************
################################## New Regression ***********
################################## New Regression *********
################################## New Regression ********


####### NEED TO NADEL MISSING VALUES IN TEH NEW DATA IN EACH COLUMN


data1 = pd.read_csv('Book2.csv')

# for column in data1.columns :
#     if data1[column].isnull().sum() > 0 :
#         if type(data1[column][1]) != "str" : 
#             data1[column] = data1[column].fillna(data1[column].mode()[0])
#             data1 = data1.dropna(axis = 0)
#             data1 = data1.reset_index(drop = True)
#         else :
#             data1[column] = data1[column].fillna(data1[column].min())

x_=data1.iloc[:,:-1]
y=data1.iloc[:,-1]
x_.fillna(x_.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)
data1.drop_duplicates(inplace=True)
x_["Countries"] = x_["Hotel_Address"].apply(lambda X: X.split()[-1])
# x_["State"] =x_["Hotel_Address"].apply(lambda X: X.split()[-2])
x_=x_.drop(['Hotel_Address'], axis=1)
colms = ['Countries', 'Reviewer_Nationality', 'Hotel_Name', 'Positive_Review', 'Negative_Review']

for c in colms:
    encoding(x_, c)
NewDate = pd.to_datetime(x_['Review_Date'], errors='ignore', yearfirst=True)
x_['Review_Date'] = NewDate.dt.month
x_["days_since_review"] = x_["days_since_review"].apply(lambda X: X.split()[0])
x_['days_since_review'] = pd.to_numeric(x_['days_since_review'])
x_['with pet'] = x_['Tags'].apply(lambda x: fill_new(x,"pet"))
x_['kind of trip'] = x_['Tags'].apply(lambda x: fill_new(x,"trip"))
x_['number of guests'] = x_['Tags'].apply(lambda x: fill_new2(x))
x_['rooms'] = x_['Tags'].apply(lambda x: fill_new(x,"room"))
x_['nights'] = x_['Tags'].apply(lambda x: fill_new(x,"night"))
x_['Submit way'] = x_['Tags'].apply(lambda x: fill_new(x,"Submitted"))
x_['with pet'] = x_['with pet'].str.strip("[]")
x_['kind of trip'] = x_['kind of trip'].str.strip("[]")
x_['number of guests'] = x_['number of guests'].str.strip("[]")
x_['rooms'] = x_['rooms'].str.strip("[]")
x_['nights'] = x_['nights'].str.strip("[]")
x_['Submit way'] = x_['Submit way'].str.strip("[]")
colols = ["with pet", "kind of trip", "number of guests", "rooms", "nights", "Submit way"]

for cololo in colols:
    encoding(x_, cololo)
x_.drop('Tags', inplace=True, axis=1)
x_ = selector.transform(x_)
###########################################################################################
# scaling the X_test
x_ = scale.transform(x_)
x_ = poly.transform(x_)
loaded_model1 = pickle.load(open(fileLinearRegression, 'rb'))
###########################################################################################
y_pred_reg1 = loaded_model1.predict(x_)
###########################################################################################
print("liner regression Score:", reg.score(x_,y))
print('MSE of a liner regression:', mean_squared_error(y, y_pred_reg1))
print("##########################################################################################")
loaded_model2 = pickle.load(open(fileLasso, 'rb'))
###########################################################################################
y_pred_lasso1 = loaded_model2.predict(x_)
###########################################################################################
print("lasso Score:", g_lasso_best.score(x_, y))
print('MSE of lasso:', mean_squared_error(y, y_pred_lasso1))
print("##########################################################################################")
loaded_model3 = pickle.load(open(fileRidge, 'rb'))
###########################################################################################
y_pred_ridge1 = loaded_model3.predict(x_)
###########################################################################################
print("RIDGE Score:", g_ridge_best.score(x_, y))
print("MSE RIDGE:", mean_squared_error(y, y_pred_ridge1))
# Evaluate performance using mean squared error
print("##########################################################################################")
result = pd.DataFrame({'Logistic Regression': y_pred_reg1,
                    'Random Forest Classifier': y_pred_lasso1,
                    'Decision Tree Classifier': y_pred_ridge1})
# Print the result DataFrame
print(result.head(10))
print(result['Logistic Regression'].min())
print(result['Random Forest Classifier'].min())
print(result['Decision Tree Classifier'].min())


################################## New classification ******************************************
################################## New classification ************************************
################################## New classification *********************************
################################## New classification *****************************
################################## New classification ***********************
################################## New classification ********************
################################## New classification *****************
################################## New classification ***************
################################## New classification *************
################################## New classification ***********
################################## New classification *********
################################## New classification ********

data2 = pd.read_csv('Book3.csv')
x_1=data2.iloc[:, :-1]
y1=data2.iloc[:,-1]
# Look At me Know
y1=le.fit_transform(y1)
x_1.fillna(x_.mean(), inplace=True)
data2.drop_duplicates(inplace=True)
x_1["Countries"] = x_1["Hotel_Address"].apply(lambda X: X.split()[-1])
# x_1["State"] =x_1["Hotel_Address"].apply(lambda X: X.split()[-2])
x_1=x_1.drop(['Hotel_Address'], axis=1)
colms = ['Countries', 'Reviewer_Nationality', 'Hotel_Name', 'Positive_Review', 'Negative_Review']
for c in colms:
    encodingClassification(x_1, c)
NewDate = pd.to_datetime(x_1['Review_Date'], errors='ignore', yearfirst=True)
x_1['Review_Date'] = NewDate.dt.month
x_1["days_since_review"] = x_1["days_since_review"].apply(lambda X: X.split()[0])
x_1['days_since_review'] = pd.to_numeric(x_1['days_since_review'])

x_1['with pet'] = x_1['Tags'].apply(lambda x: fill_new(x,"pet"))
x_1['kind of trip'] = x_1['Tags'].apply(lambda x: fill_new(x,"trip"))
x_1['number of guests'] = x_1['Tags'].apply(lambda x: fill_new2(x))
x_1['rooms'] = x_1['Tags'].apply(lambda x: fill_new(x,"room"))
x_1['nights'] = x_1['Tags'].apply(lambda x: fill_new(x,"night"))
x_1['Submit way'] = x_1['Tags'].apply(lambda x: fill_new(x,"Submitted"))

x_1['with pet'] = x_1['with pet'].str.strip("[]")
x_1['kind of trip'] = x_1['kind of trip'].str.strip("[]")
x_1['number of guests'] = x_1['number of guests'].str.strip("[]")
x_1['rooms'] = x_1['rooms'].str.strip("[]")
x_1['nights'] = x_1['nights'].str.strip("[]")
x_1['Submit way'] = x_1['Submit way'].str.strip("[]")
colols = ["with pet", "kind of trip", "number of guests", "rooms", "nights", "Submit way"]
for cololo in colols:
    encodingClassification(x_1, cololo)
x_1.drop('Tags', inplace=True, axis=1)
x_1= selector1.transform(x_1)
x_1= scale.transform(x_1)
###########################################################################################

loaded_model4= pickle.load(open(fileLogRegression , 'rb'))

###########################################################################################

lr_y_pred1= loaded_model4.predict(x_1)
###########################################################################################


print("LOGISTIG REG Score:",LogRegression.score(x_1,y1))
print("MSE LOGISTIG REG:", mean_squared_error(y1,lr_y_pred1))
print("###########################################################################################")

loaded_model5 = pickle.load(open(fileRandomForest, 'rb'))

###########################################################################################

rfc_y_pred1= loaded_model5.predict(x_1)
###########################################################################################


print("Random Forest Score:",loaded_model5.score(x_1,y1))
print("MSE Random Forest:", mean_squared_error(y1,rfc_y_pred1))
print("###########################################################################################")

loaded_model6 = pickle.load(open(fileDecisionTree  , 'rb'))

###########################################################################################

dtc_y_pred1= loaded_model6.predict(x_1)
###########################################################################################


print("Descision Tree Score:",loaded_model6.score(x_1,y1))
print("MSE Descision Tree:", mean_squared_error(y1,dtc_y_pred1))  
print("###########################################################################################")


print("###########################################################################################")
print(le_clasific.inverse_transform(lr_y_pred1))
print("###########################################################################################")
print(le_clasific.inverse_transform(rfc_y_pred1))
print("###########################################################################################")
print(le_clasific.inverse_transform(dtc_y_pred1))

lr_pred_labels = le_clasific.inverse_transform(lr_y_pred1)
rfc_pred_labels = le_clasific.inverse_transform(rfc_y_pred1)
dtc_pred_labels = le_clasific.inverse_transform(dtc_y_pred1)

result = pd.DataFrame({'Logistic Regression': lr_pred_labels,
                    'Random Forest Classifier': rfc_pred_labels,
                    'Decision Tree Classifier': dtc_pred_labels})

# Print the result DataFrame
print(result.tail(10))

print("Finlay Finito It's so hard to be good for end ,bro")


# Tages already modified
# Train time calculated
# Test Time calculated
# Time for scripts ?????????
# Visualisation using matplotlib.pyplot semi Done
