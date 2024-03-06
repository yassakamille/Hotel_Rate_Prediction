import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression,f_classif,chi2,SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.linear_model import LinearRegression
class regression:
        def __init__(self,data):
            self.data=data
        ############################################################################################
        def split(self):
            # SPLIT DATA TO XTRAIN XTEST
            self.feature = self.data.iloc[:, :-1]
            self.target = self.data.iloc[:, -1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.feature, self.target,test_size=0.2, random_state=0)


        def handling_tags(self,data_types):
            def fill_new(self, tags):
                if isinstance(tags, str) and "pet" in tags:
                    tags_list = tags.split(",")
                for tag in tags_list:
                    if "pet" in tag:
                        return tag.strip()
                return "nan"
            def fill_new1(self, tags):
                if isinstance(tags, str) and "trip" in tags:
                    tags_list = tags.split(",")
                for tag in tags_list:
                    if "trip" in tag:
                        return tag.strip()
                return "nan"
            def fill_new2(self, tags):
                if isinstance(tags, str) and "Couple" in tags or "Family" in tags or "Solo" in tags or "Group" in tags:
                    tags_list = tags.split(",")
                    for tag in tags_list:
                        if "Couple" in tag:
                            return tag.strip()
                        elif "Family" in tag:
                            return tag.strip()
                        elif "Solo" in tag:
                            return tag.strip()
                        elif "Group" in tag:
                            return tag.strip()
                return "nan"
            def fill_new3(self, tags):
                if isinstance(tags, str) and "room" in tags:
                    tags_list = tags.split(",")
                for tag in tags_list:
                    if "room" in tag:
                        return tag.strip()
                return "nan"
            def fill_new4(self, tags):
                if isinstance(tags, str) and "night" in tags:
                    tags_list = tags.split(",")
                for tag in tags_list:
                    if "night" in tag:
                        return tag.strip()
                return "nan"
            def fill_new5(self, tags):
                if isinstance(tags, str) and "Submitted" in tags:
                    tags_list = tags.split(",")
                for tag in tags_list:
                    if "Submitted" in tag:
                        return tag.strip()
                return "nan"
            data_types['with pet'] =data_types['Tags'].apply(lambda x:fill_new(x))
            data_types['kind of trip']=data_types['Tags'].apply(lambda x:fill_new1(x))
            data_types['number of guests'] =data_types['Tags'].apply(lambda x:fill_new2(x))
            data_types['rooms'] =data_types['Tags'].apply(lambda x:fill_new3(x))
            data_types['nights'] =data_types['Tags'].apply(lambda x:fill_new4(x))
            data_types['Submit way'] =data_types['Tags'].apply(lambda x:fill_new5(x))

            data_types['with pet'] =data_types['with pet'].str.strip("[]")
            data_types['kind of trip'] =data_types['kind of trip'].str.strip("[]")
            data_types['number of guests'] =data_types['number of guests'].str.strip("[]")
            data_types['rooms'] =data_types['rooms'].str.strip("[]")
            data_types['nights'] =data_types['nights'].str.strip("[]")
            data_types['Submit way'] =data_types['Submit way'].str.strip("[]")
            data_types.drop('Tags',inplace=True,axis=1)
        ###################################################################################
        def reg(self):

        ##################################################################################
            #check nulls from X_train
            print(" sum of nulls",self.feature.isna().sum())
            print(" sum of nulls",self.target.isna().sum())
            # print(self.X_train['lat'].value_counts())
            # print(self.X_train['lat'].mean())
            # print(self.X_train['lng'].value_counts())
            # print(self.X_train['lng'].mean())
            #print(self.X_test['lat'].value_counts())
            #print(self.X_test['lat'].mean())
            #print(self.X_test['lng'].value_counts())
            #print(self.X_test['lng'].mean())
            ############################################################################################
            #fill nulls with mean in x_train
            self.X_train['lat'].fillna(self.X_train['lat'].mean(),inplace=True)
            self.X_train['lng'].fillna(self.X_train['lng'].mean(),inplace=True)
            ##########################################################################################
            #check if duplicted rows
            print("duplicated rows in X_train:",self.X_train.duplicated())
            print("duplicated rows in X_test:",self.X_test.duplicated())
            ##########################################################################################
            #handle address column in train
            self.X_train["Countries"]=self.X_train["Hotel_Address"].apply(lambda X:X.split()[-1])
            self.X_train["State"]=self.X_train["Hotel_Address"].apply(lambda X:X.split()[-2])
            self.X_train =self.X_train.drop(['Hotel_Address'], axis=1)
            ###########################################################################################
            # Encode the categories columns
            le = LabelEncoder()
            cols = ['Countries', 'State', 'Reviewer_Nationality','Hotel_Name','Positive_Review','Negative_Review']
            for co in cols:
                self.X_train[co] = le.fit_transform(self.X_train[co])
            ##########################################################################################
            # handle Review_Date column in x_tarin
            NewDate = pd.to_datetime(self.X_train['Review_Date'],errors='ignore',yearfirst=True)
            self.X_train['Review_Date'] =NewDate.dt.month

            ############################################################################################
            # # take just num of days in x_tarin
            self.X_train["days_since_review"]=self.X_train["days_since_review"].apply(lambda X:X.split()[0])
            self.X_train['days_since_review']=pd.to_numeric(self.X_train['days_since_review'])
            ############################################################################################
            self.handling_tags(self.X_train)
            colo = ["with pet", "kind of trip", "number of guests", "rooms", "nights", "Submit way"]
            for col in colo:
                self.X_train[col] =le.fit_transform(self.X_train[col])

            ###########################################################################################
            selector = SelectKBest(score_func=f_regression, k=5)
            selector.fit_transform(self.X_train,self.y_train)
            ###########################################################################################
            scale = StandardScaler(copy=True,with_mean=True,with_std=True)
            self.X_train= scale.fit_transform(self.X_train)
            ###########################################################################################
            poly = PolynomialFeatures(degree=1)
            self.X_train=poly.fit_transform(self.X_train)
            ###########################################################################################
            reg = LinearRegression()
            reg.fit(self.X_train,self.y_train)
            ##########################################################################################
            lasso = Lasso()
            params = {
                'alpha': [0.1, 1.0, 10.0]
             }
            g_lasso = GridSearchCV(lasso, param_grid=params, cv=5)
            g_lasso.fit(self.X_train,self.y_train)
            print("Best parameters of lasso alpha= ",g_lasso.best_params_["alpha"])
            ############################################################################################
            ridge_model = Ridge()
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
            g_ridge= GridSearchCV(ridge_model, param_grid, cv=5)
            g_ridge.fit(self.X_train,self.y_train)
            print("Best parameters of Ridge alpha=",g_ridge.best_params_['alpha'])
            g_ridge.fit(self.X_train,self.y_train)
            ############################################################################################
             #fill nulls with mean in test
            self.X_test['lat'].fillna(self.X_test['lat'].mean(),inplace=True)
            self.X_test['lng'].fillna(self.X_test['lng'].mean(),inplace=True)
            ############################################################################################
            # handle address column in test
            self.X_test["Countries"]=self.X_test["Hotel_Address"].apply(lambda x1:x1.split()[-1])
            self.X_test["State"]=self.X_test["Hotel_Address"].apply(lambda x1:x1.split()[-2])
            self.X_test =self.X_test.drop(['Hotel_Address'], axis=1)
            ############################################################################################
            def encoding(self,data_type, name):
              self.data_type[name].fillna('Unknown', inplace=True)
              le.classes_ = np.append(le.classes_, 'Unknown')
              self.data_type[name] =self.data_type[name].apply(lambda x: x if x in le.classes_ else 'Unknown')
              self.data_type[name] = le.transform(self.data_type[name])

            colms = ['Countries', 'State', 'Reviewer_Nationality', 'Hotel_Name', 'Positive_Review', 'Negative_Review']
            for c in colms:
             encoding(self.X_test,c)

            ############################################################################################
             #handle Review_Date column in x_test
            NewDate = pd.to_datetime(self.X_test['Review_Date'],errors='ignore',yearfirst=True)
            self.X_test['Review_Date'] = NewDate.dt.month
            ############################################################################################
             # take just num of days in x_test
            self.X_test["days_since_review"]=self.X_test["days_since_review"].apply(lambda X:X.split()[0])
            self.X_test['days_since_review']=pd.to_numeric(self.X_test['days_since_review'])
            #############################################################################################
            self.handling_tags(self.X_train)
            colols=["with pet", "kind of trip", "number of guests","rooms","nights","Submit way"]
            for cololo in colols:
               encoding(self.X_test,cololo)
            self.X_test.drop('Tags',inplace=True,axis=1)
            ###########################################################################################
            #select best columns in X_test
            self.X_test= selector.transform(self.X_test)
            ###########################################################################################
            #scaling the X_test
            self.X_test= scale.transform(self.X_test)
            ###########################################################################################
            #test score of a liner regression
            self.X_test=poly.transform(self.X_test)
            print("liner regression Score:",reg.score(self.X_test,self.y_test))
            self.y_pred_reg = reg.predict(self.X_test)
            print('MSE of a liner regression:',mean_squared_error(self.y_test,self.y_pred_reg))
            print("##########################################################################################")
            best_lasso = g_lasso.best_estimator_
            self.y_pred_lasso = best_lasso.predict(self.X_test)
            print("lasso Score:",best_lasso.score(self.X_test,self.y_test))
            print('MSE of lasso:', mean_squared_error(self.y_test,self.y_pred_lasso))
            print("##########################################################################################")
            best_ridge = g_ridge.best_estimator_
            self.y_pred = best_ridge.predict(self.X_test)
            print("RIDGE Score:",best_ridge.score(self.X_test,self.y_test))
            print("MSE RIDGE:", mean_squared_error(self.y_test,self.y_pred))
            print("##########################################################################################")