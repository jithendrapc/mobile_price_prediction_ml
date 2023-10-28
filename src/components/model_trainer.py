import os
import sys
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,VotingRegressor,ExtraTreesRegressor
from sklearn.ensemble import ( GradientBoostingRegressor,AdaBoostRegressor )
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
              "Random Forest" :RandomForestRegressor(),
              "Decision Tree" : DecisionTreeRegressor(),
              "Gradient Boosting" : GradientBoostingRegressor(),
              "Linear Regression" : LinearRegression(),
              "K_Neighbors Regressor" : KNeighborsRegressor(),
              "XGB Regressor" : XGBRegressor(),
              "CatBoosting Regressor": CatBoostRegressor(),
              "AdaBoost Regressor": AdaBoostRegressor(),
              "ExtraTree Regressor": ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='friedman_mse',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    max_samples=None, min_impurity_decrease=0.0, min_samples_leaf=1,
                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, n_jobs=-1, oob_score=False,
                    random_state=42, verbose=0, warm_start=False)   
                
            }
            
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score  < 0.6:
                raise CustomException("No best model")
            
            logging.info("Found best base model - {0}.".format(best_model_name))
                 
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
        