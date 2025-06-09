# HOUSE_PRICE_PREDCTION_USING_ADVANCED_TECHNIQUES <br/>
## PROJECT STRUCTURE <br/>
---1) _IMPORT NECESSARY LIBRARIES_ <br/>
---2) _IDENTIFYING & HANDLING MISSING VALUES_ <br/>
---3) _LABEL ENCODING THE CATEGORICAL VARIABLES_ <br/>
---4) _EXTRACTING NEW FEATURES BY FEATURE ENGINEERING_ <br/>
---5) _LOG TRANSFORM THE TARGET VARIABLE TO ENSURE IT IS UNIFORMLY DISTRIBUTED_ <br/>
---6) _SPLIT THE DATA INTO TRAIN AND TEST_ <br/>
---7) _INITIALIZE THE MODELS FOR TRAINING_ <br/>
---8) _PERFORM K-FOLD CROSS VALIDATION ON EACH MODEL_ <br/>
---9) _COMBINE THE OUT-OF-FOLD(OOF) PREDICTIONS OF EACH MODEL(STACKED FEATURES)_ <br/>
---10) _TRAIN A META MODEL ON THE STACK FEATURES_ <br/>
---11) _PREDICT EACK INDIVIDUAL MODEL ON TEST DATA_ <br/>
---12) _COMBINE & STACK THE TEST PREDICTIONS OF EACH INDIVIDUAL MODEL_ <br/>
---13) _MAKE FINAL PREDICTIONS BY TRAINING THE META MODEL ON THE COMBINED STACKED TEST PREDCTIONS_ <br/>
---14) _CONVERT THE FINAL PREDICTIONS BACK TO ORIGINAL SCALE_ <br/>
---15) _MAKE SUBMISSION BY CONVETING THE PREDICTIONS AND ITS CORRESPODING ID INTO A CSV FILE_ <br/> 

### STEP-1 IMPORT NECESSARY LIBRARIES <br/>
```
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,LabelEncoder
```
### DATA PROCESSING <br/>
Here the train and test data is concatenated into single df by removing the target variable in the train data <br/>
```
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
##concatenate train and test data
df = pd.concat(([train.drop('SalePrice', axis=1), test]), axis=0)
```
Here the categorical cols are filled with the most repeating value and numerical cols are filled with their mean <br/>
```
## Handle missing values
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(exclude=['object']).columns

# Fill missing values for categorical columns with mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill missing values for numerical columns with mean
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)
```
```
##label encoding of categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
```

### FEATURE ENGINEERING <br/>
Performed Feature Extraction by extracting new features from the given features <br/>
```
##feature engineering
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalBathrooms'] = df['FullBath'] + df['HalfBath'] + df['BsmtFullBath'] + df['BsmtHalfBath']
df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
df['TotalArea'] = df['GrLivArea'] + df['TotalSF'] + df['TotalPorchSF']
 
```
### IDENTIFYING TARGET VARIABLE STRUCTURE & DISTRIBUTION <br/>
The target variable is transformed into its equivalent logarithmic form <br/>
```
y = np.log1p(y)
```
#### Before applying log <br/>
![image](https://github.com/user-attachments/assets/080617c9-5746-42ad-a50f-c69a1d5204f3)

#### After Applying log <br/> <br/> <br/>
![image](https://github.com/user-attachments/assets/ec31b157-b33c-4c48-8b67-e65232ab9348)

## SPLIT THE DATA
```
##split the data  into train and test

X = df[:len(train)]
X_test = df[len(train):]
```
## K FOLD CROSS VALIDATION <br/>
The kfold cross validation split the data into 5 folds and performs 5 iterations
```
def get_oof_predictions(model,X,y):
    oof_preds = np.zeros(X.shape[0])
    kf = KFold(n_splits = 5,shuffle=True, random_state=42)
    for train_idx,val_idx in kf.split(X):
        X_train,X_val = X.iloc[train_idx],X.iloc[val_idx]
        y_train,y_val = y.iloc[train_idx],y.iloc[val_idx]
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)
    return oof_preds
```
In machine learning, OOF (Out-Of-Fold) predictions are used to evaluate a model's performance during cross-validation, while ensuring that each prediction is made on unseen data. This helps prevent overfitting and gives a more reliable estimation of model performance. <br/>

⚙️ How It Works
#### 1.Initialize:
oof_preds is a NumPy array of zeros to store predictions for every data point.

#### 2.K-Fold Splitting:
KFold(n_splits=5) divides the data into 5 parts. In each iteration 4 folds are used for training.1 fold is used for validation (i.e., out-of-fold).

#### 3.Training & Prediction:

The model is trained only on the training set.Predictions are made on the validation set only (unseen data for the model).These predictions are stored in the appropriate index of oof_preds.The function returns the complete OOF predictions array, which can be used for:Model blending /stacking and Calculating cross-validated metrics like RMSE, accuracy, etc.

```
cat_oof = get_oof_predictions(cat_model, X, y)
xgb_oof = get_oof_predictions(xgb_model, X, y)
rf_oof = get_oof_predictions(rf_model, X, y)
```
🔸 Generate out-of-fold predictions for each base model using 5-fold cross-validation.<br/>
🔸 Each element in *_oof is the prediction made only on unseen data — very useful for training a fair meta-model.<br/>

```
stacked_train = np.column_stack((cat_oof, xgb_oof, rf_oof))
```
Stack the OOF predictions into a new feature matrix of shape (n_samples, 3), where each column is the prediction from one base model. <br/>

```
meta_model = Ridge(alpha=0.01)
meta_model.fit(stacked_train, y)
```
 Train a meta-model (Ridge regression) on the stacked predictions to learn how to best combine them. <br/>
 ```
cat_model.fit(X, y)
xgb_model.fit(X, y)
rf_model.fit(X, y)
```
Now that stacking training is complete, we retrain all base models on full data to get their best versions ready for test-time predictions. <br/>
```
cat_test_preds = cat_model.predict(X_test)
xgb_test_preds = xgb_model.predict(X_test)
rf_test_preds = rf_model.predict(X_test)
```
Generate test predictions using the fully trained base models. <br/>

```
stacked_test = np.column_stack((cat_test_preds, xgb_test_preds, rf_test_preds))
final_preds = meta_model.predict(stacked_test)
```
 Stack the test predictions and feed them into the trained meta-model to get final stacked predictions. <br/>
```
final_preds = np.expm1(final_preds)
```
Finally, convert the predictions back to the original scale using inverse transformation (np.expm1) because your target y was likely log transformed earlier (e.g., using np.log1p to stabilize variance or normalize skew). <br/>

```
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
# Evaluate the model on the training data
evaluate_model(y, meta_model.predict(stacked_train))
```
Evaluates the final meta model on stacked_data(training data) <br/>

```
#submission
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': final_preds
})
submission.to_csv('submission4.csv', index=False)
```
make the finalpredictions on test data as a csv file <br/>

#  THANK YOU....

