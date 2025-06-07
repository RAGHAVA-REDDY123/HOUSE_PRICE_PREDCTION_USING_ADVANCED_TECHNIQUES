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


