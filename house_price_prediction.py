import pandas as pd 
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#filename = sys.argv[1]

from sklearn.externals import joblib

#lin_reg = joblib.load("my_model.pkl")

def house_price(filename, lin_reg):

    some_data = pd.read_csv(filename).drop("median_house_value", axis=1)
    some_labels = pd.read_csv(filename)["median_house_value"].copy()


    ## Custom Transformers

    from sklearn.base import BaseEstimator, TransformerMixin
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
      
      def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
      
      def fit(self, X, y=None):
        return self # nothing else to do
      
      
      def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
          bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
          return np.c_[X, rooms_per_household, population_per_household,
          bedrooms_per_room]
        
        else:
          return np.c_[X, rooms_per_household, population_per_household]


        




    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),])






    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    housing_num = some_data.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),
                                   ("cat", OneHotEncoder(), cat_attribs),])








    some_data_prepared = full_pipeline.fit_transform(some_data)

    print("Predictions:", lin_reg.predict(some_data_prepared))

    print("Labels:", list(some_labels))




    # Save test predictions to file
    some_data_prepared = pd.DataFrame(some_data_prepared)
    output = pd.DataFrame({'Id': some_data_prepared.index,'Y Original': some_labels, 'Y predicted':lin_reg.predict(some_data_prepared)})
    #strat_test_set.to_csv('data/train.csv', index=False)
    output.to_csv('files/outputTest.txt', index=False)
    
     
    output.to_html('output.html')



