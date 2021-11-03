#loading packages
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

#loading the dataset
airline_train = pd.read_csv('data/train.csv')

#lowering the strings and replacing space with _
airline_train.columns= airline_train.columns.str.lower().str.replace(' ','_')

#dropping unnamed:_0 and id columns from dataframe
airline_train.drop(columns=['unnamed:_0','id'], inplace=True)

#dropping the duplicate rows
airline_train = airline_train.drop_duplicates()


categorical = airline_train.select_dtypes(include='object').columns.values

#lowering all the strings object columns in dataframe and replacing space with _
for column in categorical:
    airline_train[column] = airline_train[column].str.lower().str.replace(' ','_')

#replace null values instead of 0
arrival_delay_mean= np.mean(airline_train.arrival_delay_in_minutes)
airline_train.arrival_delay_in_minutes = airline_train.arrival_delay_in_minutes.fillna(arrival_delay_mean)

airline_train.satisfaction = airline_train.satisfaction.map({'neutral_or_dissatisfied':0,'satisfied':1})

#dropping departure_delay_in_minutes and class columns from dataframe
airline_train.drop(columns=['departure_delay_in_minutes', 'class'], inplace=True)

#preparing the data
y = airline_train.satisfaction.values
y = y.astype(int)

airline_train.drop(columns=['satisfaction'], inplace=True)

df_train, df_val, y_train, y_val = train_test_split(airline_train,y, test_size=0.3, stratify=y, random_state=1)

train_dict = df_train.to_dict(orient='records')

dv = DictVectorizer()
dv.fit(train_dict)

X_train = dv.transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)


#initialising hyperparameters values
max_depth = 15
min_samples_leaf = 3

#initializing the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=75,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1)

#training the model
rf.fit(X_train, y_train)

#saving the model
output_file ='model.bin'
f_out = open(output_file, 'wb') 
pickle.dump(rf, f_out)
f_out.close()

#saving the dictvectorizer
output_file ='dv.bin'
f_out = open(output_file, 'wb') 
pickle.dump(dv, f_out)
f_out.close()