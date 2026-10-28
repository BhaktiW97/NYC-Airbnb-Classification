#1.Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#2.Load dataset
df=pd.read_csv("AB_NYC_2019.csv")
df.shape
df.head()
df.isnull().sum()
df.info()
df.describe()
df.duplicated()

#3.EDA 

#3.1 missing values
missing=df.isnull().sum()
print(missing[missing>0])

#3.2 Target variable
print(df["room_type"].value_counts())
sns.countplot(x="room_type",data=df)
plt.show()

#3.3 Univariate Analysis:Numeris feature
numerical_cols=["price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]
df[numerical_cols].hist(bins=30,figsize=(12,8))

#3.4 Univariate Analysis:Ctegorical feature
sns.countplot(data=df,x="neighbourhood_group")
plt.show()

#3.5 Bivariate Analysis: Features vs Target
sns.boxplot(x="room_type",y="price",data=df)
plt.show()

#3.6 Corelation between Numeric features
corr=df[numerical_cols + ["latitude","longitude"]].corr()
sns.heatmap(corr,annot=True)
plt.show()

#3.7 Geographical Distribution 
sns.scatterplot(x="longitude",y="latitude",data=df,hue="room_type")
plt.show()

#4.Data Cleaning and Feature Engineering
#1. Droping columns that don't help us
df_clean=df.drop(columns=["id","name","host_id","host_name","last_review"])

#2.No reviews-> 0 reviews per month ,not missing
df_clean["reviews_per_month"]=df_clean["reviews_per_month"].fillna(0)

#3. Cap Extreme outliers instead of deleting rows 
price_cap=df_clean["price"].quantile(0.99)
nights_cap=df_clean["minimum_nights"].quantile(0.99)
df_clean["price"]=df_clean["price"].clip(upper=price_cap)
df_clean["minimum_nights"]=df_clean["minimum_nights"].clip(upper=nights_cap)

df_clean

#4.4.Separating x and y
x=df_clean.drop(columns=["room_type"])
y=df_clean["room_type"]

df_clean[numerical_cols].skew()

#5.Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42,stratify=y)

#6.Preprocessing:ColumnTransformer + Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PowerTransformer
from sklearn.compose import ColumnTransformer

categorical_cols=["neighbourhood_group","neighbourhood"]
numerical_cols=["price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365","latitude","longitude"]

#1.Pipeline->numeric columns
numerc_pipeline=Pipeline(steps=[('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])

#2.Pipeline-> Categorical columns
categorical_pipeline=Pipeline(steps=[('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))])

preprocessor=ColumnTransformer(transformers=[('numerical',numerc_pipeline,numerical_cols),('categorical',categorical_pipeline,categorical_cols)])
preprocessor

#Trying multiple algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.metrics import make_scorer,f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

models={
    "Logistic Regresssion":LogisticRegression(class_weight="balanced",random_state=42),
    "Decision Tree":DecisionTreeClassifier(class_weight="balanced",random_state=42),
    "Random Forest":RandomForestClassifier(class_weight="balanced",random_state=42),
    "Gradient Boosting":GradientBoostingClassifier(random_state=42),
}


for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    accuracy=cross_val_score(pipe,x_train,y_train,cv=3,scoring="accuracy")
    macrof1=cross_val_score(pipe,x_train,y_train,cv=3,scoring="f1_macro")
    print(
        f"{name} -> Accuracy: {accuracy.mean():.3f}, "
        f"Macro F1: {macrof1.mean():.3f}"
    )
#Hyperparameter Tunning 
from sklearn.model_selection import RandomizedSearchCV
best_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

param_distribution = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [8, 12, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10]
}

search = RandomizedSearchCV(
    estimator=best_pipeline,
    param_distributions=param_distribution,
    n_iter=10,
    cv=3,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1
)

search.fit(x_train, y_train)

print("Best Parameters:", search.best_params_)
print("Best Cross-Validation Macro F1:", search.best_score_)

#Final evaluation on the test set
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
best_pipeline=search.best_estimator_
y_pred=best_pipeline.predict(x_test)

print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")

print(f"f1 Score: {f1_score(y_test,y_pred,average="macro")}")

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt=".2f",xticklabels=best_pipeline.classes_,yticklabels=best_pipeline.classes_)

#Saving the final model artifact
import joblib
joblib.dump(best_pipeline,"Model_Pipeline.pkl")