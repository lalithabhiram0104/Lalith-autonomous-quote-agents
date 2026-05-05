import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("Loading dataset...")
df = pd.read_csv("/content/sample_data/Autonomous QUOTE AGENTS.csv")
df = df.drop([
    "Quote_Num",
    "Agent_Num",
    "Policy_Bind_DT"
], axis=1)

df["Q_Creation_DT"] = pd.to_datetime(df["Q_Creation_DT"])
df["Q_Valid_DT"] = pd.to_datetime(df["Q_Valid_DT"])

df["quote_valid_days"] = (df["Q_Valid_DT"] - df["Q_Creation_DT"]).dt.days

df = df.drop(["Q_Creation_DT","Q_Valid_DT"],axis=1)
df["Policy_Bind"] = df["Policy_Bind"].map({
    "Yes":1,
    "No":0
})
encoders = {}

for col in df.select_dtypes(include="object").columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

TARGET = "Policy_Bind"

X = df.drop(TARGET,axis=1)
y = df[TARGET]
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

smote = SMOTE(random_state=42)

X_train_res,y_train_res = smote.fit_resample(
    X_train,
    y_train
)

print("Training Risk Profiler Agent...")

risk_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

risk_model.fit(X_train_res,y_train_res)

print("Training Conversion Predictor Agent...")

conversion_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss"
)

conversion_model.fit(X_train_res,y_train_res)

print("Saving models...")

joblib.dump(risk_model,"risk_agent.pkl")
joblib.dump(conversion_model,"conversion_agent.pkl")
joblib.dump(encoders,"encoders.pkl")

print("Models saved successfully!")
