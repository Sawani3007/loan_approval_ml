import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/loan_data.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop loan_id (not useful for prediction)
df = df.drop(columns=["loan_id"])

# Handle missing values
df = df.dropna()

# Encode categorical columns
label_encoder = LabelEncoder()

categorical_cols = ["education", "self_employed", "loan_status"]

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Features & target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", accuracy)

# Save model
with open("model/loan_model.pkl", "wb") as f:
    pickle.dump(model, f)
