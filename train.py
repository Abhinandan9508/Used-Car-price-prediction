import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from data_preprocess import preprocess_data



# ✅ CONFIG (PROJECT ROOT = D:\New folder)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "used_cars.csv"
MODEL_DIR = BASE_DIR / "model"

RF_MODEL_PATH = MODEL_DIR / "car_price_rf.pkl"
LR_MODEL_PATH = MODEL_DIR / "car_price_lr.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ✅ LOAD DATA
print("✅ Loading dataset from:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Optional column remove
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df = preprocess_data(df)

# ✅ TARGET
if "price" not in df.columns:
    raise ValueError("❌ Column 'price' not found in dataset!")

X = df.drop("price", axis=1)
y = df["price"]


# ✅ Column types
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns


# ✅ Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])


# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ✅ RANDOM FOREST
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)


# ✅ LINEAR REGRESSION
lr_model = LinearRegression()

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", lr_model)
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)


# ✅ Print results
print("\n📊 MODEL COMPARISON")
print("-" * 40)
print(f"Random Forest → R²: {rf_r2:.3f} | MAE: ₹{rf_mae:,.0f}")
print(f"Linear Reg.   → R²: {lr_r2:.3f} | MAE: ₹{lr_mae:,.0f}")


# ✅ Feature importance chart (RF only)
feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = rf_pipeline.named_steps["model"].feature_importances_

feature_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_df["feature"], feature_df["importance"])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


# ✅ Save models
with open(RF_MODEL_PATH, "wb") as f:
    pickle.dump(rf_pipeline, f)

with open(LR_MODEL_PATH, "wb") as f:
    pickle.dump(lr_pipeline, f)

print("\n✅ Both models trained and saved successfully")
print("✅ Saved RF Model:", RF_MODEL_PATH)
print("✅ Saved LR Model:", LR_MODEL_PATH)
