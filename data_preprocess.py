import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop unwanted column
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # ✅ Convert numeric columns safely
    num_cols = ["manufacturing_year", "km_driven", "number_of_owners", "price"]

    for col in num_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ✅ Convert all object columns to string
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # ✅ Drop missing after conversion
    df.dropna(inplace=True)

    return df
