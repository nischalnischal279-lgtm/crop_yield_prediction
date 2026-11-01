# Crop Yield Prediction - Streamlit web app
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

TARGET = "crop_yield_tonnes_per_hectare"
NUMERIC = ["rainfall_mm", "temperature_c", "soil_ph", "fertilizer_kg_per_hectare"]
CATEGORICAL = ["crop_type", "irrigation", "soil_type"]


@st.cache_data
def load_data():
    return pd.read_csv("crop_yield_data.csv")


@st.cache_resource
def train_model(data):
    X = data.drop(columns=["farm_id", TARGET])
    y = data[TARGET]
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
    model = Pipeline([("preprocessor", pre), ("regressor", LinearRegression())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }
    return model, metrics, y_test, y_pred


data = load_data()
model, metrics, y_test, y_pred = train_model(data)

st.title("🌾 Crop Yield Prediction")
st.caption("Machine learning project using Python, Pandas, Matplotlib and Scikit-learn (Linear Regression).")

tab1, tab2, tab3 = st.tabs(["Predict yield", "Data analysis", "Model performance"])

# ---------- Tab 1: prediction ----------
with tab1:
    st.subheader("Enter farm details")
    c1, c2 = st.columns(2)
    with c1:
        crop = st.selectbox("Crop type", sorted(data["crop_type"].dropna().unique()))
        soil = st.selectbox("Soil type", sorted(data["soil_type"].dropna().unique()))
        irrigation = st.radio("Irrigation", ["Yes", "No"], horizontal=True)
    with c2:
        rainfall = st.slider("Rainfall (mm)", 200, 2000, 1100)
        temp = st.slider("Temperature (°C)", 10.0, 45.0, 27.0)
        ph = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
        fert = st.slider("Fertilizer (kg/hectare)", 0, 400, 150)

    if st.button("Predict yield", type="primary"):
        farm = pd.DataFrame([{
            "crop_type": crop, "rainfall_mm": rainfall, "temperature_c": temp,
            "soil_ph": ph, "fertilizer_kg_per_hectare": fert,
            "irrigation": irrigation, "soil_type": soil,
        }])
        pred = model.predict(farm)[0]
        st.success(f"Predicted crop yield: **{pred:.2f} tonnes/hectare**")

# ---------- Tab 2: data analysis ----------
with tab2:
    st.subheader("Dataset preview")
    st.write(f"{data.shape[0]} rows, {data.shape[1]} columns")
    st.dataframe(data.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data["rainfall_mm"], data[TARGET], alpha=0.6)
        ax.set_xlabel("Rainfall (mm)")
        ax.set_ylabel("Crop Yield (tonnes/hectare)")
        ax.set_title("Rainfall vs Crop Yield")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data["fertilizer_kg_per_hectare"], data[TARGET], alpha=0.6)
        ax.set_xlabel("Fertilizer (kg/hectare)")
        ax.set_ylabel("Crop Yield (tonnes/hectare)")
        ax.set_title("Fertilizer Usage vs Crop Yield")
        st.pyplot(fig)

    st.subheader("Top factors affecting crop yield")
    names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = model.named_steps["regressor"].coef_
    imp = pd.DataFrame({"feature": names, "coefficient": coefs})
    imp["abs"] = imp["coefficient"].abs()
    imp = imp.sort_values("abs", ascending=False).head(12)[["feature", "coefficient"]]
    st.dataframe(imp, use_container_width=True, hide_index=True)

# ---------- Tab 3: performance ----------
with tab3:
    st.subheader("Model evaluation (20% test data)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (t/ha)", f"{metrics['mae']:.3f}")
    m2.metric("RMSE (t/ha)", f"{metrics['rmse']:.3f}")
    m3.metric("R² score", f"{metrics['r2']:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual Yield (tonnes/hectare)")
    ax.set_ylabel("Predicted Yield (tonnes/hectare)")
    ax.set_title("Actual vs Predicted Crop Yield")
    st.pyplot(fig)
