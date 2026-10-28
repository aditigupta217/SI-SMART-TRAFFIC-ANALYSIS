import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path

CSV_PATH = "/Users/aditi/Desktop/si-project/smart_city_traffic_mobility-selected-columns.csv"
REGRESSION_MODEL_PATH = "models/regression_pipeline.joblib"
CLASSIFICATION_MODEL_PATH = "models/classification_pipeline.joblib"

st.set_page_config(page_title="Smart Traffic Analysis", layout="wide")


def render_header(title, subtitle):
    st.html(
        "<div style='background:linear-gradient(90deg,#1f2a44,#2a6f97);"
        "padding:22px 26px;border-radius:12px;margin-bottom:1.3rem;'>"
        f"<div style='color:#ffffff;font-size:1.7rem;font-weight:700;'>{title}</div>"
        f"<div style='color:#d7e3f0;font-size:0.95rem;margin-top:6px;'>{subtitle}</div>"
        "</div>"
    )


def render_metric_cards(metrics):
    cards_html = "<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:1.2rem;'>"
    for label, value in metrics:
        cards_html += (
            "<div style='flex:1;min-width:180px;background-color:#ffffff;"
            "border:1px solid #e3e7ec;border-left:4px solid #2a6f97;"
            "border-radius:10px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            f"<div style='font-size:0.85rem;color:#5b6b82;margin-bottom:6px;'>{label}</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:#1f2a44;'>{value}</div>"
            "</div>"
        )
    cards_html += "</div>"
    st.html(cards_html)


def render_disclaimer(text):
    st.html(
        "<div style='background-color:#fff6e5;border-left:4px solid #e8a33d;"
        "color:#6b4f1d;padding:12px 16px;border-radius:8px;margin-top:10px;'>"
        f"{text}</div>"
    )


@st.cache_data
def load_data():
    data = pd.read_csv(CSV_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    return data


@st.cache_resource
def load_models():
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    return regression_model, classification_model


if not Path(CSV_PATH).exists():
    st.error(f"Could not find the dataset at: {CSV_PATH}. Update CSV_PATH at the top of app.py.")
    st.stop()

if not Path(REGRESSION_MODEL_PATH).exists() or not Path(CLASSIFICATION_MODEL_PATH).exists():
    st.error(
        "Trained model files were not found in the 'models' folder. "
        "Run the Practical 7, 8, and 10 cells in Smart_Traffic_Analysis.ipynb first "
        "to fit and save the models, then restart this app."
    )
    st.stop()

df = load_data()
regression_model, classification_model = load_models()

st.sidebar.title("Smart Traffic Analysis")
st.sidebar.caption("Statistics, Regression and Classification Dashboard")
page = st.sidebar.radio("Navigate", ["Dashboard", "Traffic Prediction", "Data Explorer"])
st.sidebar.markdown("---")
st.sidebar.write(f"Dataset rows: {len(df):,}")
st.sidebar.write(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

if page == "Dashboard":
    render_header("Traffic Analysis Dashboard", "A live overview of the Smart Traffic dataset.")

    render_metric_cards([
        ("Total Records", f"{len(df):,}"),
        ("Avg Vehicle Count", f"{df['vehicle_count'].mean():.1f}"),
        ("Avg Congestion Score", f"{df['congestion_score'].mean():.1f}"),
        ("Most Frequent Level", str(df["congestion_level"].mode().iloc[0])),
    ])

    st.markdown("### Explore a chart")
    chart_choice = st.selectbox(
        "Choose a chart",
        ["Average vehicle count by hour", "Congestion level distribution", "Vehicle count by road type"],
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if chart_choice == "Average vehicle count by hour":
        df.groupby("hour")["vehicle_count"].mean().plot(kind="line", marker="o", ax=ax, color="#2a6f97")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Average vehicle count")
    elif chart_choice == "Congestion level distribution":
        order = [c for c in ["Low", "Moderate", "High", "Severe"] if c in df["congestion_level"].unique()]
        sns.countplot(data=df, x="congestion_level", order=order, ax=ax, palette="viridis")
        ax.set_xlabel("Congestion level")
        ax.set_ylabel("Count")
    else:
        sns.boxplot(data=df, x="road_type", y="vehicle_count", ax=ax, palette="coolwarm")
        ax.set_xlabel("Road type")
        ax.set_ylabel("Vehicle count")
        ax.tick_params(axis="x", rotation=15)
    ax.set_title(chart_choice)
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Traffic Prediction":
    render_header("Traffic Prediction", "Enter the traffic conditions below to get a model-based prediction.")

    road_type_choices = sorted(df["road_type"].unique().tolist())
    weather_choices = sorted(df["weather_condition"].unique().tolist())
    city_zone_mode = df["city_zone"].mode().iloc[0]
    day_of_week_median = int(df["day_of_week"].median())

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            vehicle_count = st.number_input("Vehicle count", min_value=0, value=int(df["vehicle_count"].median()))
            motorcycle_count = st.number_input("Motorcycle count", min_value=0, value=int(df["motorcycle_count"].median()))
            public_transport_count = st.number_input("Public transport count", min_value=0, value=int(df["public_transport_count"].median()))
        with col2:
            speed_limit = st.number_input("Speed limit", min_value=0, value=int(df["speed_limit"].median()))
            lanes = st.number_input("Number of lanes", min_value=1, value=int(df["lanes"].median()))
            hour = st.slider("Hour of day", 0, 23, int(df["hour"].median()))
        with col3:
            temperature = st.number_input("Temperature", value=float(df["temperature"].median()))
            rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0, value=float(df["rainfall_mm"].median()))
            is_weekend = st.radio("Is weekend?", ["No", "Yes"], horizontal=True)
            is_holiday = st.radio("Is holiday?", ["No", "Yes"], horizontal=True)

        col4, col5 = st.columns(2)
        with col4:
            road_type = st.selectbox("Road type", road_type_choices)
        with col5:
            weather_condition = st.selectbox("Weather condition", weather_choices)

        submitted = st.form_submit_button("Predict")

    if submitted:
        is_weekend_val = 1 if is_weekend == "Yes" else 0
        is_holiday_val = 1 if is_holiday == "Yes" else 0

        regression_input = pd.DataFrame([{
            "vehicle_count": vehicle_count,
            "motorcycle_count": motorcycle_count,
            "public_transport_count": public_transport_count,
            "speed_limit": speed_limit,
            "lanes": lanes,
            "temperature": temperature,
            "rainfall_mm": rainfall_mm,
            "hour": hour,
            "is_weekend": is_weekend_val,
            "is_holiday": is_holiday_val,
            "road_type": road_type,
            "weather_condition": weather_condition,
        }])

        classification_input = pd.DataFrame([{
            "vehicle_count": vehicle_count,
            "motorcycle_count": motorcycle_count,
            "public_transport_count": public_transport_count,
            "speed_limit": speed_limit,
            "lanes": lanes,
            "hour": hour,
            "day_of_week": day_of_week_median,
            "is_weekend": is_weekend_val,
            "is_holiday": is_holiday_val,
            "temperature": temperature,
            "rainfall_mm": rainfall_mm,
            "road_type": road_type,
            "weather_condition": weather_condition,
            "city_zone": city_zone_mode,
        }])

        predicted_score = regression_model.predict(regression_input)[0]
        predicted_level = classification_model.predict(classification_input)[0]

        render_metric_cards([
            ("Predicted Congestion Score", f"{predicted_score:.1f} / 100"),
            ("Predicted Congestion Level", str(predicted_level)),
        ])

        render_disclaimer(
            "This is a model PREDICTION based on patterns learned from historical data, "
            "not a guaranteed real-world measurement."
        )

else:
    render_header("Data Explorer", "Summarize and visualize any column, with optional filters.")

    explore_cols = ["vehicle_count", "congestion_score", "fuel_waste_estimate", "temperature",
                     "rainfall_mm", "weather_condition", "road_type", "city_zone", "congestion_level"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        column_name = st.selectbox("Column", explore_cols)
    with col2:
        road_filter = st.selectbox("Road type filter", ["All"] + sorted(df["road_type"].unique().tolist()))
    with col3:
        weather_filter = st.selectbox("Weather filter", ["All"] + sorted(df["weather_condition"].unique().tolist()))
    with col4:
        congestion_filter = st.selectbox("Congestion level filter", ["All"] + sorted(df["congestion_level"].unique().tolist()))

    filtered = df.copy()
    if road_filter != "All":
        filtered = filtered[filtered["road_type"] == road_filter]
    if weather_filter != "All":
        filtered = filtered[filtered["weather_condition"] == weather_filter]
    if congestion_filter != "All":
        filtered = filtered[filtered["congestion_level"] == congestion_filter]

    if filtered.empty:
        st.warning("No rows match the selected filters.")
    else:
        st.write(f"Rows after filtering: {len(filtered):,}")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if filtered[column_name].dtype.kind in "if":
            st.dataframe(filtered[column_name].describe().round(2).to_frame())
            sns.histplot(filtered[column_name], kde=True, ax=ax, color="#2a6f97")
        else:
            st.dataframe(filtered[column_name].value_counts().to_frame(name="count"))
            sns.countplot(x=filtered[column_name], ax=ax, palette="viridis")
            ax.tick_params(axis="x", rotation=20)
        ax.set_title(f"{column_name} (n={len(filtered)} rows)")
        plt.tight_layout()
        st.pyplot(fig)
