import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Smartphone Price Intelligence",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{max-width:1250px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:2.2rem 2.4rem;border-radius:22px;background:linear-gradient(135deg,#eef4ff,#ffffff);border:1px solid #dce6f7;margin-bottom:1.5rem}
.hero-title{font-size:2.6rem;font-weight:800;letter-spacing:-1px;margin:0}
.hero-subtitle{color:#5b6472;font-size:1.05rem;margin-top:.55rem}
.section-title{font-size:1.35rem;font-weight:750;margin-top:1rem;margin-bottom:.2rem}
.section-caption{color:#6b7280;margin-bottom:1rem}
.prediction-card{padding:1.7rem 2rem;border-radius:20px;background:linear-gradient(135deg,#111827,#26354d);color:white;text-align:center;margin:1rem 0 1.5rem}
.prediction-label{font-size:.95rem;opacity:.78}
.prediction-price{font-size:3rem;font-weight:850;margin:.2rem 0}
.prediction-note{font-size:.9rem;opacity:.72}
div[data-testid="stMetric"]{border:1px solid #e5e7eb;border-radius:15px;padding:1rem;background:white}
.stButton>button{border-radius:12px;font-weight:700;min-height:3rem}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "smartphone_price_model.pkl"
DATA_PATH = BASE_DIR / "smartphone_cleaned_v5.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# Every dropdown below is generated from the dataset.
brand_options = sorted(df["brand_name"].dropna().unique().tolist())
processor_options = sorted(df["processor_brand"].dropna().unique().tolist())
os_options = sorted(df["os"].dropna().unique().tolist())
resolution_options = sorted(df["resolution"].dropna().unique().tolist())
num_core_options = sorted(df["num_cores"].dropna().unique().tolist())
ram_options = sorted(df["ram_capacity"].dropna().unique().tolist())
storage_options = sorted(df["internal_memory"].dropna().unique().tolist())
refresh_options = sorted(df["refresh_rate"].dropna().unique().tolist())
rear_camera_options = sorted(df["num_rear_cameras"].dropna().unique().tolist())
front_camera_options = sorted(df["num_front_cameras"].dropna().unique().tolist())
fast_charge_options = sorted(df["fast_charging"].dropna().unique().tolist())
extended_options = sorted(df["extended_upto"].dropna().unique().tolist())
speed_options = sorted(df["processor_speed"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("## 📱 Price Intelligence")
    st.caption("Random Forest smartphone price prediction")
    st.divider()
    st.markdown("### Model")
    st.metric("R² Score", "0.729")
    st.metric("MAE", "₹7,495")
    st.metric("Test Samples", "196")
    st.divider()
    st.markdown("### Dataset")
    st.write(f"**{len(df):,}** smartphones")
    st.write(f"**{df.shape[1]}** original columns")
    st.write(f"**{df['brand_name'].nunique()}** brands")
    st.write("**Outliers:** retained in training data")

st.markdown("""
<div class="hero">
<div class="hero-title">📱 Smartphone Price Predictor</div>
<div class="hero-subtitle">Estimate smartphone prices from hardware, camera, display, battery, connectivity, and storage specifications.</div>
</div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    st.markdown('<div class="section-title">📋 Basic Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">All categorical options come directly from the training dataset.</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: brand_name=st.selectbox("Brand",brand_options)
    with c2: processor_brand=st.selectbox("Processor Brand",processor_options)
    with c3: os=st.selectbox("Operating System",os_options)

    st.markdown('<div class="section-title">⚙️ Performance</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: rating=st.slider("Rating",int(df.rating.min()),int(df.rating.max()),int(df.rating.median()))
    with c2: num_cores=st.selectbox("CPU Cores",num_core_options,format_func=lambda x:f"{int(x)} cores")
    with c3:
        processor_speed=st.selectbox("Processor Speed",speed_options,
            index=speed_options.index(2.5) if 2.5 in speed_options else 0,
            format_func=lambda x:f"{x:g} GHz")

    st.markdown('<div class="section-title">💾 Memory & Storage</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: ram_capacity=st.selectbox("RAM",ram_options,index=ram_options.index(8) if 8 in ram_options else 0,format_func=lambda x:f"{int(x)} GB")
    with c2: internal_memory=st.selectbox("Internal Storage",storage_options,index=storage_options.index(128) if 128 in storage_options else 0,format_func=lambda x:f"{int(x)} GB")

    st.markdown('<div class="section-title">🖥️ Display</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: screen_size=st.number_input("Screen Size",float(df.screen_size.min()),float(df.screen_size.max()),6.6,0.01,format="%.2f")
    with c2: refresh_rate=st.selectbox("Refresh Rate",refresh_options,index=refresh_options.index(120) if 120 in refresh_options else 0,format_func=lambda x:f"{int(x)} Hz")
    with c3: resolution=st.selectbox("Screen Resolution",resolution_options)

    st.markdown('<div class="section-title">🔋 Battery & Charging</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: battery_capacity=st.number_input("Battery Capacity",int(df.battery_capacity.min()),int(df.battery_capacity.max()),5000,100,format="%d")
    with c2: fast_charging_available=st.selectbox("Fast Charging Available",[True,False],format_func=lambda x:"Yes" if x else "No")
    with c3:
        if fast_charging_available:
            fast_charging=st.selectbox("Fast Charging",fast_charge_options,index=fast_charge_options.index(25) if 25 in fast_charge_options else 0,format_func=lambda x:f"{int(x)} W")
        else:
            fast_charging=None
            st.info("Wattage is not used when fast charging is unavailable.")

    st.markdown('<div class="section-title">📸 Camera</div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: num_rear_cameras=st.selectbox("Rear Cameras",rear_camera_options)
    with c2: num_front_cameras=st.selectbox("Front Cameras",front_camera_options)
    with c3: primary_camera_rear=st.number_input("Primary Rear Camera",float(df.primary_camera_rear.min()),float(df.primary_camera_rear.max()),50.0,0.1,format="%.1f")
    with c4: primary_camera_front=st.number_input("Primary Front Camera",float(df.primary_camera_front.min()),float(df.primary_camera_front.max()),16.0,0.1,format="%.1f")

    st.markdown('<div class="section-title">📡 Connectivity & Expansion</div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: has_5g=st.selectbox("5G",[True,False],format_func=lambda x:"Yes" if x else "No")
    with c2: has_nfc=st.selectbox("NFC",[True,False],format_func=lambda x:"Yes" if x else "No")
    with c3: has_ir_blaster=st.selectbox("IR Blaster",[True,False],format_func=lambda x:"Yes" if x else "No")
    with c4: extended_memory_available=st.selectbox("Expandable Memory",[True,False],format_func=lambda x:"Yes" if x else "No")

    if extended_memory_available:
        extended_upto=st.selectbox("Expandable Storage Up To",extended_options,
            index=extended_options.index(1024) if 1024 in extended_options else 0,
            format_func=lambda x:f"{int(x)} GB")
    else:
        extended_upto=None

    st.write("")
    submitted=st.form_submit_button("🔮 Predict Smartphone Price",use_container_width=True,type="primary")

if submitted:
    input_data=pd.DataFrame({
        "rating":[rating],"has_5g":[has_5g],"has_nfc":[has_nfc],"has_ir_blaster":[has_ir_blaster],
        "num_cores":[num_cores],"processor_speed":[processor_speed],"battery_capacity":[battery_capacity],
        "fast_charging_available":[int(fast_charging_available)],"fast_charging":[fast_charging],
        "ram_capacity":[ram_capacity],"internal_memory":[internal_memory],"screen_size":[screen_size],
        "refresh_rate":[refresh_rate],"resolution":[resolution],"num_rear_cameras":[num_rear_cameras],
        "num_front_cameras":[num_front_cameras],"os":[os],"primary_camera_rear":[primary_camera_rear],
        "primary_camera_front":[primary_camera_front],"extended_memory_available":[int(extended_memory_available)],
        "extended_upto":[extended_upto],"brand_name":[brand_name],"processor_brand":[processor_brand]
    })
    try:
        prediction=model.predict(input_data)[0]
        st.markdown(f"""
        <div class="prediction-card">
        <div class="prediction-label">Estimated Smartphone Price</div>
        <div class="prediction-price">₹{prediction:,.0f}</div>
        <div class="prediction-note">Estimated by the trained Random Forest model</div>
        </div>
        """,unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1: st.metric("Model R²","0.729")
        with c2: st.metric("Mean Absolute Error","₹7,495")
        with c3: st.metric("Test Samples","196")
        with st.expander("🔎 View submitted specifications"):
            st.dataframe(input_data,use_container_width=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

with st.expander("ℹ️ About the model"):
    st.markdown("""
    **Model:** Random Forest Regressor  
    **Target:** `price`  
    **R²:** 0.729  
    **MAE:** ₹7,495  

    The model was evaluated on a held-out test set. R² is not an accuracy
    percentage; 0.729 means the model explained about 72.9% of the variation
    in price on that test set.
    """)

with st.expander("📌 How are outliers handled?"):
    st.markdown("""
    **The current production model does not remove or cap price outliers.**

    Extreme prices were retained because they are legitimate observations in
    the smartphone dataset, including premium/luxury devices. The Random Forest
    model therefore learns from the full training distribution.

    However, error analysis showed that prediction error is substantially larger
    for phones above ₹50,000. This means the model should be interpreted more
    cautiously for premium and luxury smartphones.

    **Important distinction:** the app's input limits are validation boundaries
    based on the dataset. They are not an outlier-removal method.
    """)

with st.expander("📊 Dataset information"):
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Smartphones",f"{len(df):,}")
    with c2: st.metric("Original Columns",df.shape[1])
    with c3: st.metric("Brands",df.brand_name.nunique())
    with c4: st.metric("Processor Brands",df.processor_brand.nunique())
    st.caption("Dropdown values are generated directly from the dataset.")
