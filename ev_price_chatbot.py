# ev_price_chatbot.py
import streamlit as st
import joblib
import pandas as pd
import re
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG & STYLE
# -------------------------------------------------
st.set_page_config(page_title="EV Price Predictor", page_icon="electric_car", layout="centered")

st.markdown("""
<style>
.big-price {font-size:2.8rem !important; font-weight:900; color:#2E8B57; text-align:center; margin:1.5rem 0;}
.spec-box {background:#f9f9fb; padding:1rem; border-radius:0.8rem; border:1px solid #ddd; font-family:monospace;}
.chat-user {background:#e3f2fd; border-radius:12px; padding:0.8rem; margin:0.5rem 0; border-left:4px solid #1976d2;}
.chat-bot  {background:#e8f5e9; border-radius:12px; padding:0.8rem; margin:0.5rem 0; border-left:4px solid #388e3c;}
.hero {background:linear-gradient(135deg,#1e3a8a,#3b82f6); color:white; padding:2.5rem; border-radius:1rem; text-align:center;}
.hint {font-size:0.9rem; color:#555; margin-top:0.5rem;}
.form-container {background:#f0f4f8; padding:1.5rem; border-radius:1rem; border:1px solid #bbd1ea;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("price_prediction_model.pkl")
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_model()

# -------------------------------------------------
# HERO
# -------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>EV Price Predictor</h1>
    <p>Supports <strong>12 brands</strong> • Chat or use Manual Form</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab_graph = st.tabs(["Chat Mode", "Manual Form", "Result Graphs"])

# ------------------------------------------------------------------
# HELPER: generate 20 synthetic neighbours
# ------------------------------------------------------------------
def generate_neighbours(base_specs: dict, n: int = 20):
    np.random.seed(42)
    df = pd.DataFrame([base_specs.copy() for _ in range(n)])
    num_cols = ["battery_capacity_kwh", "range_km", "top_speed_kmh",
                "fast_charging_power_kw_dc", "cargo_volume_l", "seats"]
    for col in num_cols:
        if col in df.columns:
            mean = base_specs.get(col, df[col].mean())
            std = mean * 0.12
            df[col] = np.clip(np.random.normal(mean, std, n),
                              df[col].min() * 0.7, df[col].max() * 1.3).astype(df[col].dtype)
    return df.round(2)

# ------------------------------------------------------------------
# TAB 1 – CHAT MODE (FIXED!)
# ------------------------------------------------------------------
with tab1:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Welcome message
    if not st.session_state.chat:
        welcome = "Hi! Try: *Tata Nexon EV 40 kWh SUV* or *MG ZS EV 350km range*"
        st.markdown(f'<div class="chat-bot"><strong>Assistant:</strong> {welcome}</div>', unsafe_allow_html=True)
        st.session_state.chat.append({"role": "assistant", "content": welcome})

    for msg in st.session_state.chat:
        role = "user" if msg["role"] == "user" else "assistant"
        cls = "chat-user" if role == "user" else "chat-bot"
        st.markdown(f'<div class="{cls}"><strong>{role.title()}:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("e.g. MG ZS EV 350km range"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-user"><strong>User:</strong> {prompt}</div>', unsafe_allow_html=True)

        if model is None:
            reply = "<p style='color:red;'>Model not loaded.</p>"
        else:
            text = prompt.lower()
            specs = {}

            # BRAND
            brand_map = {
                "tesla":"Tesla","tata":"Tata","mg":"MG","byd":"BYD","hyundai":"Hyundai",
                "kia":"Kia","bmw":"BMW","audi":"Audi","mercedes":"Mercedes","porsche":"Porsche",
                "volkswagen":"Volkswagen","vw":"Volkswagen","ford":"Ford"
            }
            for k, v in brand_map.items():
                if k in text:
                    specs["brand"] = v
                    break

            # MODEL
            model_map = {
                "model x":"Model X","model y":"Model Y","model 3":"Model 3",
                "nexon":"Nexon EV","zs":"ZS EV","ioniq":"Ioniq 5","ev6":"EV6",
                "taycan":"Taycan","q8":"Q8 e-tron","mach-e":"Mach-E","mach e":"Mach-E",
                "id.4":"ID.4","id4":"ID.4","eqs":"EQS","i4":"i4"
            }
            for k, v in model_map.items():
                if k in text:
                    specs["model"] = v
                    break

            # BATTERY & RANGE (FIXED!)
            battery_match = re.search(r"(\d+(?:\.\d+)?)\s*kwh", text)
            range_match = re.search(r"range\s*(\d+)", text)

            # Always ensure both exist
            if battery_match:
                specs["battery_capacity_kwh"] = float(battery_match.group(1))
            if range_match:
                specs["range_km"] = int(range_match.group(1))

            # Fallback logic: if one is missing, estimate the other
            if "battery_capacity_kwh" in specs and "range_km" not in specs:
                specs["range_km"] = int(specs["battery_capacity_kwh"] * 5.5)
            elif "range_km" in specs and "battery_capacity_kwh" not in specs:
                specs["battery_capacity_kwh"] = round(specs["range_km"] / 5.5, 1)
            elif "battery_capacity_kwh" not in specs and "range_km" not in specs:
                specs["battery_capacity_kwh"] = 50.0
                specs["range_km"] = 275

            # BODY & DRIVETRAIN
            if any(x in text for x in ["suv","xuv"]):   specs["car_body_type"] = "SUV"
            elif "sedan" in text:                       specs["car_body_type"] = "Sedan"
            elif any(x in text for x in ["hatch","hatchback"]): specs["car_body_type"] = "Hatchback"

            if any(x in text for x in ["awd","all wheel","4wd"]): specs["drivetrain"] = "AWD"
            elif "rwd" in text: specs["drivetrain"] = "RWD"
            elif "fwd" in text: specs["drivetrain"] = "FWD"

            # DEFAULTS
            defaults = {
                "top_speed_kmh": 200, "acceleration_0_100_s": 5.0, "seats": 5,
                "fast_charging_power_kw_dc": 150, "cargo_volume_l": 800,
                "car_body_type": specs.get("car_body_type", "SUV"),
                "drivetrain": specs.get("drivetrain", "AWD")
            }
            specs = {**defaults, **specs}

            # PREDICT
            try:
                df_in = pd.DataFrame([specs])
                price = model.predict(df_in)[0]
                price_str = f"₹{price:,.0f}"
                reply = f'<div class="big-price">{price_str}</div>'
                reply += "<details><summary><b>Extracted Specs</b></summary>"
                reply += f'<div class="spec-box"><pre>{json.dumps(specs, indent=2)}</pre></div></details>'

                # Store for graphs
                st.session_state.last_specs = specs
                st.session_state.last_price = price
            except Exception as e:
                reply = f"<p style='color:red;'>Prediction error: {e}</p>"

        st.markdown(reply, unsafe_allow_html=True)
        st.session_state.chat.append({"role": "assistant", "content": reply})

# ------------------------------------------------------------------
# TAB 2 – MANUAL FORM (unchanged)
# ------------------------------------------------------------------
with tab2:
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    st.subheader("Manual EV Price Prediction")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", ["Tesla","Tata","MG","BYD","Hyundai","Kia",
                                      "BMW","Audi","Mercedes","Porsche","Volkswagen","Ford"])
        model_name = st.selectbox("Model", ["Model X","Model Y","Model 3","Nexon EV","ZS EV",
                                           "Ioniq 5","EV6","Taycan","Q8 e-tron","Mach-E",
                                           "ID.4","EQS","i4"])
        battery = st.slider("Battery Capacity (kWh)", 30.0, 120.0, 75.0, 0.5)
        range_km = st.slider("Range (km)", 200, 700, 400, 10)

    with col2:
        body = st.selectbox("Body Type", ["SUV","Sedan","Hatchback"])
        drivetrain = st.selectbox("Drivetrain", ["AWD","RWD","FWD"])
        fast_charge = st.selectbox("Fast Charging (kW)", [50,100,150,200,250], index=2)
        seats = st.selectbox("Seats", [4,5,7])

    top_speed = st.slider("Top Speed (km/h)", 140, 260, 200, 5)
    cargo = st.slider("Cargo Volume (L)", 300, 1800, 800, 50)

    if st.button("Predict Price", type="primary"):
        if model is None:
            st.error("Model not loaded.")
        else:
            specs = {
                "brand": brand, "model": model_name,
                "battery_capacity_kwh": battery, "range_km": range_km,
                "top_speed_kmh": top_speed, "acceleration_0_100_s": 5.0,
                "seats": seats, "drivetrain": drivetrain,
                "car_body_type": body, "fast_charging_power_kw_dc": fast_charge,
                "cargo_volume_l": cargo
            }
            try:
                df_in = pd.DataFrame([specs])
                price = model.predict(df_in)[0]
                price_str = f"₹{price:,.0f}"
                st.markdown(f'<div class="big-price">{price_str}</div>', unsafe_allow_html=True)
                st.success("Prediction Successful!")
                with st.expander("View Full Specs"):
                    st.json(specs)
                st.session_state.last_specs = specs
                st.session_state.last_price = price
            except Exception as e:
                st.error(f"Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# TAB 3 – RESULT GRAPHS (6 perspectives)
# ------------------------------------------------------------------
with tab_graph:
    if "last_specs" not in st.session_state:
        st.info("Make a prediction to see the graphs.")
    else:
        specs = st.session_state.last_specs
        price = st.session_state.last_price
        neigh_df = generate_neighbours(specs, n=20)
        neigh_df["price"] = model.predict(neigh_df)
        user_df = pd.DataFrame([specs])
        user_df["price"] = price

        # 1. Price vs Battery
        fig1 = px.scatter(neigh_df, x="battery_capacity_kwh", y="price", size="range_km",
                          color="car_body_type", hover_data=["brand","model"],
                          title="Price vs Battery (kWh)")
        fig1.add_scatter(x=user_df["battery_capacity_kwh"], y=user_df["price"],
                         mode="markers", marker=dict(color="red", size=14, symbol="star"),
                         name="Your EV")
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Price vs Range
        fig2 = px.scatter(neigh_df, x="range_km", y="price", size="fast_charging_power_kw_dc",
                          color="drivetrain", title="Price vs Range (km)")
        fig2.add_scatter(x=user_df["range_km"], y=user_df["price"],
                         mode="markers", marker=dict(color="red", size=14, symbol="star"),
                         name="Your EV")
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Price vs Fast-Charge
        fig3 = px.scatter(neigh_df, x="fast_charging_power_kw_dc", y="price", size="top_speed_kmh",
                          color="seats", title="Price vs Fast-Charge (kW)")
        fig3.add_scatter(x=user_df["fast_charging_power_kw_dc"], y=user_df["price"],
                         mode="markers", marker=dict(color="red", size=14, symbol="star"),
                         name="Your EV")
        st.plotly_chart(fig3, use_container_width=True)

        # 4. Price vs Top-Speed
        fig4 = px.scatter(neigh_df, x="top_speed_kmh", y="price", size="cargo_volume_l",
                          color="car_body_type", title="Price vs Top Speed (km/h)")
        fig4.add_scatter(x=user_df["top_speed_kmh"], y=user_df["price"],
                         mode="markers", marker=dict(color="red", size=14, symbol="star"),
                         name="Your EV")
        st.plotly_chart(fig4, use_container_width=True)

        # 5. Price vs Cargo
        fig5 = px.scatter(neigh_df, x="cargo_volume_l", y="price", size="battery_capacity_kwh",
                          color="drivetrain", title="Price vs Cargo (L)")
        fig5.add_scatter(x=user_df["cargo_volume_l"], y=user_df["price"],
                         mode="markers", marker=dict(color="red", size=14, symbol="star"),
                         name="Your EV")
        st.plotly_chart(fig5, use_container_width=True)

        # 6. Radar Chart
        radar_cols = ["battery_capacity_kwh","range_km","top_speed_kmh",
                      "fast_charging_power_kw_dc","cargo_volume_l","seats"]
        radar_df = neigh_df[radar_cols].copy()
        for c in radar_cols:
            mn, mx = radar_df[c].min(), radar_df[c].max()
            radar_df[c] = (radar_df[c] - mn) / (mx - mn)
        user_norm = {c: (specs[c] - neigh_df[c].min())/(neigh_df[c].max()-neigh_df[c].min())
                     for c in radar_cols}
        fig6 = go.Figure()
        fig6.add_trace(go.Scatterpolar(r=radar_df.mean().values, theta=radar_cols,
                                       fill='toself', name='Neighbour Avg', line_color='lightgray'))
        fig6.add_trace(go.Scatterpolar(r=list(user_norm.values()), theta=radar_cols,
                                       fill='toself', name='Your EV', line_color='crimson'))
        fig6.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                           title="Spec Radar (normalised)")
        st.plotly_chart(fig6, use_container_width=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(f"""
<div style="text-align:center;margin-top:2rem;color:#555;font-size:0.9rem;">
    Developed by <strong>Aryan Singh</strong> • 
    <a href="https://github.com/aaryan100102" target="_blank">GitHub: @aaryan100102</a><br>
    Trained on Indian EV Market Data • {datetime.now().strftime("%B %Y")}
</div>
""", unsafe_allow_html=True)