import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------

st.set_page_config(
    page_title="Oil Supply Risk Dashboard",
    page_icon="🛢️",
    layout="wide"
)


# ---------------------------------------------------------
# Static data
# ---------------------------------------------------------

CHOKEPOINTS = {
    "Strait of Hormuz": {
        "lat": 26.56,
        "lon": 56.25,
        "region": "Persian Gulf",
        "description": "Critical export route for Gulf oil producers including Saudi Arabia, Iraq, Kuwait, UAE, Qatar and Iran.",
        "risk_base": "High",
        "daily_flow": "Very high"
    },
    "Suez Canal": {
        "lat": 30.58,
        "lon": 32.27,
        "region": "Egypt",
        "description": "Major route between Europe and Asia, important for crude oil, refined products and LNG.",
        "risk_base": "Medium",
        "daily_flow": "High"
    },
    "Bab el-Mandeb": {
        "lat": 12.58,
        "lon": 43.32,
        "region": "Red Sea / Gulf of Aden",
        "description": "Strategic passage connecting the Red Sea to the Arabian Sea. Important for Europe-Asia energy flows.",
        "risk_base": "High",
        "daily_flow": "Medium to high"
    },
    "Strait of Malacca": {
        "lat": 2.45,
        "lon": 101.20,
        "region": "Southeast Asia",
        "description": "Key route for Middle East oil flows toward China, Japan, South Korea and Southeast Asia.",
        "risk_base": "Medium",
        "daily_flow": "Very high"
    },
    "Turkish Straits": {
        "lat": 41.12,
        "lon": 29.07,
        "region": "Turkey",
        "description": "Important route for oil exports from the Black Sea and Caspian region.",
        "risk_base": "Medium",
        "daily_flow": "Medium"
    },
    "Panama Canal": {
        "lat": 9.08,
        "lon": -79.68,
        "region": "Panama",
        "description": "Important global shipping route, though less central for Middle East crude flows.",
        "risk_base": "Low",
        "daily_flow": "Medium"
    },
    "Cape of Good Hope": {
        "lat": -34.36,
        "lon": 18.47,
        "region": "South Africa",
        "description": "Alternative route when Suez or Red Sea passages are disrupted. Longer transit time and higher shipping cost.",
        "risk_base": "Low",
        "daily_flow": "Alternative route"
    }
}


ROUTES = [
    {
        "name": "Persian Gulf to East Asia",
        "points": [
            (26.5, 52.0),
            (26.56, 56.25),
            (15.0, 65.0),
            (7.0, 80.0),
            (2.45, 101.20),
            (22.3, 114.2),
            (35.6, 139.8)
        ],
        "color": "#ff7f0e"
    },
    {
        "name": "Persian Gulf to Europe via Suez",
        "points": [
            (26.5, 52.0),
            (26.56, 56.25),
            (12.58, 43.32),
            (20.0, 38.0),
            (30.58, 32.27),
            (36.0, 15.0),
            (45.0, 5.0)
        ],
        "color": "#d62728"
    },
    {
        "name": "Red Sea to Europe",
        "points": [
            (12.58, 43.32),
            (20.0, 38.0),
            (30.58, 32.27),
            (36.0, 15.0),
            (51.0, 1.0)
        ],
        "color": "#9467bd"
    },
    {
        "name": "Black Sea to Mediterranean",
        "points": [
            (44.0, 35.0),
            (41.12, 29.07),
            (38.0, 24.0),
            (36.0, 15.0)
        ],
        "color": "#2ca02c"
    },
    {
        "name": "Alternative: Cape of Good Hope",
        "points": [
            (26.5, 52.0),
            (12.58, 43.32),
            (-10.0, 45.0),
            (-34.36, 18.47),
            (-10.0, 0.0),
            (36.0, -5.0),
            (51.0, 1.0)
        ],
        "color": "#1f77b4"
    },
    {
        "name": "US Gulf to Europe",
        "points": [
            (29.7, -95.0),
            (25.0, -80.0),
            (35.0, -50.0),
            (50.0, -5.0)
        ],
        "color": "#8c564b"
    },
    {
        "name": "US Gulf to Asia via Panama",
        "points": [
            (29.7, -95.0),
            (9.08, -79.68),
            (10.0, -120.0),
            (25.0, -150.0),
            (35.6, 139.8)
        ],
        "color": "#17becf"
    }
]


SCENARIOS = {
    "No major disruption": {
        "chokepoint": None,
        "risk_level": "Low",
        "crude_impact_pct": (0, 2),
        "gasoline_impact_pct": (0, 2),
        "description": "Baseline market conditions without a major chokepoint disruption."
    },
    "Partial Strait of Hormuz disruption": {
        "chokepoint": "Strait of Hormuz",
        "risk_level": "High",
        "crude_impact_pct": (5, 15),
        "gasoline_impact_pct": (4, 12),
        "description": "Partial disruption affecting Persian Gulf exports. Asian importers would be highly exposed."
    },
    "Full Strait of Hormuz closure": {
        "chokepoint": "Strait of Hormuz",
        "risk_level": "Critical",
        "crude_impact_pct": (20, 45),
        "gasoline_impact_pct": (15, 35),
        "description": "Severe global oil supply shock. Large impact on crude, refined products, shipping and inflation expectations."
    },
    "Suez Canal blockage": {
        "chokepoint": "Suez Canal",
        "risk_level": "Medium to high",
        "crude_impact_pct": (3, 10),
        "gasoline_impact_pct": (2, 8),
        "description": "Disruption to Europe-Asia flows. Alternative routing around the Cape of Good Hope increases time and cost."
    },
    "Bab el-Mandeb instability": {
        "chokepoint": "Bab el-Mandeb",
        "risk_level": "High",
        "crude_impact_pct": (4, 14),
        "gasoline_impact_pct": (3, 10),
        "description": "Red Sea instability affecting flows toward Suez and Europe. Shipping insurance and rerouting costs may increase."
    },
    "Strait of Malacca disruption": {
        "chokepoint": "Strait of Malacca",
        "risk_level": "High",
        "crude_impact_pct": (6, 18),
        "gasoline_impact_pct": (5, 14),
        "description": "Major disruption for Asian importers, especially China, Japan, South Korea and Southeast Asia."
    },
    "Multiple chokepoint stress scenario": {
        "chokepoint": None,
        "risk_level": "Critical",
        "crude_impact_pct": (15, 35),
        "gasoline_impact_pct": (12, 28),
        "description": "Combined stress across several global energy routes. High uncertainty and strong market sensitivity."
    }
}


MARKET_TICKERS = {
    "Brent Crude": "BZ=F",
    "WTI Crude": "CL=F",
    "RBOB Gasoline": "RB=F",
    "Heating Oil": "HO=F",
    "ExxonMobil": "XOM",
    "Chevron": "CVX",
    "Shell": "SHEL",
    "BP": "BP",
    "TotalEnergies": "TTE",
    "Equinor": "EQNR",
    "Airline ETF": "JETS"
}


# ---------------------------------------------------------
# Data functions
# ---------------------------------------------------------

@st.cache_data(ttl=1800)
def fetch_market_data(ticker, period="1y"):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=False
        )

        if data.empty:
            return pd.DataFrame()

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Adj Close" in data.columns:
            close = data["Adj Close"]
        elif "Close" in data.columns:
            close = data["Close"]
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            "date": close.index,
            "close": close.values
        })

        df = df.dropna()
        return df

    except Exception:
        return pd.DataFrame()


def get_latest_value(df):
    if df.empty:
        return None

    return float(df["close"].iloc[-1])


def get_change_pct(df):
    if df.empty or len(df) < 2:
        return None

    latest = df["close"].iloc[-1]
    previous = df["close"].iloc[-2]

    if previous == 0:
        return None

    return float((latest - previous) / previous * 100)


def format_value(value, decimals=2):
    if value is None:
        return "N/A"

    return f"{value:.{decimals}f}"


def format_change(value):
    if value is None:
        return "N/A"

    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


# ---------------------------------------------------------
# Scenario estimation
# ---------------------------------------------------------

def duration_multiplier(days):
    if days <= 3:
        return 0.6
    elif days <= 7:
        return 0.9
    elif days <= 14:
        return 1.1
    elif days <= 30:
        return 1.3
    else:
        return 1.6


def estimate_price_impact(scenario_name, duration_days, current_gasoline_price):
    scenario = SCENARIOS[scenario_name]

    crude_low, crude_high = scenario["crude_impact_pct"]
    gas_low, gas_high = scenario["gasoline_impact_pct"]

    multiplier = duration_multiplier(duration_days)

    crude_low_adj = crude_low * multiplier
    crude_high_adj = crude_high * multiplier

    gas_low_adj = gas_low * multiplier
    gas_high_adj = gas_high * multiplier

    result = {
        "crude_low_pct": crude_low_adj,
        "crude_high_pct": crude_high_adj,
        "gas_low_pct": gas_low_adj,
        "gas_high_pct": gas_high_adj,
        "gasoline_low_price": None,
        "gasoline_high_price": None
    }

    if current_gasoline_price is not None:
        result["gasoline_low_price"] = current_gasoline_price * (1 + gas_low_adj / 100)
        result["gasoline_high_price"] = current_gasoline_price * (1 + gas_high_adj / 100)

    return result


# ---------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------

def create_world_map(selected_scenario):
    fig = go.Figure()

    scenario = SCENARIOS[selected_scenario]
    highlighted_chokepoint = scenario["chokepoint"]

    # Add routes
    for route in ROUTES:
        lats = [p[0] for p in route["points"]]
        lons = [p[1] for p in route["points"]]

        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=2.5, color=route["color"]),
            name=route["name"],
            opacity=0.85
        ))

    # Add chokepoints
    for name, info in CHOKEPOINTS.items():
        is_selected = name == highlighted_chokepoint

        marker_size = 18 if is_selected else 10
        marker_color = "red" if is_selected else "orange"

        if selected_scenario == "Multiple chokepoint stress scenario":
            marker_size = 14
            marker_color = "red"

        hover_text = (
            f"<b>{name}</b><br>"
            f"Region: {info['region']}<br>"
            f"Base risk: {info['risk_base']}<br>"
            f"Flow: {info['daily_flow']}<br>"
            f"{info['description']}"
        )

        fig.add_trace(go.Scattergeo(
            lon=[info["lon"]],
            lat=[info["lat"]],
            mode="markers+text",
            text=[name],
            textposition="top center",
            marker=dict(
                size=marker_size,
                color=marker_color,
                line=dict(width=1, color="black")
            ),
            name=name,
            hovertext=hover_text,
            hoverinfo="text"
        ))

    fig.update_layout(
        title="Major Oil Supply Routes and Chokepoints",
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(230, 230, 230)",
            showocean=True,
            oceancolor="rgb(205, 225, 245)",
            showcountries=True,
            countrycolor="rgb(150, 150, 150)",
            showcoastlines=True,
            coastlinecolor="rgb(100, 100, 100)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def create_price_chart(price_data, title):
    fig = go.Figure()

    for label, df in price_data.items():
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df["close"],
                mode="lines",
                name=label
            ))

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.25)
    )

    return fig


def create_stock_chart(stock_data):
    fig = go.Figure()

    for label, df in stock_data.items():
        if not df.empty:
            normalized = df.copy()
            first_value = normalized["close"].iloc[0]

            if first_value != 0:
                normalized["indexed"] = normalized["close"] / first_value * 100

                fig.add_trace(go.Scatter(
                    x=normalized["date"],
                    y=normalized["indexed"],
                    mode="lines",
                    name=label
                ))

    fig.update_layout(
        title="Selected Energy and Transport Stocks, Indexed",
        height=400,
        xaxis_title="Date",
        yaxis_title="Index",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.25)
    )

    return fig


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("🛢️ Oil Risk Simulator")

selected_scenario = st.sidebar.selectbox(
    "Select disruption scenario",
    list(SCENARIOS.keys()),
    index=0
)

duration_days = st.sidebar.slider(
    "Estimated disruption duration, days",
    min_value=1,
    max_value=90,
    value=7
)

selected_period = st.sidebar.selectbox(
    "Market data period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard is an MVP prototype. The scenario outputs are simplified estimates and are not financial advice."
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Oil Supply Risk & Gasoline Price Impact Dashboard")

st.markdown(
    """
    This MVP visualizes major oil supply routes, strategic chokepoints, market prices, 
    selected energy-related stocks, and simple scenario-based gasoline price impact estimates.
    """
)


# ---------------------------------------------------------
# Fetch data
# ---------------------------------------------------------

with st.spinner("Loading market data from Yahoo Finance..."):
    brent_df = fetch_market_data(MARKET_TICKERS["Brent Crude"], selected_period)
    wti_df = fetch_market_data(MARKET_TICKERS["WTI Crude"], selected_period)
    gasoline_df = fetch_market_data(MARKET_TICKERS["RBOB Gasoline"], selected_period)
    heating_oil_df = fetch_market_data(MARKET_TICKERS["Heating Oil"], selected_period)

    xom_df = fetch_market_data(MARKET_TICKERS["ExxonMobil"], selected_period)
    cvx_df = fetch_market_data(MARKET_TICKERS["Chevron"], selected_period)
    shell_df = fetch_market_data(MARKET_TICKERS["Shell"], selected_period)
    bp_df = fetch_market_data(MARKET_TICKERS["BP"], selected_period)
    tte_df = fetch_market_data(MARKET_TICKERS["TotalEnergies"], selected_period)
    eqnr_df = fetch_market_data(MARKET_TICKERS["Equinor"], selected_period)
    jets_df = fetch_market_data(MARKET_TICKERS["Airline ETF"], selected_period)


# ---------------------------------------------------------
# Key metrics
# ---------------------------------------------------------

latest_brent = get_latest_value(brent_df)
latest_wti = get_latest_value(wti_df)
latest_gasoline = get_latest_value(gasoline_df)

brent_change = get_change_pct(brent_df)
wti_change = get_change_pct(wti_df)
gasoline_change = get_change_pct(gasoline_df)

scenario = SCENARIOS[selected_scenario]
impact = estimate_price_impact(
    selected_scenario,
    duration_days,
    latest_gasoline
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Brent Crude",
        value=format_value(latest_brent),
        delta=format_change(brent_change)
    )

with col2:
    st.metric(
        label="WTI Crude",
        value=format_value(latest_wti),
        delta=format_change(wti_change)
    )

with col3:
    st.metric(
        label="RBOB Gasoline",
        value=format_value(latest_gasoline),
        delta=format_change(gasoline_change)
    )

with col4:
    st.metric(
        label="Scenario Risk Level",
        value=scenario["risk_level"]
    )


# ---------------------------------------------------------
# Main layout
# ---------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Global Map",
    "Scenario Impact",
    "Market Prices",
    "Price Prediction",
    "About MVP"
])


# ---------------------------------------------------------
# Tab 1: Map
# ---------------------------------------------------------

with tab1:
    st.subheader("Global Oil Routes and Chokepoints")

    fig_map = create_world_map(selected_scenario)
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### Selected Scenario")
    st.write(scenario["description"])

    if scenario["chokepoint"]:
        cp = CHOKEPOINTS[scenario["chokepoint"]]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("**Chokepoint**")
            st.write(scenario["chokepoint"])

        with c2:
            st.write("**Region**")
            st.write(cp["region"])

        with c3:
            st.write("**Base risk**")
            st.write(cp["risk_base"])

        st.write("**Description**")
        st.write(cp["description"])
    else:
        st.write("This scenario is not tied to one single chokepoint.")


# ---------------------------------------------------------
# Tab 2: Scenario impact
# ---------------------------------------------------------

with tab2:
    st.subheader("Scenario-Based Price Impact Estimate")

    st.markdown("### Input Assumptions")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("**Scenario**")
        st.write(selected_scenario)

    with c2:
        st.write("**Duration**")
        st.write(f"{duration_days} days")

    with c3:
        st.write("**Risk level**")
        st.write(scenario["risk_level"])

    st.markdown("### Estimated Market Pressure")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            label="Estimated crude price pressure",
            value=f"{impact['crude_low_pct']:.1f}% to {impact['crude_high_pct']:.1f}%"
        )

    with c2:
        st.metric(
            label="Estimated gasoline price pressure",
            value=f"{impact['gas_low_pct']:.1f}% to {impact['gas_high_pct']:.1f}%"
        )

    st.markdown("### Estimated Gasoline Futures Range")

    if latest_gasoline is not None:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                label="Current RBOB gasoline",
                value=format_value(latest_gasoline)
            )

        with c2:
            st.metric(
                label="Estimated low case",
                value=format_value(impact["gasoline_low_price"])
            )

        with c3:
            st.metric(
                label="Estimated high case",
                value=format_value(impact["gasoline_high_price"])
            )
    else:
        st.warning("Could not fetch gasoline futures data. Price range estimate is unavailable.")

    st.markdown("### Interpretation")

    if selected_scenario == "No major disruption":
        st.success(
            "The selected scenario represents baseline conditions. Market movements are likely driven by normal supply, demand, inventory, currency and macroeconomic factors."
        )
    elif scenario["risk_level"] == "Critical":
        st.error(
            "This scenario may represent a severe global oil shock. Market impact would depend on actual supply loss, duration, spare capacity, strategic reserves, OPEC response and demand conditions."
        )
    elif "High" in scenario["risk_level"]:
        st.warning(
            "This scenario could create significant upward pressure on crude and refined product prices, especially if the disruption lasts more than a few days."
        )
    else:
        st.info(
            "This scenario may create moderate price pressure, mainly through rerouting costs, shipping delays and uncertainty premiums."
        )

    st.caption(
        "Note: These estimates are scenario heuristics, not a predictive financial model."
    )


# ---------------------------------------------------------
# Tab 3: Market prices
# ---------------------------------------------------------

with tab3:
    st.subheader("Oil, Gasoline and Related Market Data")

    commodity_data = {
        "Brent Crude": brent_df,
        "WTI Crude": wti_df,
        "RBOB Gasoline": gasoline_df,
        "Heating Oil": heating_oil_df
    }

    fig_commodities = create_price_chart(
        commodity_data,
        "Crude Oil and Refined Product Futures"
    )

    st.plotly_chart(fig_commodities, use_container_width=True)

    stock_data = {
        "ExxonMobil": xom_df,
        "Chevron": cvx_df,
        "Shell": shell_df,
        "BP": bp_df,
        "TotalEnergies": tte_df,
        "Equinor": eqnr_df,
        "Airline ETF": jets_df
    }

    fig_stocks = create_stock_chart(stock_data)
    st.plotly_chart(fig_stocks, use_container_width=True)

    st.markdown("### Latest Market Snapshot")

    rows = []

    for label, ticker in MARKET_TICKERS.items():
        df = fetch_market_data(ticker, selected_period)
        latest = get_latest_value(df)
        change = get_change_pct(df)

        rows.append({
            "Asset": label,
            "Ticker": ticker,
            "Latest": latest,
            "Daily change": change
        })

    snapshot_df = pd.DataFrame(rows)
    snapshot_df["Latest"] = snapshot_df["Latest"].apply(
        lambda x: None if x is None else round(x, 2)
    )
    snapshot_df["Daily change"] = snapshot_df["Daily change"].apply(
        lambda x: None if x is None else f"{x:+.2f}%"
    )

    st.dataframe(snapshot_df, use_container_width=True)


# ---------------------------------------------------------
# Tab 4: Price Prediction
# ---------------------------------------------------------

with tab4:
    st.subheader("Linear Regression Price Prediction")

    pred_asset = st.selectbox(
        "Select asset to predict",
        list(MARKET_TICKERS.keys())
    )

    col_obs, col_hor = st.columns(2)
    with col_obs:
        obs_window = st.slider(
            "Observation window (days to train on)",
            min_value=7,
            max_value=730,
            value=90,
            help="Use a short window to capture recent trends (e.g. during a disruption)"
        )
    with col_hor:
        pred_horizon = st.slider(
            "Prediction horizon (months)",
            min_value=1,
            max_value=24,
            value=3
        )

    pred_df = fetch_market_data(MARKET_TICKERS[pred_asset], "5y")

    if not pred_df.empty:
        pred_df = pred_df.copy()
        # Trim to observation window
        train_df = pred_df.tail(obs_window).reset_index(drop=True)
        train_df["days"] = (train_df["date"] - train_df["date"].iloc[0]).dt.days

        X = train_df[["days"]].values
        y = train_df["close"].values

        model = LinearRegression().fit(X, y)

        last_day = train_df["days"].iloc[-1]
        future_days = np.arange(
            last_day + 1,
            last_day + pred_horizon * 30 + 1
        ).reshape(-1, 1)
        future_prices = model.predict(future_days)

        last_date = train_df["date"].iloc[-1]
        future_dates = [last_date + timedelta(days=int(d - last_day)) for d in future_days.flatten()]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=train_df["date"], y=train_df["close"],
            mode="lines", name="Observation window"
        ))
        fig_pred.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            mode="lines", name="Prediction",
            line=dict(dash="dash", color="red")
        ))
        fig_pred.update_layout(
            title=f"{pred_asset} — {pred_horizon}-Month Forecast (trained on last {obs_window} days)",
            xaxis_title="Date", yaxis_title="Price",
            height=450, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="Current price", value=f"{y[-1]:.2f}")
        with c2:
            st.metric(label=f"Predicted in {pred_horizon}mo", value=f"{future_prices[-1]:.2f}")
        with c3:
            change = (future_prices[-1] - y[-1]) / y[-1] * 100
            st.metric(label="Projected change", value=f"{change:+.1f}%")

        st.caption(
            "Tip: Use a short observation window (e.g. 7–30 days) to project recent disruption trends forward. "
            "Use a longer window for a broader market trend. This is not financial advice."
        )
    else:
        st.warning("Could not fetch data for prediction.")


# ---------------------------------------------------------
# Tab 5: About MVP
# ---------------------------------------------------------

with tab5:
    st.subheader("About This MVP")

    st.markdown(
        """
        This prototype implements the first version of an oil supply risk dashboard.

        Included features:

        - Interactive map of major oil supply routes
        - Strategic chokepoint markers
        - Scenario selector for disruption analysis
        - Live commodity and stock data from Yahoo Finance
        - Estimated crude and gasoline price pressure
        - Basic market snapshot table

        Possible future improvements:

        - Add EIA API data for real gasoline retail prices and inventory levels
        - Add GDELT or news API integration for live geopolitical risk monitoring
        - Add AIS/shipping data for vessel density around chokepoints
        - Add historical event database
        - Add machine learning model trained on oil, inventory, geopolitical and macro data
        - Add region-specific gasoline price estimates
        - Add alerting when risk exceeds a threshold
        """
    )

    st.warning(
        "Disclaimer: This application is for educational and analytical purposes only. "
        "It is not financial advice, investment advice, or a guarantee of future market movements."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")
st.caption(
    f"Last refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC | "
    "Data source: Yahoo Finance via yfinance"
)