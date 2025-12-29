# app.py
# Masszőr bevétel előrejelzés – Monte Carlo (havi bontás)
# - Szezonális új ügyfelek (hónaponként min/mode/max)
# - Lemorzsolódás (churn)
# - Ismétlődés (VPC keverék)
# - Kapacitáskorlát (munkanapok/hó, napi órák, kezelés+csere perc)
# - Lemondás/no-show, fáradás, kiesés
# - Ár/alkalom (triangular)
# - Marketing: fix/hó + CAC (triangular) * új ügyfelek
# - FIX KÖLTSÉG: egyetlen átlagos havi fix költség (minden hónapban levonva)
# - Profit + Cash-flow + Break-even (P50 cash)
#
# Futtatás:
#   pip install streamlit numpy pandas
#   streamlit run app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ----------------------------
# Helpers
# ----------------------------
def triangular_draw(rng: np.random.Generator, left: float, mode: float, right: float, size):
    left_ = min(left, mode, right)
    right_ = max(left_, mode, right)
    mode_ = min(max(mode, left_), right_)
    return rng.triangular(left_, mode_, right_, size=size)

def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))

def format_huf(x: float) -> str:
    return f"{x:,.0f} Ft".replace(",", " ")

def validate_triangular(min_v, mode_v, max_v, name: str) -> bool:
    ok = (min_v <= mode_v <= max_v)
    if not ok:
        st.error(f"Hibás {name} paraméter: elvárt min ≤ mode ≤ max, de most: {min_v}, {mode_v}, {max_v}")
    return ok

# ----------------------------
# Simulation core
# ----------------------------
def simulate_year(
    n_sims: int,
    seed: int,

    # Capacity
    working_days_per_month: np.ndarray,  # len 12
    hours_per_day: int,
    service_min: int,
    turnover_min: int,

    # Client base dynamics
    starting_active_clients: int,
    churn_min: float, churn_mode: float, churn_max: float,

    # Seasonal new clients (month-wise triangular parameters)
    new_min_by_month: np.ndarray,
    new_mode_by_month: np.ndarray,
    new_max_by_month: np.ndarray,

    # Visits per active client per month mixture
    vpc_p1: float, vpc_p15: float, vpc_p22: float,
    vpc_1: float, vpc_15: float, vpc_22: float,

    # Operational uncertainty (monthly)
    u_min: float, u_mode: float, u_max: float,      # utilization
    c_min: float, c_mode: float, c_max: float,      # cancellations
    f_min: float, f_mode: float, f_max: float,      # fatigue loss
    s_min: float, s_mode: float, s_max: float,      # absence loss

    # Price per completed visit (monthly)
    p_min: int, p_mode: int, p_max: int,

    # Marketing costs
    mkt_fixed_by_month: np.ndarray,  # len 12, Ft
    cac_min: int, cac_mode: int, cac_max: int,       # Ft per new client

    # FIX COST (single average monthly)
    fixed_cost_monthly: int,

    # Cash
    starting_cash: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    minutes_per_client = int(service_min + turnover_min)
    if minutes_per_client <= 0:
        raise ValueError("A kezelés+csere idő nem lehet 0 vagy negatív.")

    max_clients_per_day = int(np.floor((hours_per_day * 60) / minutes_per_client))
    max_clients_per_day = max(max_clients_per_day, 0)

    capacity_visits = working_days_per_month.astype(float) * max_clients_per_day  # len 12

    # Monthly random matrices
    U = triangular_draw(rng, u_min, u_mode, u_max, size=(n_sims, 12))
    C = triangular_draw(rng, c_min, c_mode, c_max, size=(n_sims, 12))
    F = triangular_draw(rng, f_min, f_mode, f_max, size=(n_sims, 12))
    S = triangular_draw(rng, s_min, s_mode, s_max, size=(n_sims, 12))
    P = triangular_draw(rng, float(p_min), float(p_mode), float(p_max), size=(n_sims, 12))
    CHURN = triangular_draw(rng, churn_min, churn_mode, churn_max, size=(n_sims, 12))
    CAC = triangular_draw(rng, float(cac_min), float(cac_mode), float(cac_max), size=(n_sims, 12))

    # Seasonal NEW clients month-by-month
    NEW = np.zeros((n_sims, 12), dtype=int)
    for m in range(12):
        new_draw = triangular_draw(
            rng,
            float(new_min_by_month[m]),
            float(new_mode_by_month[m]),
            float(new_max_by_month[m]),
            size=n_sims
        )
        NEW[:, m] = np.round(new_draw).astype(int)
        NEW[:, m] = np.clip(NEW[:, m], 0, None)

    # Visits-per-client mixture per (sim, month)
    probs = np.array([vpc_p1, vpc_p15, vpc_p22], dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.array([1.0, 0.0, 0.0])
    choices = np.array([vpc_1, vpc_15, vpc_22], dtype=float)
    VPC = rng.choice(choices, size=(n_sims, 12), p=probs)

    # State + results arrays
    active = np.zeros((n_sims, 12), dtype=float)
    active[:, 0] = float(starting_active_clients)

    completed_visits = np.zeros((n_sims, 12), dtype=float)
    revenue = np.zeros((n_sims, 12), dtype=float)
    marketing_cost = np.zeros((n_sims, 12), dtype=float)

    total_cost = np.zeros((n_sims, 12), dtype=float)
    profit = np.zeros((n_sims, 12), dtype=float)  # after marketing + fixed
    cash = np.zeros((n_sims, 12), dtype=float)    # cumulative cash

    for m in range(12):
        if m > 0:
            active[:, m] = active[:, m - 1] * (1.0 - CHURN[:, m]) + NEW[:, m]
            active[:, m] = np.clip(active[:, m], 0, 1_000_000)

        demand_visits = active[:, m] * VPC[:, m]

        available_visits = (
            capacity_visits[m]
            * U[:, m]
            * (1.0 - F[:, m])
            * (1.0 - S[:, m])
        )

        booked_visits = np.minimum(demand_visits, available_visits)
        completed_visits[:, m] = booked_visits * (1.0 - C[:, m])

        revenue[:, m] = completed_visits[:, m] * P[:, m]

        marketing_cost[:, m] = float(mkt_fixed_by_month[m]) + (CAC[:, m] * NEW[:, m])

        total_cost[:, m] = marketing_cost[:, m] + float(fixed_cost_monthly)
        profit[:, m] = revenue[:, m] - total_cost[:, m]

        cash[:, m] = float(starting_cash) + profit[:, : m + 1].sum(axis=1)

    # Monthly stats
    rows = []
    for m in range(12):
        rows.append({
            "month": MONTHS[m],

            "new_clients_p10": pct(NEW[:, m], 10),
            "new_clients_p50": pct(NEW[:, m], 50),
            "new_clients_p90": pct(NEW[:, m], 90),

            "active_clients_p10": pct(active[:, m], 10),
            "active_clients_p50": pct(active[:, m], 50),
            "active_clients_p90": pct(active[:, m], 90),

            "completed_visits_p10": pct(completed_visits[:, m], 10),
            "completed_visits_p50": pct(completed_visits[:, m], 50),
            "completed_visits_p90": pct(completed_visits[:, m], 90),

            "revenue_p10_huf": pct(revenue[:, m], 10),
            "revenue_p50_huf": pct(revenue[:, m], 50),
            "revenue_p90_huf": pct(revenue[:, m], 90),

            "marketing_p10_huf": pct(marketing_cost[:, m], 10),
            "marketing_p50_huf": pct(marketing_cost[:, m], 50),
            "marketing_p90_huf": pct(marketing_cost[:, m], 90),

            "fixed_cost_huf": float(fixed_cost_monthly),

            "total_cost_p10_huf": pct(total_cost[:, m], 10),
            "total_cost_p50_huf": pct(total_cost[:, m], 50),
            "total_cost_p90_huf": pct(total_cost[:, m], 90),

            "profit_p10_huf": pct(profit[:, m], 10),
            "profit_p50_huf": pct(profit[:, m], 50),
            "profit_p90_huf": pct(profit[:, m], 90),

            "cash_p10_huf": pct(cash[:, m], 10),
            "cash_p50_huf": pct(cash[:, m], 50),
            "cash_p90_huf": pct(cash[:, m], 90),
        })

    df = pd.DataFrame(rows)

    # Annual totals (by sim)
    annual_rev = revenue.sum(axis=1)
    annual_mkt = marketing_cost.sum(axis=1)
    annual_total_cost = total_cost.sum(axis=1)
    annual_profit = profit.sum(axis=1)
    annual_cash_end = cash[:, -1]

    total = pd.DataFrame([{
        "month": "TOTAL",

        "new_clients_p10": pct(NEW.sum(axis=1), 10),
        "new_clients_p50": pct(NEW.sum(axis=1), 50),
        "new_clients_p90": pct(NEW.sum(axis=1), 90),

        "active_clients_p10": pct(active[:, -1], 10),
        "active_clients_p50": pct(active[:, -1], 50),
        "active_clients_p90": pct(active[:, -1], 90),

        "completed_visits_p10": pct(completed_visits.sum(axis=1), 10),
        "completed_visits_p50": pct(completed_visits.sum(axis=1), 50),
        "completed_visits_p90": pct(completed_visits.sum(axis=1), 90),

        "revenue_p10_huf": pct(annual_rev, 10),
        "revenue_p50_huf": pct(annual_rev, 50),
        "revenue_p90_huf": pct(annual_rev, 90),

        "marketing_p10_huf": pct(annual_mkt, 10),
        "marketing_p50_huf": pct(annual_mkt, 50),
        "marketing_p90_huf": pct(annual_mkt, 90),

        "fixed_cost_huf": float(fixed_cost_monthly) * 12.0,

        "total_cost_p10_huf": pct(annual_total_cost, 10),
        "total_cost_p50_huf": pct(annual_total_cost, 50),
        "total_cost_p90_huf": pct(annual_total_cost, 90),

        "profit_p10_huf": pct(annual_profit, 10),
        "profit_p50_huf": pct(annual_profit, 50),
        "profit_p90_huf": pct(annual_profit, 90),

        "cash_p10_huf": pct(annual_cash_end, 10),
        "cash_p50_huf": pct(annual_cash_end, 50),
        "cash_p90_huf": pct(annual_cash_end, 90),
    }])

    return pd.concat([df, total], ignore_index=True)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Masszőr bevétel – Monte Carlo", layout="wide")
st.title("Masszőr bevétel előrejelzés – Monte Carlo")
st.caption("Szezonális új ügyfelek • Marketing (fix + CAC) • Fix havi költség • Cash-flow • Break-even • P10/P50/P90")

# Defaults
default_work_days = [21, 20, 23, 21, 22, 21, 23, 21, 22, 23, 20, 21]
default_new = {
    "min":  [8,  8, 10, 12, 12, 10,  8,  8, 10, 12, 12, 10],
    "mode": [15,15,18, 20, 20, 18, 15, 15, 18, 20, 20, 18],
    "max":  [25,25,30, 35, 35, 30, 25, 25, 30, 35, 35, 30],
}
default_mkt_fixed = [50_000, 50_000, 60_000, 60_000, 60_000, 50_000, 40_000, 40_000, 50_000, 60_000, 60_000, 60_000]

with st.sidebar:
    st.header("Beállítások")

    with st.form("params_form", border=False):
        st.subheader("Szimuláció")
        n_sims = st.number_input("Szimulációk száma", min_value=5_000, max_value=300_000, value=80_000, step=5_000)
        seed = 123

        st.divider()
        st.subheader("Fix költség")
        fixed_cost_monthly = st.number_input(
            "Átlagos havi fix költség (Ft) – minden hónapban levonva",
            min_value=0, max_value=50_000_000, value=300_000, step=10_000
        )

        st.divider()
        st.subheader("Cash-flow indulás")
        starting_cash = st.number_input(
            "Induló pénzkészlet (Ft) – lehet negatív is",
            min_value=-50_000_000, max_value=50_000_000, value=0, step=50_000
        )

        st.divider()
        st.subheader("Kapacitás")
        hours_per_day = st.slider("Napi munkaóra", 4, 12, 8, 1)
        service_min = st.slider("Kezelés (perc)", 30, 120, 60, 5)
        turnover_min = st.slider("Csere/admin (perc)", 0, 30, 10, 1)

        st.caption("Munkanapok/hó (átírható)")
        wd = []
        cols = st.columns(3)
        for i in range(12):
            with cols[i % 3]:
                wd.append(st.number_input(f"{MONTHS[i]} munkanap", min_value=0, max_value=31, value=default_work_days[i], step=1))
        working_days_per_month = np.array(wd, dtype=int)

        st.divider()
        st.subheader("Ügyfélbázis")
        starting_active_clients = st.number_input("Induló aktív ügyfelek", min_value=0, max_value=20_000, value=0, step=1)

        st.caption("Lemorzsolódás / hó (Triangular)")
        churn_min = st.slider("CHURN min", 0.0, 0.6, 0.08, 0.01)
        churn_mode = st.slider("CHURN mode", 0.0, 0.6, 0.12, 0.01)
        churn_max = st.slider("CHURN max", 0.0, 0.6, 0.20, 0.01)

        st.divider()
        st.subheader("Szezonális új ügyfelek / hó (Triangular)")
        st.caption("Hónaponként külön min/mode/max (egészre kerekítve).")

        new_min_by_month, new_mode_by_month, new_max_by_month = [], [], []
        grid = st.columns(4)
        with grid[0]:
            st.markdown("**Hónap**")
            for m in MONTHS:
                st.write(m)
        with grid[1]:
            st.markdown("**min**")
            for i in range(12):
                new_min_by_month.append(st.number_input(f"new_min_{i}", label_visibility="collapsed",
                                                        min_value=0, max_value=500, value=default_new["min"][i], step=1))
        with grid[2]:
            st.markdown("**mode**")
            for i in range(12):
                new_mode_by_month.append(st.number_input(f"new_mode_{i}", label_visibility="collapsed",
                                                         min_value=0, max_value=500, value=default_new["mode"][i], step=1))
        with grid[3]:
            st.markdown("**max**")
            for i in range(12):
                new_max_by_month.append(st.number_input(f"new_max_{i}", label_visibility="collapsed",
                                                        min_value=0, max_value=500, value=default_new["max"][i], step=1))

        new_min_by_month = np.array(new_min_by_month, dtype=int)
        new_mode_by_month = np.array(new_mode_by_month, dtype=int)
        new_max_by_month = np.array(new_max_by_month, dtype=int)

        st.divider()
        st.subheader("Marketing")
        st.caption("Fix marketing költség / hó (Ft)")
        mkt_fixed = []
        cols2 = st.columns(3)
        for i in range(12):
            with cols2[i % 3]:
                mkt_fixed.append(st.number_input(f"{MONTHS[i]} fix mkt", min_value=0, max_value=10_000_000,
                                                 value=default_mkt_fixed[i], step=5_000))
        mkt_fixed_by_month = np.array(mkt_fixed, dtype=int)

        st.caption("CAC / új ügyfél (Triangular, Ft)")
        cac_min = st.number_input("CAC min", min_value=0, max_value=500_000, value=2_000, step=500)
        cac_mode = st.number_input("CAC mode", min_value=0, max_value=500_000, value=4_000, step=500)
        cac_max = st.number_input("CAC max", min_value=0, max_value=500_000, value=8_000, step=500)

        st.divider()
        st.subheader("Ismétlődés (alkalom / aktív ügyfél / hó)")
        st.caption("Keverék arányok + értékek (átírható).")
        vpc_p1 = st.slider("Arány: 1.0x", 0.0, 1.0, 0.70, 0.01)
        vpc_p15 = st.slider("Arány: 1.5x", 0.0, 1.0, 0.25, 0.01)
        vpc_p22 = st.slider("Arány: 2.2x", 0.0, 1.0, 0.05, 0.01)
        vpc_1 = st.number_input("Érték: 1.0x", value=1.0, step=0.1)
        vpc_15 = st.number_input("Érték: 1.5x", value=1.5, step=0.1)
        vpc_22 = st.number_input("Érték: 2.2x", value=2.2, step=0.1)

        st.divider()
        st.subheader("Működési bizonytalanságok (Triangular)")
        st.caption("Kihasználtság (U)")
        u_min = st.slider("U min", 0.0, 1.0, 0.55, 0.01)
        u_mode = st.slider("U mode", 0.0, 1.0, 0.70, 0.01)
        u_max = st.slider("U max", 0.0, 1.0, 0.85, 0.01)

        st.caption("Lemondás/no-show (C)")
        c_min = st.slider("C min", 0.0, 0.6, 0.05, 0.01)
        c_mode = st.slider("C mode", 0.0, 0.6, 0.10, 0.01)
        c_max = st.slider("C max", 0.0, 0.6, 0.18, 0.01)

        st.caption("Fáradás miatti kapacitásvesztés (F)")
        f_min = st.slider("F min", 0.0, 0.5, 0.00, 0.01)
        f_mode = st.slider("F mode", 0.0, 0.5, 0.05, 0.01)
        f_max = st.slider("F max", 0.0, 0.5, 0.10, 0.01)

        st.caption("Kiesés: szabadság/betegség/admin (S)")
        s_min = st.slider("S min", 0.0, 0.5, 0.03, 0.01)
        s_mode = st.slider("S mode", 0.0, 0.5, 0.08, 0.01)
        s_max = st.slider("S max", 0.0, 0.5, 0.15, 0.01)

        st.divider()
        st.subheader("Ár/alkalom (Triangular, Ft)")
        p_min = st.number_input("P min", min_value=0, max_value=200_000, value=8_000, step=500)
        p_mode = st.number_input("P mode", min_value=0, max_value=200_000, value=10_000, step=500)
        p_max = st.number_input("P max", min_value=0, max_value=200_000, value=14_000, step=500)

        submitted = st.form_submit_button("Szimuláció futtatása", type="primary")

# ----------------------------
# Run + store
# ----------------------------
if submitted:
    ok = True
    ok &= validate_triangular(u_min, u_mode, u_max, "U")
    ok &= validate_triangular(c_min, c_mode, c_max, "C")
    ok &= validate_triangular(f_min, f_mode, f_max, "F")
    ok &= validate_triangular(s_min, s_mode, s_max, "S")
    ok &= validate_triangular(churn_min, churn_mode, churn_max, "CHURN")
    ok &= validate_triangular(p_min, p_mode, p_max, "P (ár)")
    ok &= validate_triangular(cac_min, cac_mode, cac_max, "CAC")
    for i in range(12):
        ok &= validate_triangular(new_min_by_month[i], new_mode_by_month[i], new_max_by_month[i], f"NEW ({MONTHS[i]})")

    if ok:
        with st.spinner("Szimuláció futtatása..."):
            df = simulate_year(
                n_sims=int(n_sims),
                seed=int(seed),

                working_days_per_month=working_days_per_month,
                hours_per_day=int(hours_per_day),
                service_min=int(service_min),
                turnover_min=int(turnover_min),

                starting_active_clients=int(starting_active_clients),
                churn_min=float(churn_min), churn_mode=float(churn_mode), churn_max=float(churn_max),

                new_min_by_month=new_min_by_month,
                new_mode_by_month=new_mode_by_month,
                new_max_by_month=new_max_by_month,

                vpc_p1=float(vpc_p1), vpc_p15=float(vpc_p15), vpc_p22=float(vpc_p22),
                vpc_1=float(vpc_1), vpc_15=float(vpc_15), vpc_22=float(vpc_22),

                u_min=float(u_min), u_mode=float(u_mode), u_max=float(u_max),
                c_min=float(c_min), c_mode=float(c_mode), c_max=float(c_max),
                f_min=float(f_min), f_mode=float(f_mode), f_max=float(f_max),
                s_min=float(s_min), s_mode=float(s_mode), s_max=float(s_max),

                p_min=int(p_min), p_mode=int(p_mode), p_max=int(p_max),

                mkt_fixed_by_month=mkt_fixed_by_month,
                cac_min=int(cac_min), cac_mode=int(cac_mode), cac_max=int(cac_max),

                fixed_cost_monthly=int(fixed_cost_monthly),
                starting_cash=int(starting_cash),
            )
        st.session_state["df"] = df

# ----------------------------
# Display
# ----------------------------
if "df" not in st.session_state:
    st.info("Állítsd be a paramétereket bal oldalt, majd kattints: Szimuláció futtatása.")
else:
    df = st.session_state["df"].copy()

    # month order for plots
    dfm = df[df["month"] != "TOTAL"].copy()
    dfm["month"] = pd.Categorical(dfm["month"], categories=MONTHS, ordered=True)
    dfm = dfm.sort_values("month").set_index("month")

    total = df[df["month"] == "TOTAL"].iloc[0]

    # Break-even month based on P50 cash (first month where >= 0)
    cash_p50 = dfm["cash_p50_huf"].to_numpy()
    idx = np.where(cash_p50 >= 0)[0]
    break_even = MONTHS[int(idx[0])] if len(idx) else "Nincs (év végéig sem)"

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Éves bevétel (P50)", format_huf(total["revenue_p50_huf"]))
    with k2:
        st.metric("Éves összköltség (P50)", format_huf(total["total_cost_p50_huf"]))
    with k3:
        st.metric("Éves profit (P50)", format_huf(total["profit_p50_huf"]))
    with k4:
        st.metric("Profit (P10–P90)", f"{format_huf(total['profit_p10_huf'])} – {format_huf(total['profit_p90_huf'])}")
    with k5:
        st.metric("Break-even (P50 cash)", break_even)

    st.divider()

    minutes_per_client = int(service_min + turnover_min)
    max_clients_per_day = int(np.floor((hours_per_day * 60) / minutes_per_client)) if minutes_per_client > 0 else 0
    st.caption(
        f"Kapacitás: {hours_per_day} óra/nap • {minutes_per_client} perc/ügyfél → "
        f"{max_clients_per_day} ügyfél/nap max • éves elméleti max alkalom ≈ {(working_days_per_month.sum() * max_clients_per_day):,}"
    )

    st.subheader("Havi eredmények (P10 / P50 / P90)")
    show = df.copy()
    money_cols = [c for c in show.columns if c.endswith("_huf")]
    for c in money_cols:
        show[c] = np.round(show[c]).astype(int)

    # keep month order + TOTAL bottom
    show_m = show[show["month"] != "TOTAL"].copy()
    show_m["month"] = pd.Categorical(show_m["month"], categories=MONTHS, ordered=True)
    show_m = show_m.sort_values("month")
    show_t = show[show["month"] == "TOTAL"]
    show = pd.concat([show_m, show_t], ignore_index=True)

    st.dataframe(show, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Grafikonok")

    st.markdown("Bevétel (P10 / P50 / P90)")
    st.line_chart(dfm[["revenue_p10_huf", "revenue_p50_huf", "revenue_p90_huf"]], height=260)

    st.markdown("Profit (marketing + fix költség után) (P10 / P50 / P90)")
    st.line_chart(dfm[["profit_p10_huf", "profit_p50_huf", "profit_p90_huf"]], height=260)

    st.markdown("Kumulált cash-flow / pénzegyenleg (P10 / P50 / P90)")
    st.line_chart(dfm[["cash_p10_huf", "cash_p50_huf", "cash_p90_huf"]], height=260)

    st.markdown("Új ügyfelek és aktív bázis (P50)")
    st.line_chart(dfm[["new_clients_p50", "active_clients_p50"]], height=260)

    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Eredmények letöltése CSV-ben",
        data=csv,
        file_name="masszor_montecarlo_cashflow.csv",
        mime="text/csv"
    )
