"""Nurtec copay fraud-signal pipeline -- shared module.

Imported by BOTH notebooks (01_train_models, 02_score_lookalikes) so the
deterministic rule engine and feature logic can never drift between
training and scoring. The ML pieces are split into fit_* / apply_* pairs.
"""


from __future__ import annotations
import json
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# COPAY_PATH   = "COPAY_NURTEC_CLEAN.csv"
COPAY_PATH   = "COPAY_EUCRISA_CLEAN.csv"
RISRX_PATH   = "RisRx_List.csv"
OUTPUT_XLSX  = "Eucrisay_Signal_Scorecard.xlsx"
PHARMACY_TYPE_PATH = "Pharmacy_Type_Crosswalk.csv"   # master-type crosswalk (exact+fuzzy)
CARDS_JSON_OUT     = "Nurtec_Signal_Cards.json"      # dashboard signal cards
PHARM_JSON_OUT     = "Nurtec_Pharmacy_Records.json"  # dashboard pharmacy records
SIGNAL_OUTPUT_CSV  = "Nurtec_Signal_Output.xlsx"     # tidy one-row-per-entity signal table (Excel)
SIGNAL_TABLE_DAYS  = 365     # recency window for the signal table (1 year)
RECENT_DAYS        = 90      # only export signals/pharmacies active within this many days
AS_OF              = pd.Timestamp.today().normalize()  # "today" for detectedDaysAgo (run date = 2026-06-17)
TYPE_SHORT = {"Chain Pharmacy": "Chain", "Independent Pharmacy": "Independent",
              "Franchise Pharmacy": "Franchise", "Alternate Dispensing Site": "Alternate",
              "Government Pharmacy": "Government"}

# Cost-per-unit vs chain baseline (Signal 1)
CHAIN_MIN_STORES   = 5      # FALLBACK only: brand at >= this many stores = "chain" when type is unknown
CHAIN_BASELINE_TYPES = {"Chain Pharmacy", "Franchise Pharmacy"}  # master types that form the baseline
CPU_CHAIN_DEV_PCT  = 0.25   # flag non-chain pharmacy if CPU exceeds chain baseline by >25%
S1_MIN_FILLS       = 3      # min positive fills before a pharmacy's CPU is trusted

# Cost-per-claim at/near program max (Signal 2)
MAX_BENEFIT_PCTILE = 0.995  # robust proxy for the "program max allowable" per claim
NEAR_MAX_BAND_PCT  = 0.95   # "near max" band = >= 95% of program max  (calibrate)
MAX_CLAIM_SHARE    = 0.80   # flag if >= 80% of a pharmacy's claims fall in the band (calibrate)
S2_MIN_CLAIMS      = 5      # min claims before the share is trusted

# Cost-per-unit (Signal 3)
CPU_MIN_FILLS_PER_WINDOW = 3      # min fills in a window before CPU is trusted
CPU_DROP_MIN_PCT         = 0.15   # CPU must drop >=15% PRE->DURING to be a candidate
CPU_REBOUND_PCT          = 0.50   # POST recovers >=50% of the drop -> flag rebound

# Quantity change (Signal 4)
QTY_TRAILING_PERIODS     = 12     # trailing months used for the rolling baseline
QTY_MIN_HISTORY          = 4      # min prior periods before z-scores are trusted
QTY_Z_THRESHOLD          = 2.0    # |z| of avg-qty/dispense vs trailing baseline
QTY_ABS_JUMP_UNITS       = 42.0   # absolute MoM jump in avg qty/dispense

# High HCP utilization at flagged pharmacies (Signal 5 -- amplifier)
HCP_MIN_CLAIMS           = 5      # min claims for an HCP to be evaluated (low-vol volatility)
HCP_CONCENTRATION_PCT    = 0.50   # >=50% of an HCP's claims route to the flagged pharmacy
HCP_VOLUME_MULT          = 3.0    # OR volume there > 3x the HCP's average per-pharmacy volume

# Popup / dormant pharmacy (Signal 7)
DORMANCY_MONTHS          = 6      # >= this many consecutive zero-claim months
REACTIVATION_WINDOW_DAYS = 28     # "first 2-4 weeks" volume window after restart
HIGH_VOLUME_FLOOR        = 10     # min absolute claims to ever be "high volume"
HIGH_VOLUME_PCTILE       = 0.90   # population monthly-volume pctile for "high"
NEW_PHARMACY_HIGH_MULT   = 1.0    # brand-new pharmacy window vs the high baseline

# Recency / remediation layer
RECENCY_ACTIVE_MONTHS   = 2     # last signal activity within 2 months -> "Active"
RECENCY_RECENT_MONTHS   = 5     # within 5 months -> "Recent"
RECENCY_AGING_MONTHS    = 11    # within 11 months -> "Aging"; older -> "Stale / likely remediated"
RECENCY_HALFLIFE_MONTHS = 6     # half-life (months) for the recency weight in priority_score

# Composite anomaly model
ISO_CONTAMINATION        = 0.05   # expected fraction of anomalous pharmacies
RANDOM_STATE             = 42

# Supervised classifier (label = triggered any signal)
CLF_TEST_SIZE            = 0.30   # hold-out fraction for evaluation
CLF_FEATURES = [                  # behavioral PROFILE features only -- NOT the
    "total_claims", "total_units", "total_benefit",   # thresholded trigger
    "avg_qty", "std_qty", "avg_cpu", "std_cpu",        # metrics, to avoid the
    "avg_cpu_pct_wac", "n_months", "n_states",         # label being trivially
    "reversal_rate", "claims_per_month",               # reconstructable (leakage)
]

NURTEC_NDC = 72618300002          # single Nurtec 75MG ODT NDC in this feed


def _norm_name(s: str) -> str:
    """Normalize a pharmacy name for joining (uppercase, strip punctuation)."""
    s = str(s).upper().strip()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_copay(path: str = COPAY_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # --- dates ---
    df["fill_date"] = pd.to_datetime(df["DATE_OF_FILL"], errors="coerce")
    df = df.dropna(subset=["fill_date"])
    df["month"] = df["fill_date"].dt.to_period("M").dt.to_timestamp()

    # --- pharmacy key (NPI is null -> use normalized name) ---
    df["pharmacy_key"] = df["PHARMACY_NAME"].map(_norm_name)

    # --- numeric hygiene ---
    df["QTY_DISPENSED"] = pd.to_numeric(df["QTY_DISPENSED"], errors="coerce")
    df["BENEFIT_PAID"]  = pd.to_numeric(df["BENEFIT_PAID"], errors="coerce")
    df["WAC"]           = pd.to_numeric(df["WAC"], errors="coerce")

    # --- reversal flag (negative qty/benefit) ---
    df["is_reversal"] = (
        df["CLAIM_TYPE_FIXED"].astype(str).str.contains("Revers", case=False, na=False)
        | (df["QTY_DISPENSED"] < 0)
    )

    # --- per-claim cost-per-unit (only meaningful on positive redemptions) ---
    pos = df["QTY_DISPENSED"] > 0
    df["cpu"] = np.where(pos, df["BENEFIT_PAID"] / df["QTY_DISPENSED"], np.nan)
    # benefit as % of WAC (WAC is per-unit; compare benefit/unit to it)
    df["cpu_pct_wac"] = np.where(df["WAC"] > 0, df["cpu"] / df["WAC"], np.nan)

    return df


def load_risrx(path: str = RISRX_PATH, data_min=None, data_max=None) -> pd.DataFrame:
    ris = pd.read_csv(path, low_memory=False)
    ris["pharmacy_key"] = ris["Pharmacy Name"].map(_norm_name)

    def parse_start(v):
        if str(v).strip().lower() == "inception":
            return data_min
        return pd.to_datetime(v, errors="coerce")

    def parse_end(v):
        if str(v).strip().upper() in ("TBD", "ACTIVE", ""):
            return data_max          # still monitored -> no POST window yet
        return pd.to_datetime(v, errors="coerce")

    ris["start_dt"] = ris["Start_Date"].map(parse_start)
    ris["end_dt"]   = ris["End_Date"].map(parse_end)
    ris["still_monitored"] = ris["End_Date"].astype(str).str.upper().str.strip() == "TBD"
    return ris


def _chain_brand(key: str) -> str:
    """Collapse a normalized pharmacy name to a brand by stripping store numbers
    and legal suffixes (FALLBACK chain inference when master type is unknown)."""
    toks = [t for t in str(key).split() if not t.isdigit()]
    drop = {"LLC", "INC", "LP", "LTD", "CORP", "CO", "PLLC", "PC"}
    toks = [t for t in toks if t not in drop]
    return " ".join(toks).strip() or str(key)


def attach_pharmacy_type(df: pd.DataFrame,
                         crosswalk_path: str = PHARMACY_TYPE_PATH) -> pd.DataFrame:
    """Attach the master pharmacy type (DISPENSER_CLAS_NM) onto each claim via the
    4-key crosswalk. Memory-light: builds a composite-key lookup and maps in place
    (no full-frame merge copy, which matters on the 893K x 51 feed). Adds
    PHARMACY_TYPE and type_confidence; leaves NA where the crosswalk is missing."""
    keys = ["PHARMACY_NAME", "PHARMA_CITY", "PHARMA_ST", "PHARMA_ZIP"]
    try:
        cw = pd.read_csv(crosswalk_path, low_memory=False)
    except FileNotFoundError:
        df["PHARMACY_TYPE"] = pd.NA
        df["type_confidence"] = pd.NA
        return df
    cw = (cw[keys + ["PHARMACY_TYPE", "match_confidence"]]
          .drop_duplicates(keys)
          .rename(columns={"match_confidence": "type_confidence"}))

    def _jk(frame):
        return (frame["PHARMACY_NAME"].astype(str) + "|" + frame["PHARMA_CITY"].astype(str)
                + "|" + frame["PHARMA_ST"].astype(str) + "|" + frame["PHARMA_ZIP"].astype(str))

    cw_jk = _jk(cw)
    type_map = pd.Series(cw["PHARMACY_TYPE"].values, index=cw_jk)
    conf_map = pd.Series(cw["type_confidence"].values, index=cw_jk)
    jk = _jk(df)
    df["PHARMACY_TYPE"] = jk.map(type_map)
    df["type_confidence"] = jk.map(conf_map)
    return df


def _key_pharmacy_type(df: pd.DataFrame) -> pd.Series:
    """One resolved type per pharmacy_key: the most common type across its claims,
    preferring higher-confidence matches on ties."""
    if "PHARMACY_TYPE" not in df:
        return pd.Series(dtype=object)
    t = df.loc[df["PHARMACY_TYPE"].notna(),
               ["pharmacy_key", "PHARMACY_TYPE", "type_confidence"]].copy()
    if t.empty:
        return pd.Series(dtype=object)
    order = {"exact": 0, "fuzzy_high": 1, "fuzzy_review": 2}
    t["_w"] = t["type_confidence"].map(order).fillna(3)
    g = (t.groupby(["pharmacy_key", "PHARMACY_TYPE"])
           .agg(n=("PHARMACY_TYPE", "size"), w=("_w", "min")).reset_index()
           .sort_values(["pharmacy_key", "n", "w"], ascending=[True, False, True]))
    return g.groupby("pharmacy_key").head(1).set_index("pharmacy_key")["PHARMACY_TYPE"]


def signal1_cpu_vs_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Flag independent / specialty pharmacies whose avg CPU exceeds the CHAIN
    baseline by more than CPU_CHAIN_DEV_PCT.

    Chain status now comes from the **master pharmacy type** (DISPENSER_CLAS_NM via
    the crosswalk): CHAIN_BASELINE_TYPES form the baseline; everything else is a
    flag candidate. Where the type is unknown (unmatched pharmacies), we fall back
    to the store-count heuristic so coverage isn't lost. chain_source records which.
    """
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0) & df["cpu"].notna()].copy()

    key_type = _key_pharmacy_type(df)
    red["pharmacy_type"] = red["pharmacy_key"].map(key_type)

    # fallback heuristic (only used where master type is unknown)
    red["brand"] = red["pharmacy_key"].map(_chain_brand)
    stores_per_brand = red.groupby("brand")["pharmacy_key"].nunique()
    heur_chain = set(stores_per_brand[stores_per_brand >= CHAIN_MIN_STORES].index)

    type_known = red["pharmacy_type"].notna()
    red["is_chain"] = np.where(type_known,
                               red["pharmacy_type"].isin(CHAIN_BASELINE_TYPES),
                               red["brand"].isin(heur_chain))
    red["chain_source"] = np.where(type_known, "master_type", "store_count_heuristic")

    chain_baseline = red.loc[red["is_chain"], "cpu"].mean()

    # print(chain_baseline)

    g = (red.groupby("pharmacy_key")
            .agg(avg_cpu_s1=("cpu", "mean"),
                 n_fills_s1=("cpu", "size"),
                 is_chain=("is_chain", "max"),
                 pharmacy_type=("pharmacy_type", "first"),
                 chain_source=("chain_source", "first"))
            .reset_index())
    g["chain_baseline_cpu"] = chain_baseline
    g["cpu_dev_vs_chain"] = np.where((pd.notna(chain_baseline)) & (chain_baseline > 0),
                                     (g["avg_cpu_s1"] - chain_baseline) / chain_baseline,
                                     np.nan)
    g["signal1_cpu_above_chain"] = (
        (~g["is_chain"])
        & (g["n_fills_s1"] >= S1_MIN_FILLS)
        & (g["cpu_dev_vs_chain"] >= CPU_CHAIN_DEV_PCT)
    ).fillna(False)
    g.attrs["chain_baseline_cpu"] = float(chain_baseline) if pd.notna(chain_baseline) else float("nan")
    g.attrs["n_type_known"] = int(type_known.any() and key_type.notna().sum())
    g.attrs["pct_type_known"] = float(g["pharmacy_type"].notna().mean() * 100)
    return g


def signal2_cost_at_max(df: pd.DataFrame) -> pd.DataFrame:
    """Flag pharmacies where >= MAX_CLAIM_SHARE of claims are in the 'near max'
    band (>= NEAR_MAX_BAND_PCT of the program max allowable per claim)."""
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)
             & df["BENEFIT_PAID"].notna()].copy()
    program_max = float(red["BENEFIT_PAID"].quantile(MAX_BENEFIT_PCTILE))
    band_floor = NEAR_MAX_BAND_PCT * program_max
    red["at_max"] = red["BENEFIT_PAID"] >= band_floor

    g = (red.groupby("pharmacy_key")
            .agg(n_claims_s2=("BENEFIT_PAID", "size"),
                 claims_in_max_band=("at_max", "sum"))
            .reset_index())
    g["pct_claims_at_max"] = g["claims_in_max_band"] / g["n_claims_s2"].clip(lower=1)
    g["signal2_cost_at_max"] = (
        (g["n_claims_s2"] >= S2_MIN_CLAIMS)
        & (g["pct_claims_at_max"] >= MAX_CLAIM_SHARE)
    ).fillna(False)
    g.attrs["program_max"] = program_max
    g.attrs["band_floor"] = band_floor
    return g


def signal3_cpu_risrx(df: pd.DataFrame, ris: pd.DataFrame) -> pd.DataFrame:
    """
    For each monitored pharmacy, split fills into PRE / DURING / POST monitoring
    windows (interval-aware, merging multiple RisRx rows per pharmacy) and flag
    a CPU rebound: CPU drops while monitored, then recovers after removal.
    """
    # Merge monitoring intervals per pharmacy
    intervals = (
        ris.dropna(subset=["start_dt"])
           .groupby("pharmacy_key")
           .agg(first_start=("start_dt", "min"),
                last_end=("end_dt", "max"),
                still_monitored=("still_monitored", "max"),
                npi=("NPI", "first"))
           .reset_index()
    )

    sub = df[df["pharmacy_key"].isin(intervals["pharmacy_key"])].copy()
    sub = sub.merge(intervals[["pharmacy_key", "first_start", "last_end",
                               "still_monitored"]],
                    on="pharmacy_key", how="left")

    def window(row):
        if row["fill_date"] < row["first_start"]:
            return "PRE"
        if row["still_monitored"] or row["fill_date"] <= row["last_end"]:
            return "DURING"
        return "POST"

    sub["window"] = sub.apply(window, axis=1)

    # Avg CPU per pharmacy per window (positive redemptions only)
    g = (sub[sub["cpu"].notna()]
         .groupby(["pharmacy_key", "window"])
         .agg(avg_cpu=("cpu", "mean"), n_fills=("cpu", "size"))
         .reset_index())

    piv_cpu = g.pivot(index="pharmacy_key", columns="window", values="avg_cpu")
    piv_n   = g.pivot(index="pharmacy_key", columns="window", values="n_fills")
    for c in ["PRE", "DURING", "POST"]:
        if c not in piv_cpu: piv_cpu[c] = np.nan
        if c not in piv_n:   piv_n[c] = 0

    out = pd.DataFrame({
        "pharmacy_key": piv_cpu.index,
        "cpu_pre":    piv_cpu["PRE"].values,
        "cpu_during": piv_cpu["DURING"].values,
        "cpu_post":   piv_cpu["POST"].values,
        "n_pre":    piv_n["PRE"].values,
        "n_during": piv_n["DURING"].values,
        "n_post":   piv_n["POST"].values,
    }).merge(intervals[["pharmacy_key", "npi", "first_start", "last_end",
                        "still_monitored"]], on="pharmacy_key", how="left")

    # Drop during monitoring, and rebound after removal
    enough = ((out["n_pre"]    >= CPU_MIN_FILLS_PER_WINDOW) &
              (out["n_during"] >= CPU_MIN_FILLS_PER_WINDOW))
    out["cpu_drop_pct"] = np.where(out["cpu_pre"] > 0,
                                   (out["cpu_pre"] - out["cpu_during"]) / out["cpu_pre"],
                                   np.nan)

    denom = (out["cpu_pre"] - out["cpu_during"])
    out["cpu_rebound_pct"] = np.where(denom > 0,
                                      (out["cpu_post"] - out["cpu_during"]) / denom,
                                      np.nan)

    out["signal3_cpu_rebound"] = (
        enough
        & (out["n_post"] >= CPU_MIN_FILLS_PER_WINDOW)
        & (~out["still_monitored"])
        & (out["cpu_drop_pct"]    >= CPU_DROP_MIN_PCT)
        & (out["cpu_rebound_pct"] >= CPU_REBOUND_PCT)
    ).fillna(False)

    return out


def signal4_qty_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly avg qty/dispense per pharmacy; flag a period whose value is >Z SD
    from the trailing baseline OR jumps more than ABS_JUMP units month-over-month.
    """
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)].copy()

    m = (red.groupby(["pharmacy_key", "month"])
            .agg(avg_qty=("QTY_DISPENSED", "mean"),
                 n_fills=("QTY_DISPENSED", "size"))
            .reset_index()
            .sort_values(["pharmacy_key", "month"]))

    # trailing baseline (shifted so the current period is excluded)
    grp = m.groupby("pharmacy_key")["avg_qty"]
    m["roll_mean"] = grp.transform(
        lambda s: s.shift(1).rolling(QTY_TRAILING_PERIODS, min_periods=QTY_MIN_HISTORY).mean())
    m["roll_std"] = grp.transform(
        lambda s: s.shift(1).rolling(QTY_TRAILING_PERIODS, min_periods=QTY_MIN_HISTORY).std())
    m["mom_change"] = grp.transform(lambda s: s.diff())

    m["qty_z"] = (m["avg_qty"] - m["roll_mean"]) / m["roll_std"].replace(0, np.nan)

    m["signal4_qty_spike"] = (
        (m["qty_z"].abs() >= QTY_Z_THRESHOLD)
        | (m["mom_change"].abs() >= QTY_ABS_JUMP_UNITS)
    ).fillna(False)
    m["qty_direction"] = np.where(m["mom_change"] > 0, "up",
                          np.where(m["mom_change"] < 0, "down", "flat"))

    # roll up to pharmacy level
    flags = m[m["signal4_qty_spike"]]
    pharm = (m.groupby("pharmacy_key")
               .agg(qty_max_abs_z=("qty_z", lambda s: s.abs().max()),
                    qty_max_mom=("mom_change", lambda s: s.abs().max()))
               .reset_index())
    pharm["signal4_qty_spike"] = pharm["pharmacy_key"].isin(flags["pharmacy_key"])
    pharm["qty_spike_periods"] = (flags.groupby("pharmacy_key").size()
                                       .reindex(pharm["pharmacy_key"]).fillna(0).astype(int).values)
    return pharm, m


def signal7_popup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect pharmacies that go dormant (>=DORMANCY_MONTHS with no claims) and then
    reactivate with abnormally high volume in the first reactivation window, plus
    brand-new pharmacies whose very first window is abnormally high. Records both the
    triggering month (signal7_event_month) and the closest near-miss episode month
    (s7_best_episode_month) for the recency layer.
    """
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)].copy()

    pm_vol = red.groupby(["pharmacy_key", "month"]).size()
    high_vol_threshold = max(float(pm_vol.quantile(HIGH_VOLUME_PCTILE)),
                             float(HIGH_VOLUME_FLOOR))

    results = []
    for key, g in red.sort_values("fill_date").groupby("pharmacy_key"):
        dates = g["fill_date"].sort_values()
        months = pd.Series(sorted(g["month"].unique()))

        dormant_reactivation = False
        reactivation_vol = 0
        dormancy_len = 0
        reactivation_event_month = pd.NaT
        best_gap_months = 0.0
        best_gap_burst = 0
        best_episode_score = -1.0
        best_episode_month = pd.NaT
        if len(months) > 1:
            gaps = months.diff().dropna().dt.days / 30.44
            for i, gap in enumerate(gaps, start=1):
                start = months.iloc[i]
                end = start + pd.Timedelta(days=REACTIVATION_WINDOW_DAYS)
                vol = int(((dates >= start) & (dates < end)).sum())
                ep_score = min(gap / DORMANCY_MONTHS, vol / high_vol_threshold)
                if ep_score > best_episode_score:
                    best_episode_score = ep_score
                    best_gap_months = round(float(gap), 1)
                    best_gap_burst = vol
                    best_episode_month = start
                if gap >= DORMANCY_MONTHS and vol >= high_vol_threshold:
                    dormant_reactivation = True
                    reactivation_vol = max(reactivation_vol, vol)
                    dormancy_len = max(dormancy_len, round(float(gap), 1))
                    if pd.isna(reactivation_event_month) or start > reactivation_event_month:
                        reactivation_event_month = start

        f0 = dates.min()
        new_first_month = months.iloc[0] if len(months) else pd.NaT
        new_window_vol = ((dates >= f0) &
                          (dates < f0 + pd.Timedelta(days=REACTIVATION_WINDOW_DAYS))).sum()
        new_high = new_window_vol >= NEW_PHARMACY_HIGH_MULT * high_vol_threshold

        ev = [m for m, fired in [(reactivation_event_month, dormant_reactivation),
                                 (new_first_month, new_high)]
              if fired and pd.notna(m)]
        event_month = max(ev) if ev else pd.NaT
        if pd.isna(best_episode_month):       # brand-new pharmacies with no gaps
            best_episode_month = new_first_month

        results.append({
            "pharmacy_key": key,
            "signal7_dormant_reactivation": dormant_reactivation,
            "reactivation_window_vol": reactivation_vol,
            "dormancy_months": dormancy_len,
            "signal7_new_high_volume": bool(new_high),
            "new_window_vol": int(new_window_vol),
            "s7_best_gap_months": best_gap_months,
            "s7_best_gap_burst": best_gap_burst,
            "signal7_event_month": event_month,
            "s7_best_episode_month": best_episode_month,
            "total_claims": int(len(g)),
        })

    out = pd.DataFrame(results)
    out["signal7_popup"] = (out["signal7_dormant_reactivation"]
                            | out["signal7_new_high_volume"])
    out.attrs["high_vol_threshold"] = high_vol_threshold
    return out

def signal5_high_hcp(df: pd.DataFrame, flagged_keys: set):
    """Amplifier signal. For pharmacies already flagged by another signal, find
    HCPs whose prescribing is abnormally concentrated there, and roll up to a
    pharmacy-level flag. Also records signal5_event_month (most recent fill month
    of a flagged HCP at the pharmacy). Returns (pharmacy_level_df, flagged_pairs)."""
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)].copy()
    red["hcp_id"] = red["PRESC_ID"].astype(str)

    pair = (red.groupby(["hcp_id", "pharmacy_key"]).size()
               .rename("claims_at_pharm").reset_index())
    pair_last = (red.groupby(["hcp_id", "pharmacy_key"])["month"].max()
                    .rename("pair_last_month").reset_index())
    pair = pair.merge(pair_last, on=["hcp_id", "pharmacy_key"], how="left")
    hcp_tot = (pair.groupby("hcp_id")["claims_at_pharm"]
                   .agg(hcp_total="sum", hcp_n_pharm="count").reset_index())
    pair = pair.merge(hcp_tot, on="hcp_id", how="left")
    pair["hcp_avg_per_pharm"] = pair["hcp_total"] / pair["hcp_n_pharm"].clip(lower=1)
    pair["concentration"] = pair["claims_at_pharm"] / pair["hcp_total"].clip(lower=1)
    pair["volume_mult"] = pair["claims_at_pharm"] / pair["hcp_avg_per_pharm"].replace(0, np.nan)

    pair["at_flagged"] = pair["pharmacy_key"].isin(flagged_keys)
    pair["hcp_flag"] = (
        pair["at_flagged"]
        & (pair["claims_at_pharm"] >= HCP_MIN_CLAIMS)
        & ((pair["concentration"] >= HCP_CONCENTRATION_PCT)
           | (pair["volume_mult"] >= HCP_VOLUME_MULT))
    ).fillna(False)

    flagged_pairs = pair[pair["hcp_flag"]].copy()
    g = (pair.groupby("pharmacy_key")
             .agg(s5_best_concentration=("concentration", "max"),
                  s5_best_volume_mult=("volume_mult", "max"))
             .reset_index())
    g["n_high_hcps"] = (flagged_pairs.groupby("pharmacy_key").size()
                        .reindex(g["pharmacy_key"]).fillna(0).astype(int).values)
    g["signal5_high_hcp"] = g["pharmacy_key"].isin(flagged_pairs["pharmacy_key"])
    g["s5_applicable"] = g["pharmacy_key"].isin(flagged_keys)
    ev = (flagged_pairs.groupby("pharmacy_key")["pair_last_month"].max()
          .rename("signal5_event_month").reset_index())
    g = g.merge(ev, on="pharmacy_key", how="left")
    return g, flagged_pairs


def build_pharmacy_features(df: pd.DataFrame) -> pd.DataFrame:
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)].copy()
    feats = (red.groupby("pharmacy_key")
                .agg(total_claims=("QTY_DISPENSED", "size"),
                     total_units=("QTY_DISPENSED", "sum"),
                     total_benefit=("BENEFIT_PAID", "sum"),
                     avg_qty=("QTY_DISPENSED", "mean"),
                     std_qty=("QTY_DISPENSED", "std"),
                     avg_cpu=("cpu", "mean"),
                     std_cpu=("cpu", "std"),
                     avg_cpu_pct_wac=("cpu_pct_wac", "mean"),
                     n_months=("month", "nunique"),
                     n_states=("PHARMA_ST", "nunique"))
                .reset_index())

    rev = (df[df["is_reversal"]].groupby("pharmacy_key").size()
              .rename("reversal_count").reset_index())
    feats = feats.merge(rev, on="pharmacy_key", how="left")
    feats["reversal_count"] = feats["reversal_count"].fillna(0)
    feats["reversal_rate"] = feats["reversal_count"] / feats["total_claims"].clip(lower=1)
    feats["claims_per_month"] = feats["total_claims"] / feats["n_months"].clip(lower=1)
    return feats.fillna(0)


def train_signal_classifier(score: pd.DataFrame):
    """
    Train a classifier to predict whether a pharmacy is flagged by ANY signal,
    using only behavioral PROFILE features (CLF_FEATURES) -- deliberately
    excluding the thresholded trigger metrics so the model learns the *profile*
    of a flagged pharmacy rather than memorizing the rule thresholds.

    Returns (score_with_probability, metrics_dict, importances_df, fitted_model).

    NOTE: labels are rule-derived, so this is a propensity / lookalike / ranking
    model and a feature-importance lens -- NOT an independent ground-truth fraud
    detector. Interpret accordingly.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 classification_report, confusion_matrix)

    data = score.copy()
    data["label"] = (data["signals_triggered"] > 0).astype(int)

    X = data[CLF_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    y = data["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=CLF_TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)

    proba_te = clf.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= 0.5).astype(int)

    metrics = {
        "n_total": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_te, proba_te)),
        "pr_auc": float(average_precision_score(y_te, proba_te)),
        "report": classification_report(y_te, pred_te, digits=3),
        "confusion": confusion_matrix(y_te, pred_te).tolist(),
    }

    importances = (pd.DataFrame({"feature": CLF_FEATURES,
                                 "importance": clf.feature_importances_})
                     .sort_values("importance", ascending=False)
                     .reset_index(drop=True))

    # score every pharmacy (full-population propensity)
    data["signal_probability"] = clf.predict_proba(X)[:, 1]
    # "lookalikes": high model score but NOT actually flagged by any rule
    data["lookalike_flag"] = ((data["signal_probability"] >= 0.5) &
                              (data["label"] == 0))
    return data, metrics, importances, clf


def signal_proximity(score: pd.DataFrame, s3: pd.DataFrame,
                     high_vol_threshold: float) -> pd.DataFrame:
    """Per pharmacy, a distance-to-threshold proximity for each signal, where
    **1.0 == exactly at the trigger threshold** (>=1.0 would fire; <1.0 is a
    near-miss). The "closest signal" is the highest proximity among the signals
    *applicable* to that pharmacy, with a plain-English reason.

    Proximity definitions (mirror the trigger logic):
      * Signal 1 (CPU vs chain): cpu_dev_vs_chain / CPU_CHAIN_DEV_PCT
                                  (only for non-chain pharmacies; NaN for chains)
      * Signal 2 (cost at max)  : pct_claims_at_max / MAX_CLAIM_SHARE
      * Signal 3 (CPU rebound)  : min(cpu_drop_pct/CPU_DROP_MIN_PCT,
                                       cpu_rebound_pct/CPU_REBOUND_PCT)
                                  (only RisRx pharmacies w/ PRE+DURING+POST; else NaN)
      * Signal 4 (qty)          : max(max_abs_z/QTY_Z_THRESHOLD,
                                       max_mom/QTY_ABS_JUMP_UNITS)
      * Signal 5 (HCP)          : max(best_conc/HCP_CONCENTRATION_PCT,
                                       best_mult/HCP_VOLUME_MULT)
                                  (amplifier; only where Signal 5 is applicable, i.e.
                                   the pharmacy already carries an upstream flag; else NaN)
      * Signal 7 (popup)        : max(min(best_gap/DORMANCY_MONTHS,
                                          best_burst/high_vol_threshold),
                                      new_window_vol/high_vol_threshold)
    """
    out = score.copy()

    def col(name, default=np.nan):
        return out[name] if name in out else pd.Series(default, index=out.index)

    # ---- Signal 1 proximity (non-chain only) ----
    dev = col("cpu_dev_vs_chain").astype(float)
    is_chain = col("is_chain", False).fillna(False).astype(bool)
    p1 = dev / CPU_CHAIN_DEV_PCT
    out["prox_signal1"] = np.where(is_chain, np.nan, p1)

    # ---- Signal 2 proximity ----
    out["prox_signal2"] = col("pct_claims_at_max").fillna(0) / MAX_CLAIM_SHARE

    # ---- Signal 3 proximity (only where CPU windows exist) ----
    s3i = s3.set_index("pharmacy_key")
    drop = out["pharmacy_key"].map(s3i["cpu_drop_pct"]) if "cpu_drop_pct" in s3 else np.nan
    reb  = out["pharmacy_key"].map(s3i["cpu_rebound_pct"]) if "cpu_rebound_pct" in s3 else np.nan
    out["prox_signal3"] = np.minimum(drop / CPU_DROP_MIN_PCT, reb / CPU_REBOUND_PCT)

    # ---- Signal 4 proximity ----
    z = col("qty_max_abs_z").fillna(0)
    mom = col("qty_max_mom").fillna(0)
    out["prox_signal4"] = np.maximum(z / QTY_Z_THRESHOLD, mom / QTY_ABS_JUMP_UNITS)

    # ---- Signal 5 proximity (amplifier; applicable pharmacies only) ----
    conc = col("s5_best_concentration").fillna(0)
    mult = col("s5_best_volume_mult").fillna(0)
    s5_app = col("s5_applicable", False).fillna(False).astype(bool)
    p5 = np.maximum(conc / HCP_CONCENTRATION_PCT, mult / HCP_VOLUME_MULT)
    out["prox_signal5"] = np.where(s5_app, p5, np.nan)

    # ---- Signal 7 proximity ----
    gap = col("s7_best_gap_months", 0.0).fillna(0)
    burst = col("s7_best_gap_burst", 0.0).fillna(0)
    new_vol = col("new_window_vol", 0.0).fillna(0)
    dormant_path = np.minimum(gap / DORMANCY_MONTHS, burst / high_vol_threshold)
    newlaunch_path = new_vol / high_vol_threshold
    out["prox_signal7"] = np.maximum(dormant_path, newlaunch_path)

    # ---- closest signal among APPLICABLE proximities ----
    prox_cols = ["prox_signal1", "prox_signal2", "prox_signal3", "prox_signal5", "prox_signal7"]
    names = np.array(["Signal 1 (CPU vs chain)", "Signal 2 (Cost at max)",
                      "Signal 3 (CPU rebound)",
                      "Signal 5 (HCP)", "Signal 7 (Popup)"])
    arr = out[prox_cols].to_numpy(dtype=float)
    arr_for_argmax = np.where(np.isnan(arr), -np.inf, arr)
    idx = arr_for_argmax.argmax(axis=1)
    out["closest_signal"] = names[idx]
    out["closest_proximity"] = arr_for_argmax[np.arange(len(arr)), idx]
    out.loc[out["closest_proximity"] == -np.inf, ["closest_signal"]] = "None applicable"
    out.loc[out["closest_proximity"] == -np.inf, "closest_proximity"] = np.nan

    # bring s3 detail columns onto out for the reason text
    for c in ["cpu_drop_pct", "cpu_rebound_pct"]:
        if c in s3:
            out[c] = out["pharmacy_key"].map(s3i[c])

    def reason(r):
        s = r["closest_signal"]
        if s == "Signal 1 (CPU vs chain)":
            return (f"avg CPU {r.get('cpu_dev_vs_chain', float('nan')):+.0%} vs chain baseline "
                    f"(fires at +{CPU_CHAIN_DEV_PCT:.0%}, non-chain only)")
        if s == "Signal 2 (Cost at max)":
            return (f"{r.get('pct_claims_at_max', float('nan')):.0%} of claims at/near program max "
                    f"(fires at {MAX_CLAIM_SHARE:.0%})")
        if s == "Signal 3 (CPU rebound)":
            return (f"CPU dropped {r.get('cpu_drop_pct', float('nan')):.0%} / "
                    f"rebounded {r.get('cpu_rebound_pct', float('nan')):.0%} "
                    f"(fires at {CPU_DROP_MIN_PCT:.0%}/{CPU_REBOUND_PCT:.0%})")
        if s == "Signal 4 (Qty)":
            return (f"qty/dispense reached {r.get('qty_max_abs_z', float('nan')):.1f} SD "
                    f"(fires at {QTY_Z_THRESHOLD:.0f}) / max MoM jump "
                    f"{r.get('qty_max_mom', float('nan')):.0f} units (fires at {QTY_ABS_JUMP_UNITS:.0f})")
        if s == "Signal 5 (HCP)":
            return (f"top HCP {r.get('s5_best_concentration', float('nan')):.0%} concentrated / "
                    f"{r.get('s5_best_volume_mult', float('nan')):.1f}x baseline "
                    f"(fires at {HCP_CONCENTRATION_PCT:.0%} / {HCP_VOLUME_MULT:.0f}x)")
        if s == "Signal 7 (Popup)":
            return (f"best gap {r.get('s7_best_gap_months', 0):.1f} mo then "
                    f"{int(r.get('s7_best_gap_burst', 0))} claims; new-launch window "
                    f"{int(r.get('new_window_vol', 0))} claims "
                    f"(fires at {DORMANCY_MONTHS} mo & {high_vol_threshold:.0f} claims)")
        return "no signal is applicable to this pharmacy's data"

    out["closest_reason"] = out.apply(reason, axis=1)
    return out


def _peak_month(frame, metric, name):
    """Most recent month per pharmacy at which `metric` is highest (latest month
    breaks ties), returned as a 2-col frame [pharmacy_key, name]."""
    f = frame[["pharmacy_key", "month", metric]].dropna(subset=[metric])
    f = f.sort_values(["pharmacy_key", metric, "month"])      # asc; tail = max metric, latest month
    return (f.groupby("pharmacy_key").tail(1)[["pharmacy_key", "month"]]
              .rename(columns={"month": name}))


def compute_signal_recency(df: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame,
                           s4_monthly: pd.DataFrame) -> pd.DataFrame:
    """Per-pharmacy last fill month plus, for the level/behavior-based signals,
    BOTH a triggered event month (signal{1,2,4}_event_month, gated by the
    threshold) and a near-miss month (s{1,2,4}_nearmiss_month, the most recent
    month that signal's monthly metric peaked, regardless of triggering). The
    near-miss months feed closest_signal_event_month for lookalikes."""
    chain_baseline = s1.attrs.get("chain_baseline_cpu", float("nan"))
    band_floor = s2.attrs.get("band_floor", float("nan"))

    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)].copy()
    red["at_max"] = red["BENEFIT_PAID"] >= band_floor
    monthly = (red.groupby(["pharmacy_key", "month"])
                  .agg(m_cpu=("cpu", "mean"), m_claims=("cpu", "size"),
                       m_atmax=("at_max", "sum"))
                  .reset_index())
    monthly["m_share_max"] = monthly["m_atmax"] / monthly["m_claims"].clip(lower=1)

    last_fill = (red.groupby("pharmacy_key")["month"].max()
                    .rename("last_fill_month").reset_index())

    is_chain = s1.set_index("pharmacy_key")["is_chain"]
    dev_floor = chain_baseline * (1 + CPU_CHAIN_DEV_PCT)

    # --- TRIGGERED event months (threshold-gated) ---
    m1 = monthly[monthly["m_cpu"] >= dev_floor].copy()
    m1 = m1[~m1["pharmacy_key"].map(is_chain).fillna(False).astype(bool)]
    s1_ev = (m1.groupby("pharmacy_key")["month"].max()
                .rename("signal1_event_month").reset_index())
    m2 = monthly[monthly["m_share_max"] >= MAX_CLAIM_SHARE]
    s2_ev = (m2.groupby("pharmacy_key")["month"].max()
                .rename("signal2_event_month").reset_index())
    sp = s4_monthly[s4_monthly["signal4_qty_spike"]]
    s4_ev = (sp.groupby("pharmacy_key")["month"].max()
                .rename("signal4_event_month").reset_index())

    # --- NEAR-MISS months (peak of the monthly metric, no threshold) ---
    s4m = s4_monthly.copy()
    s4m["s4_prox"] = np.maximum(s4m["qty_z"].abs() / QTY_Z_THRESHOLD,
                                s4m["mom_change"].abs() / QTY_ABS_JUMP_UNITS)
    nm1 = _peak_month(monthly, "m_cpu", "s1_nearmiss_month")
    nm2 = _peak_month(monthly, "m_share_max", "s2_nearmiss_month")
    nm4 = _peak_month(s4m, "s4_prox", "s4_nearmiss_month")

    rec = last_fill
    for ev in [s1_ev, s2_ev, s4_ev, nm1, nm2, nm4]:
        rec = rec.merge(ev, on="pharmacy_key", how="left")
    return rec


def apply_recency(score: pd.DataFrame, ref_month: pd.Timestamp) -> pd.DataFrame:
    """Roll per-signal event months into last_signal_month, months_since_signal,
    recency_tier, recency_weight and priority_score. Only signals the pharmacy
    actually triggered contribute to last_signal_month."""
    out = score.copy()
    ref = pd.Timestamp(ref_month)

    ev_map = {"signal1_cpu_above_chain": "signal1_event_month",
              "signal2_cost_at_max":     "signal2_event_month",
              "signal3_cpu_rebound":     "signal3_event_month",
              "signal4_qty_spike":       "signal4_event_month",
              "signal5_high_hcp":        "signal5_event_month",
              "signal7_popup":           "signal7_event_month"}

    masked = pd.DataFrame(index=out.index)
    for flag, evc in ev_map.items():
        if evc not in out:
            out[evc] = pd.NaT
        em = pd.to_datetime(out[evc], errors="coerce")
        masked[evc] = em.where(out[flag].astype(bool), pd.NaT)
    out["last_signal_month"] = masked.max(axis=1)

    def months_to(d):
        if pd.isna(d):
            return np.nan
        return (ref.year - d.year) * 12 + (ref.month - d.month)

    out["months_since_signal"] = out["last_signal_month"].map(months_to)
    out["months_since_last_fill"] = pd.to_datetime(
        out.get("last_fill_month"), errors="coerce").map(months_to)

    def tier(r):
        if r["signals_triggered"] == 0 or pd.isna(r["months_since_signal"]):
            return "5 - No signal"
        ms = r["months_since_signal"]
        if ms <= RECENCY_ACTIVE_MONTHS: return "1 - Active"
        if ms <= RECENCY_RECENT_MONTHS: return "2 - Recent"
        if ms <= RECENCY_AGING_MONTHS:  return "3 - Aging"
        return "4 - Stale / likely remediated"

    out["recency_tier"] = out.apply(tier, axis=1)
    ms = out["months_since_signal"]
    out["recency_weight"] = np.where(out["signals_triggered"] > 0,
                                     0.5 ** (ms.fillna(np.inf) / RECENCY_HALFLIFE_MONTHS),
                                     0.0)
    out["priority_score"] = out["signals_triggered"] * out["recency_weight"]
    return out


def lookalike_recency(score: pd.DataFrame, ref_month: pd.Timestamp) -> pd.DataFrame:
    """For every pharmacy, the event month of its CLOSEST signal even when that
    signal did not trigger (near-miss), so the lookalike queue can be sorted by
    recency. Falls back to last_fill_month when a near-miss month is unavailable."""
    out = score.copy()
    ref = pd.Timestamp(ref_month)

    nearmiss_col = {"Signal 1 (CPU vs chain)": "s1_nearmiss_month",
                    "Signal 2 (Cost at max)":  "s2_nearmiss_month",
                    "Signal 3 (CPU rebound)":  "last_fill_month",
                    "Signal 5 (HCP)":          "signal5_event_month",
                    "Signal 7 (Popup)":        "s7_best_episode_month"}

    def pick(r):
        col = nearmiss_col.get(r.get("closest_signal"))
        if col is None:
            return pd.NaT
        v = r.get(col, pd.NaT)
        if pd.isna(v):
            v = r.get("last_fill_month", pd.NaT)
        return v

    out["closest_signal_event_month"] = pd.to_datetime(out.apply(pick, axis=1),
                                                        errors="coerce")

    def months_to(d):
        if pd.isna(d):
            return np.nan
        return (ref.year - d.year) * 12 + (ref.month - d.month)

    out["months_since_closest_signal"] = out["closest_signal_event_month"].map(months_to)
    return out


# ---- Pharmacy Risk Scoring / Prioritization model (Exposure + Distance + Persistence) ----
def compute_persistence(df, chain_baseline, band_floor, is_chain_map, qty_spike_map):
    """Per-pharmacy persistence = the most monthly periods any single signal stayed
    active: months a non-chain pharmacy's monthly CPU breached the chain threshold
    (S1), months its near-max share breached (S2), or S4 spike months. Max across
    them (floored to >=1 for any flagged pharmacy by the scorer)."""
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)].copy()
    red["at_max"] = red["BENEFIT_PAID"] >= band_floor
    monthly = (red.groupby(["pharmacy_key", "month"])
                  .agg(m_cpu=("cpu", "mean"), m_claims=("cpu", "size"),
                       m_atmax=("at_max", "sum")).reset_index())
    monthly["m_share"] = monthly["m_atmax"] / monthly["m_claims"].clip(lower=1)
    monthly["is_chain"] = monthly["pharmacy_key"].map(is_chain_map).fillna(False).astype(bool)
    dev_floor = chain_baseline * (1 + CPU_CHAIN_DEV_PCT)

    s1m = (monthly[(~monthly["is_chain"]) & (monthly["m_cpu"] >= dev_floor)]
           .groupby("pharmacy_key").size())
    s2m = monthly[monthly["m_share"] >= MAX_CLAIM_SHARE].groupby("pharmacy_key").size()

    keys = monthly["pharmacy_key"].drop_duplicates()
    per = pd.DataFrame(index=keys)
    per["s1"] = s1m.reindex(keys).fillna(0)
    per["s2"] = s2m.reindex(keys).fillna(0)
    per["s4"] = pd.Series(qty_spike_map).reindex(keys).fillna(0)
    return per[["s1", "s2", "s4"]].max(axis=1).astype(int)


def compute_last_signal_date(df, last_signal_month_map):
    """Day-level date of the most recent signal activity = latest fill date within
    each pharmacy's last_signal_month (signals are monthly, so this is the best
    day-resolution anchor for detectedDaysAgo)."""
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)][["pharmacy_key", "month", "fill_date"]].copy()
    red["lsm"] = red["pharmacy_key"].map(last_signal_month_map)
    inmonth = red[red["month"] == red["lsm"]]
    return inmonth.groupby("pharmacy_key")["fill_date"].max()


def compute_signal_month_benefit(df, last_signal_month_map):
    """Benefit paid in the pharmacy's triggered month (last_signal_month): sum of
    BENEFIT_PAID over valid positive redemptions whose fill month equals that month."""
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)][["pharmacy_key", "month", "BENEFIT_PAID"]].copy()
    red["lsm"] = red["pharmacy_key"].map(last_signal_month_map)
    inmonth = red[red["month"] == red["lsm"]]
    return inmonth.groupby("pharmacy_key")["BENEFIT_PAID"].sum()


def apply_risk_model(score):
    """Additive risk score: Exposure(0-40) + Distance(0-35) + Persistence(0-25).
    Risk tier from the percentile of the total over the flagged population:
    >75%ile High, 50-74%ile Medium, <50%ile Low."""
    out = score.copy()
    flag = out["signals_triggered"] > 0

    def expo(b):
        b = float(b or 0)
        return 40 if b > 100_000 else (25 if b >= 50_000 else 10)
    def dist(p):
        p = float(p or 0)
        return 35 if p >= 0.92 else (30 if p >= 0.80 else (20 if p >= 0.65 else 0))
    def pers(n):
        n = max(int(n or 0), 1)            # >=1 period for any flagged pharmacy
        return 25 if n >= 3 else (15 if n == 2 else 5)

    out["exposure_score"] = out["total_benefit"].map(expo)
    out["distance_score"] = out["signal_probability"].map(dist)
    out["persistence_score"] = out["persistence_periods"].map(pers)
    out["risk_total"] = np.where(
        flag, out["exposure_score"] + out["distance_score"] + out["persistence_score"], 0)

    tot = out.loc[flag, "risk_total"]
    p50, p75 = (float(tot.quantile(0.50)), float(tot.quantile(0.75))) if len(tot) else (0.0, 0.0)
    def tier(v, f):
        if not f: return "None"
        if v > p75: return "High"
        if v >= p50: return "Medium"
        return "Low"
    out["risk_tier"] = [tier(v, f) for v, f in zip(out["risk_total"], flag)]
    out.attrs["risk_p50"], out.attrs["risk_p75"] = p50, p75
    return out


def _flagged_ranked(score, recent_days=RECENT_DAYS, as_of=AS_OF):
    """Flagged pharmacies ranked by risk_total, with stable pharmacy_ids assigned
    over the full flagged set, then filtered to detectedDaysAgo <= recent_days.
    detectedDaysAgo = (today - last_signal_date) in days (day-level)."""
    as_of = pd.Timestamp(as_of)
    fl = score[score["signals_triggered"] > 0].copy()
    fl = fl.sort_values(["risk_total", "signals_triggered", "signal_probability"],
                        ascending=False).reset_index(drop=True)
    if "pharmacy_uid" in fl.columns:
        fl["pharmacy_id"] = fl["pharmacy_uid"].fillna(fl["pharmacy_key"])
    else:
        fl["pharmacy_id"] = [f"pharm-{j:04d}" for j in range(1, len(fl) + 1)]
    lsd = pd.to_datetime(fl.get("last_signal_date"), errors="coerce")
    dd = (as_of - lsd).dt.days
    ms = pd.to_numeric(fl.get("months_since_signal"), errors="coerce") * 30.44
    fl["detected_days_ago"] = dd.fillna(ms)
    if recent_days is not None:
        fl = fl[fl["detected_days_ago"] <= recent_days].copy()
    return fl


def build_signal_cards(score, ref_month=None, program_max=None,
                       recent_days=RECENT_DAYS, limit=None, out_path=CARDS_JSON_OUT):
    """One dashboard card per flagged pharmacy (strongest signal); detectedDaysAgo
    is day-level vs. today; priority follows the additive risk tier."""
    import json
    prox_of = {"signal1_cpu_above_chain": "prox_signal1", "signal2_cost_at_max": "prox_signal2",
               "signal3_cpu_rebound": "prox_signal3", "signal4_qty_spike": "prox_signal4",
               "signal5_high_hcp": "prox_signal5", "signal7_popup": "prox_signal7"}
    flagged = _flagged_ranked(score, recent_days)

    def primary(r):
        best, bestv = None, -1.0
        for flag, pc in prox_of.items():
            if bool(r.get(flag)):
                v = r.get(pc); v = -1.0 if pd.isna(v) else float(v)
                if v > bestv: bestv, best = v, flag
        return best
    flagged["primary_signal"] = flagged.apply(primary, axis=1)
    if limit: flagged = flagged.head(limit)

    def status_of(t):
        s = str(t)
        if s.startswith(("1", "2")): return "Active", False
        if s.startswith("3"):        return "Monitoring", False
        return "Resolved", True

    def content(r):
        sig = r["primary_signal"]; g = lambda k, d=0: (r.get(k) if pd.notna(r.get(k)) else d)
        if sig == "signal1_cpu_above_chain":
            base, avg = g("chain_baseline_cpu"), g("avg_cpu_s1"); ratio = (avg / base) if base else 0
            return ("Payout Anomaly", f"Payout {ratio:.1f}x chain pharmacy baseline",
                    "Unusual cost-per-unit vs. the chain baseline for Nurtec.",
                    [{"label": "Avg Payout/Unit", "value": f"${avg:,.0f}", "context": f"vs. ${base:,.0f} chain baseline"},
                     {"label": "Deviation", "value": f"+{g('cpu_dev_vs_chain')*100:.0f}%", "context": "above chain baseline"}])
        if sig == "signal2_cost_at_max":
            pct = g("pct_claims_at_max") * 100
            return ("Payout Anomaly", f"{pct:.0f}% of claims at program maximum",
                    "Cost per claim sits at or near the program maximum on most fills.",
                    [{"label": "Claims at Max", "value": f"{pct:.0f}%", "context": "of redemptions"},
                     {"label": "Program Max", "value": f"${(program_max or 0):,.0f}", "context": "per claim"}])
        if sig == "signal3_cpu_rebound":
            return ("Monitoring Evasion", f"CPU rebounded {g('cpu_rebound_pct')*100:.0f}% after monitoring",
                    "Cost-per-unit dropped under RisRx monitoring and rebounded afterward.",
                    [{"label": "CPU Drop", "value": f"-{g('cpu_drop_pct')*100:.0f}%", "context": "during monitoring"},
                     {"label": "CPU Rebound", "value": f"+{g('cpu_rebound_pct')*100:.0f}%", "context": "after removal"}])
        if sig == "signal4_qty_spike":
            return ("Quantity Anomaly", f"Quantity per fill spiked {g('qty_max_abs_z'):.1f} SD",
                    "Sudden change in average quantity dispensed per fill.",
                    [{"label": "Max Qty Z-score", "value": f"{g('qty_max_abs_z'):.1f}", "context": f"fires at {QTY_Z_THRESHOLD:.0f} SD"},
                     {"label": "Max MoM Jump", "value": f"{g('qty_max_mom'):.0f}", "context": "units/fill"}])
        if sig == "signal5_high_hcp":
            n = int(g("n_high_hcps"))
            return ("Prescriber Concentration", f"{n} prescriber(s) concentrated at flagged pharmacy",
                    "Prescribing is abnormally concentrated at an already-flagged pharmacy.",
                    [{"label": "High-Util HCPs", "value": f"{n}", "context": "above threshold"},
                     {"label": "Top Concentration", "value": f"{g('s5_best_concentration')*100:.0f}%", "context": "of HCP's claims here"}])
        if sig == "signal7_popup":
            if bool(r.get("signal7_dormant_reactivation")):
                rv = int(g("reactivation_window_vol"))
                return ("Pharmacy Lifecycle", f"Dormant pharmacy reactivated with {rv} claims",
                        "A dormant pharmacy resumed with abnormally high volume.",
                        [{"label": "Dormancy", "value": f"{g('dormancy_months'):.0f} mo", "context": "no claims"},
                         {"label": "Reactivation Vol", "value": f"{rv}", "context": "claims in first window"}])
            nv = int(g("new_window_vol"))
            return ("Pharmacy Lifecycle", f"New pharmacy launched at high volume ({nv} claims)",
                    "A brand-new pharmacy opened at abnormally high volume.",
                    [{"label": "Launch Volume", "value": f"{nv}", "context": "first-window claims"},
                     {"label": "High-Vol Threshold", "value": "exceeded", "context": "population p90"}])
        return ("Anomaly", "Signal triggered", "Anomalous pattern detected.", [])

    cards = []
    for i, r in enumerate(flagged.to_dict("records"), start=1):
        cat, title, summary, metrics = content(r)
        status, resolved = status_of(r.get("recency_tier"))
        dd = r.get("detected_days_ago")
        days = int(dd) if pd.notna(dd) else None
        cards.append({
            "id": f"SIG-{i:03d}", "title": title, "category": cat,
            "priority": r.get("risk_tier", "Low"), "status": status,
            "detectedDaysAgo": days,
            "exposure": int(round(float(r.get("signal_month_benefit")
                                       if pd.notna(r.get("signal_month_benefit")) else 0))),
            "fromVendor": bool(r.get("on_risrx")), "summary": summary, "resolved": resolved,
            "productIds": ["nurtec"], "pharmacyIds": [r["pharmacy_id"]],
            "metrics": metrics, "pharmacyKey": r["pharmacy_key"],
        })
    if out_path:
        with open(out_path, "w") as f: json.dump(cards, f, indent=2, default=str)
    return cards


def pharmacy_attributes(df, ris):
    """Per pharmacy_key display attributes: name, city, state, vendor names, NPI."""
    g = df.groupby("pharmacy_key")
    attrs = pd.DataFrame({"name": g["PHARMACY_NAME"].first(),
                          "city": g["PHARMA_CITY"].first(),
                          "state": g["PHARMA_ST"].first(),
                          "zip": g["PHARMA_ZIP"].first()})
    attrs["zip"] = attrs["zip"].map(
        lambda z: re.sub(r"\\D", "", str(z)).zfill(5)[:5] if pd.notna(z) else "")
    vendor_norm = {"CONNECTIVERX": "ConnectiveRx", "RELAY": "Relay"}
    def vendors(s):
        return sorted({vendor_norm.get(str(v).strip().upper(), str(v).strip().title())
                       for v in s.dropna().unique()})
    attrs["vendorNames"] = g["VNDR_NM"].agg(vendors) if "VNDR_NM" in df else [[]] * len(attrs)
    npi_map = (ris.dropna(subset=["NPI"]).drop_duplicates("pharmacy_key")
                  .set_index("pharmacy_key")["NPI"])
    attrs["npi"] = [str(int(npi_map[k])) if k in npi_map.index and pd.notna(npi_map[k]) else None
                    for k in attrs.index]
    return attrs


def build_pharmacy_records(score, attrs, recent_days=RECENT_DAYS, out_path=PHARM_JSON_OUT):
    """Pharmacy records with the additive riskScore and percentile-based tier."""
    import json
    TIER_LABEL = {"High": "Tier 1", "Medium": "Tier 2", "Low": "Tier 3", "None": "Tier 3"}
    flagged = _flagged_ranked(score, recent_days)
    def clean(v): return None if v is None or (isinstance(v, float) and pd.isna(v)) else v
    recs = []
    for r in flagged.to_dict("records"):
        key = r["pharmacy_key"]; a = attrs.loc[key] if key in attrs.index else None
        on_ris = bool(r.get("on_risrx"))
        recs.append({
            "id": r["pharmacy_id"],
            "name": clean(a["name"]) if a is not None else key,
            "npi": clean(a["npi"]) if a is not None else None,
            "city": clean(a["city"]) if a is not None else None,
            "state": clean(a["state"]) if a is not None else None,
            "type": TYPE_SHORT.get(r.get("pharmacy_type"), "Unknown"),
            "tier": TIER_LABEL.get(r.get("risk_tier"), "Tier 3"),
            "riskScore": int(round(float(r.get("risk_total") or 0))),
            "watched": on_ris, "watchlistSource": "RisRx" if on_ris else None,
            "isOnRisRxList": on_ris,
            "vendorNames": list(a["vendorNames"]) if (a is not None and isinstance(a["vendorNames"], (list, tuple))) else [],
            "productIds": ["nurtec"], "pharmacyKey": key,
        })
    if out_path:
        with open(out_path, "w") as f: json.dump(recs, f, indent=2, default=str)
    return recs


def build_signal_cards_per_signal(score, df, program_max=None, recent_days=RECENT_DAYS,
                                  per_signal_recent=False, out_path=CARDS_JSON_OUT):
    """Signal-level bifurcation: one card per (pharmacy, triggered signal). Each card
    carries that signal's own event month, exposure (benefit in that month), and
    detectedDaysAgo. Pharmacy-level fields (priority/pharmacyId) repeat across a
    pharmacy's cards. recent_days filters pharmacies (by their most recent signal);
    set per_signal_recent=True to additionally drop signal-cards older than recent_days."""
    import json
    sig_meta = [("signal1_cpu_above_chain", "S1_CPU_Inflation",     "Payout Anomaly",          "signal1_event_month"),
                ("signal2_cost_at_max",     "S2_Cost_At_Max",       "Payout Anomaly",          "signal2_event_month"),
                ("signal3_cpu_rebound",     "S3_CPU_Rebound",       "Monitoring Evasion",      "signal3_event_month"),
                ("signal4_qty_spike",       "S4_Quantity_Spike",    "Quantity Anomaly",        "signal4_event_month"),
                ("signal5_high_hcp",        "S5_HCP_Concentration", "Prescriber Concentration","signal5_event_month"),
                ("signal7_popup",           "S7_Popup_Dormant",     "Pharmacy Lifecycle",      "signal7_event_month")]

    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)][["pharmacy_key", "month", "fill_date", "BENEFIT_PAID"]]
    gm = red.groupby(["pharmacy_key", "month"])
    last_fill = gm["fill_date"].max()
    month_ben = gm["BENEFIT_PAID"].sum()

    def content(r, flag):
        g = lambda k, d=0: (r.get(k) if pd.notna(r.get(k)) else d)
        if flag == "signal1_cpu_above_chain":
            base, avg = g("chain_baseline_cpu"), g("avg_cpu_s1"); ratio = (avg / base) if base else 0
            return (f"Payout {ratio:.1f}x chain pharmacy baseline",
                    "Unusual cost-per-unit vs. the chain baseline for Nurtec.",
                    [{"label": "Avg Payout/Unit", "value": f"${avg:,.0f}", "context": f"vs. ${base:,.0f} chain baseline"},
                     {"label": "Deviation", "value": f"+{g('cpu_dev_vs_chain')*100:.0f}%", "context": "above chain baseline"}])
        if flag == "signal2_cost_at_max":
            pct = g("pct_claims_at_max") * 100
            return (f"{pct:.0f}% of claims at program maximum",
                    "Cost per claim sits at or near the program maximum on most fills.",
                    [{"label": "Claims at Max", "value": f"{pct:.0f}%", "context": "of redemptions"},
                     {"label": "Program Max", "value": f"${(program_max or 0):,.0f}", "context": "per claim"}])
        if flag == "signal3_cpu_rebound":
            return (f"CPU rebounded {g('cpu_rebound_pct')*100:.0f}% after monitoring",
                    "Cost-per-unit dropped under RisRx monitoring and rebounded afterward.",
                    [{"label": "CPU Drop", "value": f"-{g('cpu_drop_pct')*100:.0f}%", "context": "during monitoring"},
                     {"label": "CPU Rebound", "value": f"+{g('cpu_rebound_pct')*100:.0f}%", "context": "after removal"}])
        if flag == "signal4_qty_spike":
            return (f"Quantity per fill spiked {g('qty_max_abs_z'):.1f} SD",
                    "Sudden change in average quantity dispensed per fill.",
                    [{"label": "Max Qty Z-score", "value": f"{g('qty_max_abs_z'):.1f}", "context": f"fires at {QTY_Z_THRESHOLD:.0f} SD"},
                     {"label": "Max MoM Jump", "value": f"{g('qty_max_mom'):.0f}", "context": "units/fill"}])
        if flag == "signal5_high_hcp":
            n = int(g("n_high_hcps"))
            return (f"{n} prescriber(s) concentrated at flagged pharmacy",
                    "Prescribing is abnormally concentrated at an already-flagged pharmacy.",
                    [{"label": "High-Util HCPs", "value": f"{n}", "context": "above threshold"},
                     {"label": "Top Concentration", "value": f"{g('s5_best_concentration')*100:.0f}%", "context": "of HCP's claims here"}])
        if flag == "signal7_popup":
            if bool(r.get("signal7_dormant_reactivation")):
                rv = int(g("reactivation_window_vol"))
                return (f"Dormant pharmacy reactivated with {rv} claims",
                        "A dormant pharmacy resumed with abnormally high volume.",
                        [{"label": "Dormancy", "value": f"{g('dormancy_months'):.0f} mo", "context": "no claims"},
                         {"label": "Reactivation Vol", "value": f"{rv}", "context": "claims in first window"}])
            nv = int(g("new_window_vol"))
            return (f"New pharmacy launched at high volume ({nv} claims)",
                    "A brand-new pharmacy opened at abnormally high volume.",
                    [{"label": "Launch Volume", "value": f"{nv}", "context": "first-window claims"},
                     {"label": "High-Vol Threshold", "value": "exceeded", "context": "population p90"}])
        return ("Signal triggered", "Anomalous pattern detected.", [])

    def status_of(dda):
        if dda is None: return "Monitoring", False
        if dda <= 167: return "Active", False
        if dda <= 335: return "Monitoring", False
        return "Resolved", True

    flagged = _flagged_ranked(score, recent_days)
    cards = []
    i = 0
    for r in flagged.to_dict("records"):
        key = r["pharmacy_key"]; pid = r["pharmacy_id"]
        for flag, sid, cat, evcol in sig_meta:
            if not bool(r.get(flag, False)):
                continue
            em = r.get(evcol)
            if pd.isna(em):
                em = r.get("last_signal_month")
            em = pd.Timestamp(em) if pd.notna(em) else None
            sdate = last_fill.get((key, em)) if em is not None else None
            dda = int((pd.Timestamp(AS_OF) - pd.Timestamp(sdate)).days) if (sdate is not None and pd.notna(sdate)) else None
            if per_signal_recent and (dda is None or dda > recent_days):
                continue
            ben = month_ben.get((key, em), 0.0) if em is not None else 0.0
            title, summary, metrics = content(r, flag)
            status, resolved = status_of(dda)
            i += 1
            cards.append({
                "id": f"SIG-{i:04d}", "title": title, "category": cat,
                "signalId": sid,
                "signalTriggeredMonth": em.strftime("%Y-%m") if em is not None else None,
                "priority": r.get("risk_tier", "Low"), "status": status,
                "detectedDaysAgo": dda, "exposure": int(round(float(ben or 0))),
                "fromVendor": bool(r.get("on_risrx")), "summary": summary, "resolved": resolved,
                "productIds": ["nurtec"], "pharmacyIds": [pid],
                "metrics": metrics, "pharmacyKey": key,
            })
    if out_path:
        with open(out_path, "w") as f:
            json.dump(cards, f, indent=2, default=str)
    return cards


def build_signal_long_table(score, df, ris, chain_baseline, band_floor,
                            recent_days=None, primary_only=True, run_date=None,
                            out_path=SIGNAL_OUTPUT_CSV):
    run_date = run_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    if recent_days is not None:                       # restrict to the recent flagged set
        keep = set(_flagged_ranked(score, recent_days)["pharmacy_key"])
        score = score[score["pharmacy_key"].isin(keep)]
    sig_meta = [("signal1_cpu_above_chain", "S1_CPU_Inflation", "Cost"),
                ("signal2_cost_at_max",     "S2_Cost_At_Max",   "Cost"),
                ("signal3_cpu_rebound",     "S3_CPU_Rebound",   "Cost"),
                ("signal4_qty_spike",       "S4_Quantity_Spike", "Quantity"),
                ("signal5_high_hcp",        "S5_HCP_Concentration", "Utilization"),
                ("signal7_popup",           "S7_Popup_Dormant", "Pharmacy")]

    # per-signal persistence (active months) for the monthly-cadence signals
    red = df[(~df["is_reversal"]) & (df["QTY_DISPENSED"] > 0)].copy()
    red["at_max"] = red["BENEFIT_PAID"] >= band_floor
    monthly = (red.groupby(["pharmacy_key", "month"])
                  .agg(m_cpu=("cpu", "mean"), m_claims=("cpu", "size"),
                       m_atmax=("at_max", "sum")).reset_index())
    is_chain = score.set_index("pharmacy_key")["is_chain"]
    monthly["is_chain"] = monthly["pharmacy_key"].map(is_chain).fillna(False).astype(bool)
    monthly["share"] = monthly["m_atmax"] / monthly["m_claims"].clip(lower=1)
    dev_floor = chain_baseline * (1 + CPU_CHAIN_DEV_PCT)
    s1c = (monthly[(~monthly["is_chain"]) & (monthly["m_cpu"] >= dev_floor)]
           .groupby("pharmacy_key").size())
    s2c = monthly[monthly["share"] >= MAX_CLAIM_SHARE].groupby("pharmacy_key").size()

    g = df.groupby("pharmacy_key")
    name = g["PHARMACY_NAME"].first(); zipf = g["PHARMA_ZIP"].first()
    npi_map = (ris.dropna(subset=["NPI"]).drop_duplicates("pharmacy_key")
                  .set_index("pharmacy_key")["NPI"])

    prox_of = {"signal1_cpu_above_chain": "prox_signal1", "signal2_cost_at_max": "prox_signal2",
               "signal3_cpu_rebound": "prox_signal3", "signal4_qty_spike": "prox_signal4",
               "signal5_high_hcp": "prox_signal5", "signal7_popup": "prox_signal7"}
    event_of = {"signal1_cpu_above_chain": "signal1_event_month", "signal2_cost_at_max": "signal2_event_month",
                "signal3_cpu_rebound": "signal3_event_month", "signal4_qty_spike": "signal4_event_month",
                "signal5_high_hcp": "signal5_event_month", "signal7_popup": "signal7_event_month"}
    rows = []
    for key, r in score.set_index("pharmacy_key").iterrows():
        if int(r.get("signals_triggered") or 0) == 0:
            continue
        triggered = [(f, s, c) for f, s, c in sig_meta if bool(r.get(f, False))]
        if primary_only and triggered:        # one row per entity = its strongest signal
            best, bestv = triggered[0], -1.0
            for f, s, c in triggered:
                v = r.get(prox_of[f]); v = -1.0 if pd.isna(v) else float(v)
                if v > bestv: bestv, best = v, (f, s, c)
            triggered = [best]
        z = zipf.get(key)
        try: zstr = str(float(z))
        except Exception: zstr = str(z)
        entity = f"{name.get(key, key)};{zstr}"
        npi = npi_map.get(key)
        npi_str = str(int(npi)) if pd.notna(npi) else "nan"
        benefit = round(float(r.get("total_benefit") or 0), 2)
        prob = r.get("signal_probability")
        dval = round(float(prob), 6) if pd.notna(prob) else None
        for flag, sid, cat in triggered:
            if flag == "signal1_cpu_above_chain":  pp = int(s1c.get(key, 0) or 0)
            elif flag == "signal2_cost_at_max":    pp = int(s2c.get(key, 0) or 0)
            elif flag == "signal4_qty_spike":      pp = int(r.get("qty_spike_periods") or 0)
            else:                                  pp = 1
            em = r.get("last_signal_month")   # month the pharmacy was most recently flagged
            stm = pd.Timestamp(em).strftime("%Y-%m") if pd.notna(em) else ""
            rows.append({"ENTITY_ID": entity, "SIGNAL_ID": sid, "CATEGORY": cat,
                         "SIGNAL_TRIGGERED_MONTH": stm,
                         "BENEFIT_PAID": benefit, "DISTANCE_TYPE": "ML model",
                         "DISTANCE_VALUE": dval, "RUN_DATE": run_date,
                         "PERSISTENCE_PERIODS": max(pp, 1), "NPI": npi_str})

    out = pd.DataFrame(rows, columns=["ENTITY_ID", "SIGNAL_ID", "CATEGORY", "SIGNAL_TRIGGERED_MONTH",
                                      "BENEFIT_PAID", "DISTANCE_TYPE", "DISTANCE_VALUE", "RUN_DATE",
                                      "PERSISTENCE_PERIODS", "NPI"])
    if out_path:
        if str(out_path).lower().endswith((".xlsx", ".xls")):
            out.to_excel(out_path, index=False)
        else:
            out.to_csv(out_path, index=False)
    return out


# ---------------------------------------------------------------------------
# Train / score split additions
# ---------------------------------------------------------------------------
import os
import joblib
from datetime import datetime

MODEL_DIR                 = "models"
MODEL_BUNDLE              = "eucrisa_models.joblib"
MODEL_META               = "model_metadata.json"
LOOKALIKE_PROB_THRESHOLD  = 0.50   # signal_probability cutoff for a "lookalike"
PROPENSITY_MIN_POSITIVES  = 50     # per-signal one-vs-rest min positives to fit
REFIT_CLASSIFIER_ON_FULL  = True   # deploy a model fit on ALL rows (eval still uses a hold-out)
LOOKALIKE_XLSX            = "Eucrisa_Lookalike_Triage.xlsx"


# ===========================================================================
# Shared deterministic stage: rule engine -> base scorecard (NO ML, NO recency)
# Both notebooks call this so the signal logic can never drift between them.
# ===========================================================================
def run_rule_engine(df: pd.DataFrame, ris: pd.DataFrame):
    """Run signals 1-5,7 and assemble the base scorecard (one row per pharmacy)
    with all signal flags, ``signals_triggered`` and behavioral features merged
    in. Returns (score, detail). Deliberately excludes the anomaly model, the
    supervised classifier, recency and the closest-signal layer -- those are the
    train/score-specific stages layered on top by each notebook.
    """
    ref_month = df["month"].max()

    print("Signal 1 (CPU vs chain baseline) ...")
    s1 = signal1_cpu_vs_chain(df)
    print(f"  flagged: {int(s1['signal1_cpu_above_chain'].sum())}")
    print("Signal 2 (cost per claim at max) ...")
    s2 = signal2_cost_at_max(df)
    print(f"  flagged: {int(s2['signal2_cost_at_max'].sum())}")
    print("Signal 3 (CPU pre/post RisRx) ...")
    s3 = signal3_cpu_risrx(df, ris)
    print(f"  flagged rebounds: {int(s3['signal3_cpu_rebound'].sum())}")
    # print("Signal 4 (quantity change) ...")
    s4, s4_monthly = signal4_qty_change(df)
    print(f"  flagged pharmacies: {int(s4['signal4_qty_spike'].sum())}")
    print("Signal 7 (popup/dormant) ...")
    s7 = signal7_popup(df)
    print(f"  flagged pharmacies: {int(s7['signal7_popup'].sum())}")

    upstream_flagged = (
        set(s1.loc[s1["signal1_cpu_above_chain"], "pharmacy_key"])
        | set(s2.loc[s2["signal2_cost_at_max"], "pharmacy_key"])
        | set(s3.loc[s3["signal3_cpu_rebound"], "pharmacy_key"])
        | set(s7.loc[s7["signal7_popup"], "pharmacy_key"])
    )
    print(f"Signal 5 (high HCP utilization at {len(upstream_flagged):,} flagged pharmacies) ...")
    s5, s5_pairs = signal5_high_hcp(df, upstream_flagged)
    print(f"  flagged pharmacies: {int(s5['signal5_high_hcp'].sum())}")

    print("Building pharmacy features ...")
    feats = build_pharmacy_features(df)

    score = (feats
        .merge(s1[["pharmacy_key", "avg_cpu_s1", "is_chain", "chain_baseline_cpu",
                   "cpu_dev_vs_chain", "pharmacy_type", "chain_source",
                   "signal1_cpu_above_chain"]], on="pharmacy_key", how="left")
        .merge(s2[["pharmacy_key", "n_claims_s2", "claims_in_max_band",
                   "pct_claims_at_max", "signal2_cost_at_max"]], on="pharmacy_key", how="left")
        .merge(s3[["pharmacy_key", "npi", "cpu_pre", "cpu_during", "cpu_post",
                   "cpu_drop_pct", "cpu_rebound_pct", "signal3_cpu_rebound"]], on="pharmacy_key", how="left")
        .merge(s5[["pharmacy_key", "s5_best_concentration", "s5_best_volume_mult",
                   "n_high_hcps", "s5_applicable", "signal5_high_hcp",
                   "signal5_event_month"]], on="pharmacy_key", how="left")
        .merge(s7[["pharmacy_key", "signal7_dormant_reactivation",
                   "reactivation_window_vol", "dormancy_months",
                   "signal7_new_high_volume", "new_window_vol",
                   "s7_best_gap_months", "s7_best_gap_burst",
                   "signal7_event_month", "s7_best_episode_month",
                   "signal7_popup"]], on="pharmacy_key", how="left"))

    bool_cols = ["signal1_cpu_above_chain", "signal2_cost_at_max",
                 "signal3_cpu_rebound", "signal5_high_hcp",
                 "signal7_popup", "signal7_dormant_reactivation",
                 "signal7_new_high_volume", "is_chain", "s5_applicable"]
    for c in bool_cols:
        score[c] = score[c].fillna(False).astype(bool)

    signal_flags = ["signal1_cpu_above_chain", "signal2_cost_at_max",
                    "signal3_cpu_rebound",
                    "signal5_high_hcp", "signal7_popup"]
    score["signals_triggered"] = score[signal_flags].sum(axis=1)
    score["on_risrx"] = score["pharmacy_key"].isin(set(ris["pharmacy_key"]))

    print(f"\nRule engine: {int((score['signals_triggered']>0).sum()):,} pharmacies "
          f"tripped >=1 signal; {int((score['signals_triggered']>=2).sum()):,} tripped >=2.")
    for f in signal_flags:
        print(f"    {f:<26} {int(score[f].sum()):,}")

    detail = {"s1": s1, "s2": s2, "s3": s3, "s4_monthly": s4_monthly,
              "s5": s5, "s5_pairs": s5_pairs, "s7": s7, "feats": feats,
              "ref_month": ref_month, "signal_flags": signal_flags}
    return score, detail


# ===========================================================================
# ML stage 1: composite anomaly model  (fit / apply split)
# ===========================================================================
_ANOM_COLS = ["avg_qty", "std_qty", "avg_cpu", "std_cpu", "avg_cpu_pct_wac",
              "claims_per_month", "reversal_rate", "n_states", "total_units"]

def fit_anomaly_model(score: pd.DataFrame) -> dict:
    """Fit IsolationForest + RobustScaler on the training population and capture
    the min/max of the raw anomaly score so inference can normalise to the SAME
    0-1 scale. Returns an artifact dict (pickled into the model bundle)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler
    X = score[_ANOM_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    iso = IsolationForest(contamination=ISO_CONTAMINATION,
                          random_state=RANDOM_STATE, n_estimators=300).fit(Xs)
    raw = -iso.decision_function(Xs)          # higher = more anomalous
    return {"scaler": scaler, "iso": iso, "model_cols": _ANOM_COLS,
            "raw_min": float(raw.min()), "raw_max": float(raw.max())}

def apply_anomaly_model(score: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Score a (possibly new) population with a fitted anomaly artifact, using
    the training-time min/max so anomaly_score stays comparable across runs."""
    out = score.copy()
    X = out[art["model_cols"]].replace([np.inf, -np.inf], 0).fillna(0)
    Xs = art["scaler"].transform(X)
    raw = -art["iso"].decision_function(Xs)
    lo, hi = art["raw_min"], art["raw_max"]
    out["anomaly_score"] = ((raw - lo) / (hi - lo + 1e-9)).clip(0, 1)
    out["is_anomaly"] = art["iso"].predict(Xs) == -1
    return out


# ===========================================================================
# ML stage 2: supervised classifier  (train_signal_classifier above = fit+eval)
#   - fit_classifier_full  : deploy model fit on ALL rows
#   - apply_signal_classifier : score + tag lookalikes with a saved model
# ===========================================================================
def fit_classifier_full(score: pd.DataFrame):
    """Refit the lookalike classifier on the FULL population for deployment
    (train_signal_classifier still holds out a test split for honest metrics)."""
    from sklearn.ensemble import RandomForestClassifier
    y = (score["signals_triggered"] > 0).astype(int).values
    X = score[CLF_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    clf = RandomForestClassifier(n_estimators=400, max_depth=None,
                                 min_samples_leaf=5, class_weight="balanced",
                                 n_jobs=-1, random_state=RANDOM_STATE)
    clf.fit(X, y)
    return clf

def apply_signal_classifier(score: pd.DataFrame, clf,
                            prob_threshold: float = None) -> pd.DataFrame:
    """Score every pharmacy with a saved classifier and tag lookalikes =
    high model probability AND not tripped by any rule (label == 0)."""
    if prob_threshold is None:
        prob_threshold = LOOKALIKE_PROB_THRESHOLD
    out = score.copy()
    out["label"] = (out["signals_triggered"] > 0).astype(int)
    X = out[CLF_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    out["signal_probability"] = clf.predict_proba(X)[:, 1]
    out["lookalike_flag"] = ((out["signal_probability"] >= prob_threshold) &
                             (out["label"] == 0))
    return out


# ===========================================================================
# ML stage 3: per-signal propensity  (fit / apply split)
# ===========================================================================
_PROP_FLAG_MAP = {"p_signal1": "signal1_cpu_above_chain",
                  "p_signal2": "signal2_cost_at_max",
                  "p_signal3": "signal3_cpu_rebound",
                  "p_signal4": "signal4_qty_spike",
                  "p_signal5": "signal5_high_hcp",
                  "p_signal7": "signal7_popup"}

def fit_per_signal_propensity(score: pd.DataFrame, min_positives: int = None) -> dict:
    """Fit a one-vs-rest RandomForest per signal that has >= min_positives
    positives. Returns {p_signalX: fitted_model}. Signals below the bar are
    simply absent from the dict (their column becomes NaN at apply time).

    NOTE: p_signal1 / p_signal2 overlap cost features in CLF_FEATURES, so those
    propensities carry some target leakage -- read as propensity, not proof.
    """
    if min_positives is None:
        min_positives = PROPENSITY_MIN_POSITIVES
    from sklearn.ensemble import RandomForestClassifier
    X = score[CLF_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    models = {}
    for pcol, flag in _PROP_FLAG_MAP.items():
        if flag not in score:
            continue
        y = score[flag].astype(int).values
        if y.sum() < min_positives:
            continue
        models[pcol] = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE).fit(X, y)
    return models

def apply_per_signal_propensity(score: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Add p_signal* columns using saved propensity models; signals without a
    model (too few positives at train time) get NaN."""
    out = score.copy()
    X = out[CLF_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    for pcol in _PROP_FLAG_MAP:
        out[pcol] = models[pcol].predict_proba(X)[:, 1] if pcol in models else np.nan
    return out


# ===========================================================================
# Persistence
# ===========================================================================
def save_model_bundle(bundle: dict, model_dir: str = MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(bundle, os.path.join(model_dir, MODEL_BUNDLE))
    with open(os.path.join(model_dir, MODEL_META), "w") as fh:
        json.dump(bundle.get("metadata", {}), fh, indent=2, default=str)
    print(f"Saved {os.path.join(model_dir, MODEL_BUNDLE)}")
    print(f"Saved {os.path.join(model_dir, MODEL_META)}")

def load_model_bundle(model_dir: str = MODEL_DIR) -> dict:
    path = os.path.join(model_dir, MODEL_BUNDLE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run the training notebook (01) first.")
    bundle = joblib.load(path)
    meta = bundle.get("metadata", {})
    print(f"Loaded model bundle trained at {meta.get('trained_at')} "
          f"on {meta.get('n_pharmacies')} pharmacies "
          f"(ref month {meta.get('ref_month')}).")
    return bundle