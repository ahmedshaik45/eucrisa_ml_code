import pandas as pd, numpy as np

# ---- load the 3 Snowflake exports ----
df = pd.read_csv("copay_eucrisa_raw.csv", low_memory=False)
print([c for c in df.columns])

df.columns = df.columns.str.upper()          # normalize all to UPPERCASE
# now CLAIM_TYPE_FIXED already exists — no rename needed, delete the df["CLAIM_TYPE_FIXED"] = ... line
ship = pd.read_csv("shipping_867_eucrisa.csv", low_memory=False)
# dim_pharmacy.csv is used to build your Pharmacy_Type_Crosswalk separately (see note below)

# ---- STEP 2: cleaning + helper columns ----
df["PHARMA_ZIP"] = (df["PHARMA_ZIP"].astype(str)
                    .str.replace(r"\.0$", "", regex=True).str.zfill(5))
fill = pd.to_datetime(df["DATE_OF_FILL"], errors="coerce")
df["Month_of_fill"] = fill.dt.month
df["Year_of_fill"]  = fill.dt.year
df["Month_short"]   = fill.dt.strftime("%b")

# ---- NDC -> WAC (per-gram) ----
WAC_BY_NDC = {"55724021121": 793.41/60, "55724021111": 1115.68/100}
ndc = df["DRUG_NDC_NBR"].astype(str).str.replace(r"\D", "", regex=True).str.strip()
df["WAC"] = ndc.map(WAC_BY_NDC)

# # ---- CLAIM_TYPE_FIXED already comes from Snowflake as `claim_type_fixed` ----
# df["CLAIM_TYPE_FIXED"] = df["claim_type_fixed"]

# ---- STEP 4: merge shipping (867) at pharmacy + month ----
ship = pd.read_csv("shipping_867_eucrisa.csv", low_memory=False)

# normalize shipping join keys
ship["_ship_name"]  = ship["POC NAME"].astype(str).str.upper().str.strip()
ship["_ship_zip"]   = (ship["POC ZIP"].astype(str)
                       .str.replace(r"\.0$", "", regex=True).str.zfill(5))
ship["_ship_month"] = ship["CALENDAR MONTH"].astype(str).str.strip().str.upper()   # JUL
ship["_ship_year"]  = pd.to_numeric(ship["CALENDAR YEAR"], errors="coerce").astype("Int64")

# matching keys on the copay side
df["_cop_name"]  = df["PHARMACY_NAME"].astype(str).str.upper().str.strip()
df["_cop_month"] = df["Month_short"].astype(str).str.upper()                        # JUL (Month_short is 'Jul' -> upper)
df["_cop_year"]  = pd.to_numeric(df["Year_of_fill"], errors="coerce").astype("Int64")

df = df.merge(
    ship,
    left_on=["_cop_name", "PHARMA_ZIP", "_cop_month", "_cop_year"],
    right_on=["_ship_name", "_ship_zip", "_ship_month", "_ship_year"],
    how="left",
    suffixes=("", "_SHIP"))     # keeps your copay WAC as 'WAC'

# ---- STEP 7: final model filters ----
df = df[(df["COVERAGE_TYPE"] == 8)
        & (df["WAC_PRICE"].notna())
        & (df["CLAIM_TYPE_FIXED"].str.upper() == "REDEMPTION")].copy()

df.to_csv("COPAY_EUCRISA_CLEAN.csv", index=False)
print(f"Wrote COPAY_EUCRISA_CLEAN.csv : {len(df):,} rows")