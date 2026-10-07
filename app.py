import os
import streamlit as st
import copernicusmarine

st.set_page_config(
    page_title="E-AWARE Copernicus Test",
    page_icon="🐟"
)

st.title("🐟 E-AWARE Copernicus Connection Test")

username = st.secrets["COPERNICUS_USERNAME"]
password = st.secrets["COPERNICUS_PASSWORD"]

os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = username
os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = password

# Helps avoid server/environment configuration interfering
os.environ["COPERNICUSMARINE_TRUST_ENV"] = "False"

st.write("### Test 1 — Credentials")

try:

    valid = copernicusmarine.login(
        username=username,
        password=password,
        check_credentials_valid=True
    )

    if valid:
        st.success("✅ Copernicus credentials are valid.")
    else:
        st.error("❌ Copernicus credentials are NOT valid.")
        st.stop()

except Exception as e:

    st.error("❌ Credential test failed.")
    st.code(str(e))
    st.stop()


st.write("### Test 2 — Temperature request")

try:

    st.write("Testing the request without downloading data...")

    result = copernicusmarine.subset(

        dataset_id=
        "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",

        variables=[
            "thetao"
        ],

        minimum_longitude=-81.50,
        maximum_longitude=-81.20,

        minimum_latitude=-4.60,
        maximum_latitude=-4.36,

        minimum_depth=0,
        maximum_depth=1,

        start_datetime="2026-10-01T00:00:00",
        end_datetime="2026-10-02T00:00:00",

        username=username,
        password=password,

        service="arco-geo-series",

        dry_run=True,

        disable_progress_bar=True
    )

    st.success("✅ Temperature request is valid.")
    st.write(result)

except Exception as e:

    st.error("❌ Temperature request failed.")
    st.code(str(e))
