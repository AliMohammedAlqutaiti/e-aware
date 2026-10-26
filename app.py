Check official fishing restrictions before departure.

E-AWARE provides habitat guidance only.
"""

st.text_area(
    "Example SMS",
    sms_message,
    height=500
)


# ============================================================
# TRANSPARENCY / STATUS
# ============================================================
st.subheader(
    "📊 Data Sources & Transparency"
)

source_table = pd.DataFrame(
    [
        {
            "Source":
                "Copernicus Marine Physics Analysis & Forecast",
            "Use":
                "Live SST, salinity and SST anomaly"
        },

        {
            "Source":
                "Copernicus Marine Biogeochemistry Analysis & Forecast",
            "Use":
                "Live surface oxygen and vertical oxygen profile"
        },

        {
            "Source":
                "Copernicus Global Ocean Physics Reanalysis",
            "Use":
                "Historical SST and salinity from 1993 onward"
        },

        {
            "Source":
                "Copernicus Global Ocean Biogeochemistry Hindcast",
            "Use":
                "Historical oxygen and oxycline from 1993 onward"
        },

        {
            "Source":
                "GEBCO_2026 via Ocean Data Bank API",
            "Use":
                "Bathymetry / local water-column limit"
        },

        {
            "Source":
                "NOAA NCEI coastline",
            "Use":
                "Nearest-coast distance"
        },

        {
            "Source":
                "Historical anchoveta CSV",
            "Use":
                "Observed anchoveta target data when supplied "
                "from a traceable source"
        },
    ]
)

st.dataframe(
    source_table,
    use_container_width=True,
    hide_index=True
)

st.subheader(
    "🧪 V2 Step 3 Model Status"
)

cols = st.columns(
    2
)

cols[0].success(
    "Historical observations can now be ingested, audited, matched to "
    "historical environmental predictors and QC-screened without "
    "altering the live HSI."
)

cols[1].warning(
    "No machine-learning model is trained yet. "
    "The next stage after a defensible historical dataset passes Step 3 QC "
    "is a versioned final training table, followed by a Logistic Regression baseline."
)

with st.expander(
    "⚙️ V2 Step 3 Limitations"
):

    st.markdown(
        """
- Historical Copernicus matching begins on **1 January 1993**. Earlier observations remain in the table but are not assigned invented environmental values.
- Physics is matched to the nearest 0.083° reanalysis grid cell; biogeochemistry is matched independently to the nearest 0.25° hindcast grid cell.
- Historical oxycline is calculated from the 1 mL/L dissolved-oxygen threshold and constrained by local GEBCO water depth.
- A missing historical record is never interpreted as fish absence.
- The Streamlit matcher intentionally limits the number of observations processed in one interactive run. A large national archive should later be batch-processed offline and versioned.
- A true historical SST-anomaly predictor is not added in Step 2. It should be calculated from a documented historical climatology rather than copied from the live forecast-anomaly product.
- Step 2 creates candidate predictor rows only; it does not prove habitat-model accuracy.
- Step 3 QC checks structural and environmental data quality, but it cannot prove that survey absences are valid unless the source methodology supports that interpretation.
- QC match-distance thresholds are engineering warning limits for investigation; they are not biological habitat thresholds.
- Passing Step 3 QC does not remove sampling bias, spatial autocorrelation or survey-design dependence.
        """
    )

st.divider()

st.caption(
    "E-AWARE V2 Step 3 | "
    "Historical observation ↔ environmental matching ↔ training-data QC | "
    "Decision-support research prototype"
)

st.caption(
    "Habitat suitability is a model estimate and must not be used "
