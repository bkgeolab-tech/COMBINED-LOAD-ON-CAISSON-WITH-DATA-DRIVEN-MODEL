import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from pathlib import Path

# ================== LOAD MODEL ==================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "thebestmodel.pkl")

@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)

bundle  = load_bundle(MODEL_PATH)
model   = bundle["model"]
scalerX = bundle.get("scaler_X", None)
scalerY = bundle.get("scaler_y", None)

# ================== PAGE & THEME ==================
st.set_page_config(page_title="Bearing Capacity Predictor", page_icon="🧮", layout="centered")

st.markdown("""
<style>
.stApp { background:#ffffff !important; color:#000000 !important; }
header[data-testid="stHeader"] { display:none; }
.block-container { padding-top: 1rem; }

/* inputs & buttons */
.stNumberInput input { background:#fff !important; color:#000 !important; border:1px solid #00000040 !important; border-radius:6px !important; }
.stButton > button, .stDownloadButton > button { background:#fff !important; color:#000 !important; border:1px solid #000 !important; border-radius:6px !important; }

/* image helpers */
.duo-img, .footer-img {
  max-width: 100%;
  height: auto;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

st.title("Problem Definition")

# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("Banner settings")
duo_height_px = st.sidebar.slider("Top banner max height (px)", min_value=120, max_value=400, value=220, step=10)
left_name  = st.sidebar.text_input("Left image filename",  value="Problem_definition_1.svg")
right_name = st.sidebar.text_input("Right image filename", value="Problem_definition_2.svg")

root_dir = Path(__file__).parent
left_path  = (root_dir / left_name).resolve()
right_path = (root_dir / right_name).resolve()

# Show status to avoid “blank” surprises
with st.sidebar.expander("Image file status", expanded=False):
    st.write(f"Left : `{left_path}`  —  **{'FOUND' if left_path.exists() else 'MISSING'}**")
    st.write(f"Right: `{right_path}` —  **{'FOUND' if right_path.exists() else 'MISSING'}**")

# ================== IMAGE HELPERS ==================
def _data_url(path: Path):
    """Return (mime, base64_data) or (None, None) if missing/unsupported."""
    if not path.exists(): 
        return None, None
    ext = path.suffix.lower()
    mime = {
        ".svg":"image/svg+xml",
        ".png":"image/png",
        ".jpg":"image/jpeg",
        ".jpeg":"image/jpeg",
        ".webp":"image/webp",
    }.get(ext)
    if not mime: 
        return None, None
    b64 = base64.b64encode(path.read_bytes()).decode()
    return mime, b64

def _img_html(path: Path, css_class: str, max_h: int | None = None):
    """Build an <img> tag with data URL; keeps SVG vector. Returns '' if file missing."""
    mime, b64 = _data_url(path)
    if not mime: 
        return ""
    style = ""
    if max_h:
        style = f"max-height:{max_h}px;"
    return f'<img class="{css_class}" src="data:{mime};base64,{b64}" alt="{path.name}" style="{style}">'

# ================== TOP BANNER (ALWAYS ABOVE INPUTS) ==================
def render_top_banner(left: Path, right: Path, max_height_px: int):
    l_html = _img_html(left,  "duo-img",    max_h=max_height_px)
    r_html = _img_html(right, "duo-img",    max_h=max_height_px)

    if not l_html and not r_html:
        st.warning("Top banner: both images are missing or unsupported. Check filenames in the sidebar.")
        return

    # Responsive columns: side-by-side on wide, stack on narrow automatically
    c1, c2 = st.columns(2, gap="large")
    if l_html:
        with c1: st.markdown(l_html, unsafe_allow_html=True)
    else:
        with c1: st.info("Left image missing.")
    if r_html:
        with c2: st.markdown(r_html, unsafe_allow_html=True)
    else:
        with c2: st.info("Right image missing.")
    st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:10px 0 0 0;'/>", unsafe_allow_html=True)

render_top_banner(left_path, right_path, duo_height_px)

# ================== INPUT FORM ==================
st.markdown("### Input Parameters")
c1, c2 = st.columns(2)

with c1:
    st.latex(r"\frac{L}{D}")
    L_over_D = st.number_input("L_D", label_visibility="collapsed", format="%.4f", key="L_D")
    st.latex(r"r_e")
    r_e = st.number_input("r_e", label_visibility="collapsed", format="%.4f", key="r_e")

with c2:
    st.latex(r"\frac{V}{V_0}")
    V_over_V0 = st.number_input("V_V0", label_visibility="collapsed", min_value=0.0, max_value=1.0, format="%.4f", key="V_V0")
    st.latex(r"\beta")
    beta = st.number_input("beta", label_visibility="collapsed", format="%.4f", key="beta")

# ================== SESSION STATE ==================
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame(columns=["L/D","r_e","V/V0","beta","H/suTCA","M/suTCAD"])

# ================== PREDICT ==================
if st.button("🔮 Predict"):
    try:
        dfX = pd.DataFrame([[L_over_D, r_e, V_over_V0, beta]], columns=["L/D","r_e","V/V0","beta"])
        X_scaled = scalerX.transform(dfX) if scalerX is not None else dfX.values
        y_pred_norm = model.predict(X_scaled)
        y_pred = scalerY.inverse_transform(y_pred_norm)[0] if scalerY is not None else y_pred_norm[0]

        st.markdown("### Prediction Results")
        st.latex(rf"H/s_{{uTCA}}: \; {y_pred[0]:.4f}")
        st.latex(rf"M/s_{{uTCAD}}: \; {y_pred[1]:.4f}")

        new_row = {"L/D":L_over_D, "r_e":r_e, "V/V0":V_over_V0, "beta":beta,
                   "H/suTCA":y_pred[0], "M/suTCAD":y_pred[1]}
        st.session_state.results = pd.concat([st.session_state.results, pd.DataFrame([new_row])], ignore_index=True)
    except Exception as e:
        st.error(f"Errors: {e}")

# ================== RESULTS TABLE ==================
def render_results_table_white(df: pd.DataFrame):
    rename_map = {"L/D":"L/D","r_e":"rₑ","V/V0":"V/V₀","beta":"β","H/suTCA":"H/suTCA","M/suTCAD":"M/suTCAD"}
    df_show = df.rename(columns=rename_map).copy()

    df_fmt = df_show.copy()
    for col in ["H/suTCA","M/suTCAD"]:
        if col in df_fmt.columns:
            df_fmt[col] = pd.to_numeric(df_fmt[col], errors="coerce").map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    for col in ["L/D","rₑ","V/V₀","β"]:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda v: (f"{v:.10g}" if isinstance(v,(int,float,np.floating)) else str(v)))

    styled = (df_fmt.style.hide(axis="index")
              .set_table_styles([
                  {"selector":"table","props":"border-collapse:collapse;width:100%;background:#fff;color:#000;font-size:15px;"},
                  {"selector":"th","props":"background:#fff;color:#000;text-align:center;font-weight:700;font-size:16px;padding:10px 12px;border-bottom:2px solid rgba(0,0,0,.25);"},
                  {"selector":"td","props":"background:#fff;color:#000;text-align:center;padding:10px 12px;border-bottom:1px solid rgba(0,0,0,.12);"},
              ]))
    st.markdown(styled.to_html(), unsafe_allow_html=True)

if not st.session_state.results.empty:
    st.markdown("### Table of Results")
    render_results_table_white(st.session_state.results)
    csv_bytes = st.session_state.results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv_bytes, "prediction_results.csv", "text/csv")

