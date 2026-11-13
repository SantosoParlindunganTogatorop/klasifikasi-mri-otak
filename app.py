import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

# Pengaturan model 
MODEL_PATH = "model_v3.keras"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  
IMG_SIZE = (128, 128) 
LABELS = {"glioma": "Glioma", "meningioma": "Meningioma",
          "notumor": "No Tumor", "pituitary": "Pituitary"}

st.set_page_config(page_title="Brain Tumor MRI Detection", page_icon="🧠", layout="wide")


@st.cache_resource  
def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict(model, img: Image.Image):
    x = np.asarray(img.convert("L").resize(IMG_SIZE), dtype="float32")[None, ..., None]  
    return model.predict(x, verbose=0)[0]


# Tampilan 
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: linear-gradient(180deg, #e6f0fd 0%, #f6f9ff 45%, #f3f7fd 100%); }
header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
.block-container { max-width: 1120px; padding-top: 1.6rem; padding-bottom: 2rem; }

/* Hero */
.hero { display: flex; align-items: center; gap: 28px; padding: 30px 36px; border-radius: 26px; color: #fff;
  background: radial-gradient(circle at 12% 40%, rgba(80,170,255,.35), transparent 38%),
              linear-gradient(115deg, #06224f 0%, #0b3c82 55%, #2a8fdc 100%);
  box-shadow: 0 10px 30px rgba(11,60,130,.25); }
.hero .brain { font-size: 92px; line-height: 1; filter: hue-rotate(185deg) saturate(1.5) brightness(1.15) drop-shadow(0 0 18px rgba(90,190,255,.95)); }
.hero .sep { width: 1px; align-self: stretch; background: rgba(255,255,255,.22); }
.hero .badge { display: inline-block; background: #1478d4; color: #fff; font-size: 14px; font-weight: 600;
  padding: 6px 16px; border-radius: 999px; margin-bottom: 12px; }
.hero h1 { color: #fff; font-size: 40px; font-weight: 800; margin: 0 0 8px 0; padding: 0; line-height: 1.15; }
.hero p { color: #dbe9fb; font-size: 16.5px; margin: 0; max-width: 620px; }

/* Kartu */
.st-key-upload_card, .st-key-result_card, .st-key-chart_card {
  background: #fff; border: 1px solid #e3ebf7 !important; border-radius: 22px;
  box-shadow: 0 6px 22px rgba(30,70,140,.08); padding: 8px 10px 20px 10px; }
.sec { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
.sec .ic { width: 52px; height: 52px; border-radius: 50%; background: #0b4fb3; color: #fff;
  display: flex; align-items: center; justify-content: center; font-size: 24px; flex: none; }
.sec .t { font-size: 24px; font-weight: 800; color: #0b2a5b; line-height: 1.2; }
.sec .s { font-size: 15px; color: #5b6b85; }
.sec.small .ic { width: 40px; height: 40px; font-size: 18px; }
.sec.small .t { font-size: 21px; }

/* Uploader */
[data-testid="stFileUploaderDropzone"] { background: #f5f8ff; border: 2px dashed #bccae4; border-radius: 16px; padding: 28px 16px; }
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderDropzone"] * { color: #20324f !important; }
[data-testid="stFileUploaderDropzone"] button { background: #0b63d1 !important; border: none !important; border-radius: 999px !important;
  padding: 10px 26px !important; font-weight: 700; }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color: #fff !important; }
[data-testid="stFileUploaderFile"] { background: #fff !important; border-radius: 10px; }
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] * { color: #20324f !important; }
[data-testid="stFileUploaderFile"] button, [data-testid="stFileUploaderFile"] button * { color: #fff !important; }
.hint { text-align: center; color: #6b7a94; font-size: 13px; margin-top: 8px; }

/* Hasil */
.res { display: flex; align-items: center; gap: 26px; padding: 26px 28px; border-radius: 18px; min-height: 190px;
  background: linear-gradient(120deg, #d6f6f3 0%, #e4f4fb 70%, #d5eaf8 100%); }
.res.warn { background: linear-gradient(120deg, #ffe9d6 0%, #fdf0e6 70%, #fbe6e6 100%); }
.res .brain { font-size: 78px; line-height: 1; filter: hue-rotate(185deg) saturate(1.2); }
.res.warn .brain { filter: none; }
.res .sep { width: 1px; align-self: stretch; background: #0b4fb3; opacity: .55; }
.res .lbl { font-size: 17px; color: #25364f; margin-top: 4px; }
.res .big { font-size: 38px; font-weight: 800; color: #0b2a5b; line-height: 1.15; margin-bottom: 14px; }
.res .conf { font-size: 44px; font-weight: 800; color: #12a39a; line-height: 1.1; }
.res.warn .conf { color: #d9620f; }
.empty { display: flex; align-items: center; justify-content: center; text-align: center; min-height: 190px;
  border-radius: 18px; background: #f5f8ff; color: #6b7a94; font-size: 16px; padding: 20px; }
.note { text-align: center; color: #6b7a94; font-size: 13px; margin-top: 18px; }
</style>
""", unsafe_allow_html=True)


def section(icon, title, sub="", small=False):
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    cls = "sec small" if small else "sec"
    return f'<div class="{cls}"><div class="ic">{icon}</div><div><div class="t">{title}</div>{sub_html}</div></div>'


# Hero 
st.markdown("""
<div class="hero">
  <div class="brain">🧠</div>
  <div class="sep"></div>
  <div>
    <span class="badge">AI Powered • Medical Imaging</span>
    <h1>Brain Tumor MRI Detection</h1>
    <p>An intelligent application to classify brain MRI images into four categories:
       Glioma, Meningioma, Pituitary, or No Tumor.</p>
  </div>
</div>
""", unsafe_allow_html=True)
st.write("")

# Baris atas: upload + hasil 
col_up, col_res = st.columns(2, gap="large")

with col_up:
    with st.container(border=True, key="upload_card"):
        st.markdown(section("☁️", "Upload MRI Image", "Choose a brain MRI image from your device"),
                    unsafe_allow_html=True)
        file = st.file_uploader("MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown('<div class="hint">Supported formats: JPG, JPEG, PNG</div>', unsafe_allow_html=True)
        img = None
        if file:
            img = Image.open(file)
            st.image(img, width=200)

probs = None
if img is not None:
    probs = predict(load_model(), img)

with col_res:
    with st.container(border=True, key="result_card"):
        st.markdown(section("📊", "Prediction Result"), unsafe_allow_html=True)
        if probs is None:
            st.markdown('<div class="empty">Upload an MRI image to see the prediction.</div>',
                        unsafe_allow_html=True)
        else:
            i = int(probs.argmax())
            warn = "" if CLASS_NAMES[i] == "notumor" else " warn"
            st.markdown(f"""
            <div class="res{warn}">
              <div class="brain">🧠</div><div class="sep"></div>
              <div>
                <div class="lbl">Predicted Class</div>
                <div class="big">{LABELS[CLASS_NAMES[i]]}</div>
                <div class="lbl">Confidence Score</div>
                <div class="conf">{probs[i]:.1%}</div>
              </div>
            </div>""", unsafe_allow_html=True)

# Grafik probabilitas
st.write("")
with st.container(border=True, key="chart_card"):
    st.markdown(section("📈", "Prediction Probability", small=True), unsafe_allow_html=True)
    if probs is None:
        st.markdown('<div class="empty">The probability chart will appear here.</div>', unsafe_allow_html=True)
    else:
        top = LABELS[CLASS_NAMES[int(probs.argmax())]]
        df = pd.DataFrame({"Class": [LABELS[c] for c in CLASS_NAMES], "Probability": probs.astype(float)})
        base = alt.Chart(df).encode(
            x=alt.X("Class:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=14, labelColor="#44546f")))
        bars = base.mark_bar(size=90, cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                    axis=alt.Axis(format=".1f", labelColor="#44546f", gridColor="#e6ecf5")),
            color=alt.condition(alt.datum.Class == top, alt.value("#0a66d4"), alt.value("#bcd5f2")),
            tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")])
        labels = base.mark_text(dy=-9, fontSize=14, fontWeight="bold", color="#0b2a5b").encode(
            y="Probability:Q", text=alt.Text("Probability:Q", format=".1%"))
        chart = (bars + labels).properties(height=300).configure_view(strokeWidth=0).configure(background="#ffffff")
        st.altair_chart(chart, use_container_width=True)

st.markdown('<div class="note">Dibuat untuk keperluan tugas kuliah — bukan alat diagnosis medis.</div>',
            unsafe_allow_html=True)
