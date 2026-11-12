import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

MODEL_PATH = "model_v3.keras"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # urutan abjad, sama seperti saat training
IMG_SIZE = (128, 128)  # (lebar, tinggi); samakan dengan IMG_SIZE saat training


@st.cache_resource  # model dimuat sekali saja, bukan setiap interaksi
def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict(model, img: Image.Image):
    x = np.asarray(img.convert("L").resize(IMG_SIZE), dtype="float32")[None, ..., None]  # skala 0-255
    return model.predict(x, verbose=0)[0]


st.set_page_config(page_title="Klasifikasi MRI Tumor Otak", page_icon="🧠")
st.title("🧠 Klasifikasi Citra MRI Tumor Otak")
st.caption("CNN dari nol (tanpa transfer learning) untuk 4 kelas: glioma, meningioma, notumor, pituitary. "
           "Proyek tugas kuliah, bukan alat diagnosis medis.")

file = st.file_uploader("Unggah citra MRI (jpg/png)", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Citra yang diunggah", use_container_width=True)
    p = predict(load_model(), img)
    i = int(p.argmax())
    st.success(f"Prediksi: **{CLASS_NAMES[i]}** ({p[i]:.1%})")
    st.bar_chart({c: float(v) for c, v in zip(CLASS_NAMES, p)})
