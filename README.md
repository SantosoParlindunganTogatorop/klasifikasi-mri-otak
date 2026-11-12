# Klasifikasi Citra MRI Tumor Otak (CNN from scratch)

Model CNN dibangun dari nol tanpa transfer learning untuk 4 kelas: glioma, meningioma, notumor, pituitary.
Aplikasi web dibuat dengan Streamlit.

## Menjalankan secara lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Lalu buka http://localhost:8501.

## Isi repositori
- `app.py` – aplikasi Streamlit
- `model_v3.keras` – model terlatih
- `requirements.txt` – daftar library

> Hanya untuk keperluan tugas, bukan alat diagnosis medis.
