# Brain Tumor MRI Detection (CNN from scratch)

Aplikasi Streamlit untuk mengklasifikasi citra MRI otak ke 4 kelas: glioma, meningioma, notumor, pituitary.
Model CNN dibangun dari nol tanpa transfer learning.

## Menjalankan secara lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Buka http://localhost:8501.

## Isi repositori
- `app.py` – aplikasi Streamlit
- `model_v3.keras` – model terlatih
- `requirements.txt` – daftar library
- `.streamlit/config.toml` – tema terang

> Hanya untuk keperluan tugas, bukan alat diagnosis medis.
