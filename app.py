import streamlit as st
import joblib
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

# Memuat model dan TF-IDF yang sudah dilatih
@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

st.set_page_config(page_title="Detektor Pesan Spam AI", page_icon="🛡️")
st.title("🛡️ Detektor Pesan Spam AI")
st.write(
    "Periksa pesan melalui teks yang diketik atau screenshot. "
    "Sistem membaca screenshot dengan OCR, lalu memeriksanya menggunakan model AI."
)

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(
        "Model tidak berhasil dimuat. Pastikan file "
        "`spam_model.pkl` dan `tfidf_vectorizer.pkl` tersedia "
        "di repository GitHub, satu folder dengan `app.py`."
    )
    st.exception(e)
    st.stop()


def extract_text(image_file):
    image = Image.open(image_file).convert("RGB")
    image = ImageOps.grayscale(image)

    # Memperbesar gambar kecil dan meningkatkan kontras untuk membantu OCR
    if image.width < 1400:
        ratio = 1400 / image.width
        image = image.resize((1400, int(image.height * ratio)))
    image = ImageEnhance.Contrast(image).enhance(1.7)

    # Membaca teks bahasa Indonesia dan Inggris
    return image, pytesseract.image_to_string(
        image, lang="ind+eng", config="--psm 6"
    ).strip()


tab_text, tab_image = st.tabs(["⌨️ Input Teks", "📷 Upload Screenshot"])

with tab_text:
    manual_text = st.text_area(
        "Teks Pesan:",
        placeholder="Contoh: Selamat! Anda terpilih mendapatkan hadiah uang tunai 10 juta rupiah...",
        height=150,
        key="manual_text"
    )
    check_manual = st.button(
        "Identifikasi Pesan", type="primary", key="check_manual"
    )

with tab_image:
    uploaded_image = st.file_uploader(
        "Pilih screenshot pesan",
        type=["png", "jpg", "jpeg", "webp"],
        help="Gunakan gambar dengan tulisan yang jelas dan tidak terpotong."
    )

    ocr_text = ""
    if uploaded_image is not None:
        try:
            preview, ocr_text = extract_text(uploaded_image)
            st.image(preview, caption="Screenshot yang diunggah", use_container_width=True)
            st.write("**Teks yang berhasil dibaca:**")
            if ocr_text:
                ocr_text = st.text_area(
                    "Periksa atau koreksi teks hasil OCR sebelum deteksi:",
                    value=ocr_text,
                    height=150,
                    key="ocr_result"
                )
            else:
                st.warning(
                    "Belum ada teks yang terbaca. Coba gambar yang lebih jelas "
                    "atau gunakan input teks manual."
                )
        except pytesseract.pytesseract.TesseractNotFoundError:
            st.error(
                "Tesseract OCR belum tersedia di server. Periksa file "
                "`packages.txt` dan tunggu deployment selesai."
            )
        except Exception as e:
            st.error(f"Gambar gagal diproses: {e}")

    check_image = st.button(
        "Deteksi dari Screenshot", type="primary", key="check_image"
    )


def identify_message(text):
    if not text or not text.strip():
        st.warning("Mohon masukkan teks atau unggah screenshot yang berisi pesan.")
        return

    try:
        text_vectorized = vectorizer.transform([text.strip()])
        prediction = model.predict(text_vectorized)[0]

        # Mengambil probabilitas sesuai label kelas, bukan mengasumsikan urutan kelas.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vectorized)[0]
            classes = list(model.classes_)
            class_index = classes.index(prediction)
            confidence = round(float(probabilities[class_index]) * 100, 2)
            confidence_text = f" (Tingkat Keyakinan: {confidence}%)"
        else:
            confidence_text = ""

        # Mengikuti aturan pada aplikasi awal: label 1 berarti SPAM.
        if prediction == 1:
            st.error(f"🚨 **Hasil Prediksi: SPAM**{confidence_text}")
        else:
            st.success(f"✅ **Hasil Prediksi: HAM / Bukan Spam**{confidence_text}")

        st.caption(
            "Hasil ini merupakan prediksi model, bukan jaminan keamanan. "
            "Jangan membagikan OTP, PIN, atau kata sandi dan jangan membuka "
            "tautan mencurigakan hanya berdasarkan hasil deteksi."
        )
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")


# Tombol pada masing-masing tab memproses sumber teks yang sesuai.
if check_manual:
    identify_message(manual_text)

if check_image:
    identify_message(ocr_text)
