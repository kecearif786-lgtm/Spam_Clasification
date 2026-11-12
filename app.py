import streamlit as st
import joblib
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

st.set_page_config(page_title="Spam Detector AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {
 background:radial-gradient(circle at 8% 0%,rgba(56,189,248,.13),transparent 28%),
 radial-gradient(circle at 92% 8%,rgba(99,102,241,.14),transparent 30%),#0b1220;color:#e5edf8;
}
.block-container{max-width:1120px;padding-top:2rem;padding-bottom:3rem;}
.hero{padding:30px 32px;border:1px solid rgba(148,163,184,.2);border-radius:24px;
 background:linear-gradient(120deg,rgba(15,35,62,.96),rgba(24,38,77,.92));
 box-shadow:0 18px 55px rgba(0,0,0,.22);margin-bottom:24px;}
.eyebrow{color:#7dd3fc;font-size:12px;font-weight:800;letter-spacing:2px;}
.hero h1{color:#f8fafc;font-size:clamp(30px,4vw,46px);line-height:1.12;margin:10px 0;font-weight:800;}
.hero p{color:#c4d2e5;font-size:15px;margin:0;max-width:720px;}
.info-card{border:1px solid rgba(148,163,184,.18);border-radius:18px;padding:18px 20px;
 background:rgba(20,33,55,.82);min-height:105px;}
.info-card .big{font-size:25px;margin-bottom:4px;}
.info-card .label,.muted{color:#9fb1c9;font-size:13px;}
.section-title{color:#f1f5f9;font-weight:750;font-size:20px;margin:10px 0 5px;}
.result-card{border-radius:18px;padding:22px 24px;margin-top:18px;border:1px solid;}
.result-spam{background:rgba(127,29,29,.22);border-color:rgba(248,113,113,.55);}
.result-ham{background:rgba(6,78,59,.24);border-color:rgba(52,211,153,.48);}
.result-title{font-size:24px;font-weight:800;margin-bottom:5px;}
.result-desc{color:#cbd5e1;font-size:14px;}
.stTextArea textarea{background:#111d30!important;color:#f8fafc!important;border:1px solid #34445e!important;border-radius:12px!important;}
.stButton>button{border:0;border-radius:11px;min-height:44px;font-weight:800;
 background:linear-gradient(90deg,#0284c7,#4f46e5);color:white;}
.stButton>button:hover{color:white;filter:brightness(1.12);}
[data-testid="stFileUploader"]{background:rgba(17,29,48,.72);border:1px dashed #526783;border-radius:14px;padding:12px;}
div[data-testid="stTabs"] button{font-weight:700;font-size:15px;}
hr{border-color:rgba(148,163,184,.18);}
footer,#MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl"), joblib.load("tfidf_vectorizer.pkl")

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error("Model belum dapat dimuat. Pastikan spam_model.pkl dan tfidf_vectorizer.pkl berada satu folder dengan app.py.")
    st.exception(e)
    st.stop()

def extract_text(image_file):
    image = Image.open(image_file).convert("RGB")
    image = ImageOps.grayscale(image)
    if image.width < 1000:
        scale = 1000 / image.width
        image = image.resize((1000, int(image.height * scale)))
    image = ImageEnhance.Contrast(image).enhance(1.7)
    image = ImageEnhance.Sharpness(image).enhance(1.3)
    text = pytesseract.image_to_string(image, lang="ind+eng", config="--psm 6").strip()
    return image, text

def identify_message(text, source_label):
    text = (text or "").strip()
    if not text:
        st.warning("Masukkan atau pastikan teks pesan sudah terbaca sebelum melakukan deteksi.")
        return

    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized)[0]
        classes = list(model.classes_)
        confidence = float(probs[classes.index(prediction)]) * 100

    confidence_text = f"{confidence:.2f}%" if confidence is not None else "Tidak tersedia"
    st.markdown(f'<div class="muted">HASIL ANALISIS • {source_label.upper()}</div>', unsafe_allow_html=True)
    if int(prediction) == 1:
        st.markdown(f"""<div class="result-card result-spam">
        <div class="result-title">🚨 TERDETEKSI SPAM</div>
        <div class="result-desc">Pesan ini diprediksi sebagai pesan spam atau mencurigakan.</div>
        <div style="margin-top:15px;color:#fecaca;font-size:14px;">Tingkat keyakinan model: <b>{confidence_text}</b></div>
        </div>""", unsafe_allow_html=True)
        st.progress(min(max((confidence or 0)/100, 0.0), 1.0))
        st.info("Saran: berhati-hatilah terhadap tautan, permintaan uang, OTP, PIN, atau data pribadi.")
    else:
        st.markdown(f"""<div class="result-card result-ham">
        <div class="result-title">✅ TERDETEKSI HAM</div>
        <div class="result-desc">Pesan ini diprediksi sebagai pesan normal atau bukan spam.</div>
        <div style="margin-top:15px;color:#a7f3d0;font-size:14px;">Tingkat keyakinan model: <b>{confidence_text}</b></div>
        </div>""", unsafe_allow_html=True)
        st.progress(min(max((confidence or 0)/100, 0.0), 1.0))
        st.info("Pesan diprediksi bukan spam, tetapi tetap periksa konteks dan pengirimnya.")

st.markdown("""
<div class="hero">
 <div class="eyebrow">ARTIFICIAL INTELLIGENCE • TEXT CLASSIFICATION</div>
 <h1>🛡️ Spam Detector AI</h1>
 <p>Periksa pesan mencurigakan dengan mudah. Ketik teks secara langsung atau unggah screenshot,
 lalu biarkan model AI menganalisis apakah pesan termasuk SPAM atau HAM.</p>
</div>
""", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown('<div class="info-card"><div class="big">⌨️</div><b>Input Teks</b><div class="label">Tempel atau ketik pesan langsung</div></div>',unsafe_allow_html=True)
with c2:
    st.markdown('<div class="info-card"><div class="big">📷</div><b>Screenshot OCR</b><div class="label">Ambil teks dari gambar pesan</div></div>',unsafe_allow_html=True)
with c3:
    st.markdown('<div class="info-card"><div class="big">🤖</div><b>Prediksi AI</b><div class="label">Klasifikasi SPAM atau HAM</div></div>',unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
st.markdown('<div class="section-title">Mulai pemeriksaan pesan</div><div class="muted">Pilih salah satu metode input di bawah ini.</div>',unsafe_allow_html=True)

tab_text,tab_image = st.tabs(["⌨️  Input Teks","📷  Upload Screenshot"])
with tab_text:
    st.markdown("#### Masukkan isi pesan")
    st.caption("Salin pesan SMS, WhatsApp, atau pesan lainnya ke kotak berikut.")
    manual_text = st.text_area("Teks pesan",placeholder="Contoh: Selamat! Anda mendapatkan hadiah. Klik tautan berikut...",height=170,label_visibility="collapsed",key="manual_text")
    if st.button("🔍  Analisis Pesan",type="primary",use_container_width=True,key="check_manual"):
        identify_message(manual_text,"Input teks")

with tab_image:
    st.markdown("#### Unggah screenshot pesan")
    st.caption("Format yang didukung: PNG, JPG, JPEG, dan WEBP.")
    uploaded_image = st.file_uploader("Pilih gambar screenshot",type=["png","jpg","jpeg","webp"],label_visibility="collapsed",key="uploaded_image")
    if uploaded_image is not None:
        try:
            preview, detected_text = extract_text(uploaded_image)
            left,right = st.columns([1,1.15],gap="large")
            with left:
                st.image(preview,caption="Pratinjau gambar",use_container_width=True)
            with right:
                st.markdown("**Teks yang berhasil dibaca (OCR)**")
                edited_text = st.text_area("Periksa dan koreksi teks jika ada kesalahan pembacaan",value=detected_text,height=190,key="ocr_result")
                if st.button("🔍  Analisis Screenshot",type="primary",use_container_width=True,key="check_image"):
                    identify_message(edited_text,"Screenshot OCR")
            if not detected_text:
                st.warning("Tulisan belum berhasil dibaca. Coba unggah gambar yang lebih jelas atau ketik teks secara manual.")
        except pytesseract.pytesseract.TesseractNotFoundError:
            st.error("Mesin OCR Tesseract belum tersedia. Pastikan dependensi Tesseract sudah dipasang pada lingkungan deployment.")
        except Exception as e:
            st.error(f"Gambar gagal diproses: {e}")

st.markdown("<br><hr>",unsafe_allow_html=True)
st.markdown("""<div style="text-align:center;color:#8193ad;font-size:12px;padding:8px 0;">
Spam Detector AI • TF-IDF + Multinomial Naive Bayes + OCR<br>
Hasil merupakan prediksi model dan tidak menjamin keamanan pesan.</div>""",unsafe_allow_html=True)
