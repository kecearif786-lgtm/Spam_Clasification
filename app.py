from __future__ import annotations

import csv
import hashlib
import hmac
import io
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytesseract
import streamlit as st
from supabase import create_client
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
OCR_LANGUAGES = "ind+eng"
MAX_TEXT_LENGTH = 20_000
SUPABASE_TABLE = "spam_analysis_history"
HISTORY_PAGE_SIZE = 500
DISPLAY_TIMEZONE = ZoneInfo("Asia/Jakarta")

st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --ink: #17243b;
            --muted: #718096;
            --line: #e8edf5;
            --surface: #ffffff;
            --canvas: #f5f7fb;
            --blue: #3968e8;
            --blue-dark: #244dc5;
            --teal: #13a68b;
            --red: #db5364;
            --shadow: 0 12px 34px rgba(26, 43, 75, .07);
        }

        html, body, [class*="css"] {
            font-family: 'DM Sans', 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(245,247,251,.84); }
        .block-container { max-width: 1440px; padding: 1.7rem 2.3rem 3rem; }
        [data-testid="stSidebar"] {
            background: #101b31;
            border-right: 1px solid rgba(255,255,255,.06);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
        [data-testid="stSidebar"] * { color: #d8e2f2; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 11px; padding: .5rem .65rem; margin: .12rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.07); }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-weight: 600; font-size: .93rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div { color: white; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #9eacc3; }

        h1, h2, h3, h4 { color: var(--ink); letter-spacing: -.035em; }
        h1 { font-family: 'Manrope', 'DM Sans', sans-serif; font-weight: 800; }
        h2, h3 { font-weight: 800; }
        p, li { color: #4e5e75; }
        [data-testid="stCaptionContainer"] p { color: var(--muted); }
        hr { border-color: var(--line); }

        .brand-lockup { display:flex; align-items:center; gap:11px; padding: .1rem .2rem 1.35rem; }
        .brand-mark {
            display:flex; align-items:center; justify-content:center; width:42px; height:42px;
            border-radius:14px; background:linear-gradient(140deg,#4b7cff,#6858dc);
            color:white; font-size:21px; box-shadow:0 8px 20px rgba(64,104,225,.3);
        }
        .brand-name { color:#fff; font-size:1.08rem; font-weight:800; letter-spacing:-.03em; }
        .brand-sub { color:#91a3c0; font-size:.68rem; letter-spacing:.13em; text-transform:uppercase; margin-top:1px; }
        .sidebar-note { color:#8494af; font-size:.76rem; line-height:1.55; padding:.6rem .25rem; }

        .page-kicker { color:var(--blue); font-size:.71rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.45rem; }
        .page-title { font-family:'Manrope','DM Sans',sans-serif; color:#17243b; font-size:clamp(1.8rem,3vw,2.55rem); font-weight:800; letter-spacing:-.055em; line-height:1.12; margin:0; }
        .page-subtitle { color:#75839a; font-size:.98rem; max-width:780px; margin-top:.65rem; line-height:1.65; }
        .hero {
            position:relative; overflow:hidden; padding:2.0rem 2.1rem; border-radius:23px;
            color:#fff; background:linear-gradient(112deg,#172c54 0%,#294b92 58%,#4e58b8 100%);
            box-shadow:0 18px 42px rgba(39,67,132,.18); margin-bottom:1.35rem;
        }
        .hero:after { content:''; position:absolute; width:300px; height:300px; border-radius:50%; right:-70px; top:-155px; border:1px solid rgba(255,255,255,.17); box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 70px rgba(255,255,255,.025); }
        .hero-kicker { color:#b8d1ff; font-size:.7rem; font-weight:800; letter-spacing:.17em; text-transform:uppercase; }
        .hero-title { color:#fff; font-family:'Manrope','DM Sans',sans-serif; font-size:clamp(2rem,4vw,3rem); font-weight:800; letter-spacing:-.06em; line-height:1.08; margin:.65rem 0 .65rem; }
        .hero-copy { color:#d6e1f7; font-size:.97rem; line-height:1.65; max-width:660px; margin:0; }
        .hero-pill { display:inline-flex; align-items:center; gap:7px; margin-top:1.25rem; padding:7px 11px; border:1px solid rgba(255,255,255,.19); border-radius:999px; background:rgba(255,255,255,.09); color:#e9f0ff; font-size:.75rem; font-weight:700; }

        .surface-card, .metric-card {
            background:var(--surface); border:1px solid var(--line); border-radius:17px;
            box-shadow:var(--shadow); padding:1.15rem 1.2rem;
        }
        .metric-card { min-height:128px; }
        .metric-top { display:flex; justify-content:space-between; align-items:center; gap:10px; }
        .metric-label { color:#75839a; font-size:.81rem; font-weight:600; }
        .metric-icon { display:flex; align-items:center; justify-content:center; width:36px; height:36px; border-radius:11px; background:#edf2ff; font-size:1.05rem; }
        .metric-value { color:#17243b; font-size:1.9rem; font-weight:800; letter-spacing:-.06em; margin-top:.6rem; line-height:1.1; }
        .metric-foot { color:#93a0b3; font-size:.72rem; margin-top:.35rem; }
        .section-heading { color:#24334c; font-size:1.12rem; font-weight:800; letter-spacing:-.025em; margin:1.15rem 0 .2rem; }
        .section-description { color:#8490a3; font-size:.83rem; margin-bottom:.8rem; }
        .soft-panel { background:#fff; border:1px solid var(--line); border-radius:17px; padding:1.2rem; box-shadow:var(--shadow); }
        .feature-icon { width:42px; height:42px; display:flex; align-items:center; justify-content:center; border-radius:13px; background:#edf2ff; font-size:1.25rem; margin-bottom:.75rem; }
        .feature-title { font-weight:800; color:#293850; font-size:.95rem; margin-bottom:.25rem; }
        .feature-copy { color:#7a879a; font-size:.81rem; line-height:1.55; }
        .status-chip { display:inline-block; padding:5px 9px; border-radius:999px; font-size:.7rem; font-weight:800; letter-spacing:.04em; }
        .chip-spam { background:#fff0f1; color:#c33d50; border:1px solid #ffd7dc; }
        .chip-ham { background:#eaf8f4; color:#13856f; border:1px solid #c7eee2; }
        .result-panel { border-radius:17px; padding:1.3rem 1.4rem; margin:.8rem 0 1rem; border:1px solid; }
        .result-panel.spam { background:#fff5f5; border-color:#f5c7cd; }
        .result-panel.ham { background:#f0faf6; border-color:#bce8d8; }
        .result-heading { font-size:1.25rem; font-weight:800; letter-spacing:-.03em; margin-bottom:.25rem; }
        .result-panel.spam .result-heading { color:#b9384b; }
        .result-panel.ham .result-heading { color:#13836d; }
        .result-copy { font-size:.86rem; color:#69788e; line-height:1.55; }
        .result-confidence { color:#34445d; font-size:.84rem; margin-top:.85rem; }
        .confidence-track { height:8px; border-radius:99px; background:#e8edf4; overflow:hidden; margin-top:.5rem; }
        .confidence-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#4a79ed,#18a88c); }
        .notice { border-radius:12px; background:#f1f5ff; border:1px solid #dbe5ff; padding:.8rem 1rem; color:#526681; font-size:.82rem; line-height:1.55; }
        .step-number { display:inline-flex; width:25px; height:25px; align-items:center; justify-content:center; border-radius:8px; background:#edf2ff; color:#3c65d8; font-weight:800; font-size:.78rem; margin-right:8px; }
        .empty-state { padding:2rem 1rem; text-align:center; border:1px dashed #d8e0ec; border-radius:16px; background:rgba(255,255,255,.65); }
        .empty-state-icon { font-size:2rem; margin-bottom:.45rem; }
        .empty-state-title { color:#34445d; font-weight:800; }
        .empty-state-copy { color:#8693a6; font-size:.82rem; margin-top:.2rem; }
        .app-footer { text-align:center; color:#9aa5b5; font-size:.73rem; padding:2rem .5rem .4rem; }

        .stButton > button, .stDownloadButton > button {
            border-radius:11px; border:1px solid #dce4f0; min-height:43px; font-weight:700;
            color:#34445d; background:#fff; transition:all .18s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color:#9eb5f4; color:#315fda; background:#f8faff; transform:translateY(-1px); }
        .stButton > button[kind="primary"] {
            background:linear-gradient(105deg,#3d6ce9,#5b5ad9); color:#fff; border:0;
            box-shadow:0 7px 16px rgba(65,101,218,.2);
        }
        .stButton > button[kind="primary"]:hover { color:#fff; filter:brightness(1.06); box-shadow:0 9px 20px rgba(65,101,218,.27); }
        .stTextArea textarea, .stTextInput input {
            background:#fbfcfe !important; color:#26364e !important;
            border:1px solid #dce4ef !important; border-radius:12px !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus { border-color:#7e9bf0 !important; box-shadow:0 0 0 2px rgba(76,112,226,.10) !important; }
        [data-testid="stFileUploader"] { background:#fbfcff; border:1px dashed #c9d5e8; border-radius:14px; padding:.6rem; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem; }
        [data-testid="stMetricLabel"] { color:#76849a; }
        .stProgress > div > div > div > div { background:linear-gradient(90deg,#4b77ee,#17a58a); }
        div[data-testid="stAlert"] { border-radius:12px; }
        footer, #MainMenu { visibility:hidden; }

        @media (max-width: 768px) {
            .block-container { padding:1.2rem 1rem 2rem; }
            .hero { padding:1.5rem; border-radius:18px; }
            .metric-card { min-height:112px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()


# -----------------------------------------------------------------------------
# Model and OCR utilities
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any, Any]:
    """Load the original trained artifacts without changing or re-saving them."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        missing = [p.name for p in (MODEL_PATH, VECTORIZER_PATH) if not p.exists()]
        raise FileNotFoundError(
            "File model belum ditemukan: " + ", ".join(missing)
            + ". Letakkan kedua file .pkl di folder yang sama dengan app.py."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    if not hasattr(vectorizer, "transform") or not hasattr(model, "predict"):
        raise TypeError("Isi file model tidak sesuai. Pastikan vectorizer memiliki transform() dan model memiliki predict().")
    return model, vectorizer


def configure_tesseract() -> None:
    """Allow an optional executable path to be supplied through the environment."""
    executable = os.getenv("TESSERACT_CMD", "").strip()
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable


def image_to_png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def extract_text(image_file: Any) -> tuple[Image.Image, str]:
    """Preprocess an uploaded screenshot and extract Indonesian/English text."""
    image_file.seek(0)
    with Image.open(image_file) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    image = ImageOps.grayscale(image)
    if image.width < 1000:
        scale = 1000 / image.width
        image = image.resize((1000, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.7)
    image = ImageEnhance.Sharpness(image).enhance(1.3)
    text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES, config="--psm 6").strip()
    return image, text


def classify_text(text: str, model: Any, vectorizer: Any) -> dict[str, Any]:
    """Predict a message using the existing TF-IDF and Naive Bayes artifacts."""
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("Masukkan teks pesan terlebih dahulu.")
    if len(clean_text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Teks terlalu panjang. Batas maksimal adalah {MAX_TEXT_LENGTH:,} karakter.")

    vectorized = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized)[0]

    # Preserve the original application's encoded-label convention: 1 = SPAM, 0 = HAM.
    predicted_label = str(prediction).strip().lower()
    if predicted_label in {"spam", "ham"}:
        label = predicted_label.upper()
    else:
        try:
            label = "SPAM" if int(prediction) == 1 else "HAM"
        except (TypeError, ValueError):
            raise ValueError(
                f"Label prediksi '{prediction}' tidak dikenali. Model harus menghasilkan label 1/0 atau SPAM/HAM."
            )

    confidence: float | None = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized)[0]
        classes = list(model.classes_)
        predicted_index = classes.index(prediction)
        confidence = float(probabilities[predicted_index]) * 100
        confidence = min(100.0, max(0.0, confidence))

    return {
        "label": label,
        "confidence": confidence,
        "text": clean_text,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def ensure_state() -> None:
    """Initialize only per-browser UI state; the shared history lives in Supabase."""
    defaults: dict[str, Any] = {
        "last_result": None,
        "manual_input": "",
        "ocr_editor": "",
        "ocr_fingerprint": None,
        "ocr_processed_image": None,
        "ocr_detected": False,
        "ocr_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _supabase_settings() -> tuple[str, str]:
    """Read Supabase credentials from Streamlit secrets or local environment."""
    section: Any = {}
    root: Any = {}
    try:
        root = st.secrets
        section = root.get("supabase", {})
    except Exception:
        root, section = {}, {}

    url = (
        os.getenv("SUPABASE_URL")
        or section.get("url")
        or root.get("SUPABASE_URL", "")
    )
    secret_key = (
        os.getenv("SUPABASE_SECRET_KEY")
        or section.get("secret_key")
        or section.get("service_role_key")
        or root.get("SUPABASE_SECRET_KEY", "")
        or root.get("SUPABASE_SERVICE_ROLE_KEY", "")
    )
    if not url or not secret_key:
        raise RuntimeError(
            "Konfigurasi Supabase belum lengkap. Tambahkan [supabase].url dan "
            "[supabase].secret_key pada Streamlit Secrets."
        )
    if not str(url).startswith("https://"):
        raise RuntimeError("SUPABASE_URL harus berupa URL HTTPS project Supabase.")
    return str(url).strip(), str(secret_key).strip()


@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Any:
    """Create one server-side Supabase client. Never send its secret key to a browser."""
    url, secret_key = _supabase_settings()
    return create_client(url, secret_key)


def _normalise_history_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map database columns to the stable keys used by the Streamlit UI."""
    return {
        "id": row.get("id"),
        "created_at": row.get("created_at", ""),
        "source": row.get("source", "-"),
        "label": row.get("label", "-"),
        "confidence": row.get("confidence"),
        "text": row.get("message_text", ""),
    }


@st.cache_data(ttl=20, show_spinner=False)
def load_shared_history() -> list[dict[str, Any]]:
    """Fetch all shared records in pages; a short cache avoids repeat queries on reruns."""
    client = get_supabase_client()
    records: list[dict[str, Any]] = []
    offset = 0

    while True:
        response = (
            client.table(SUPABASE_TABLE)
            .select("id, created_at, source, label, confidence, message_text")
            .order("id", desc=True)
            .range(offset, offset + HISTORY_PAGE_SIZE - 1)
            .execute()
        )
        batch = response.data or []
        records.extend(_normalise_history_row(row) for row in batch)
        if len(batch) < HISTORY_PAGE_SIZE:
            break
        offset += HISTORY_PAGE_SIZE

    return records


def save_analysis(result: dict[str, Any], source: str) -> dict[str, Any]:
    """Persist one analysis in the global history table and return its saved row."""
    client = get_supabase_client()
    payload = {
        "source": source,
        "label": result["label"],
        "confidence": result.get("confidence"),
        "message_text": result["text"],
    }
    response = (
        client.table(SUPABASE_TABLE)
        .insert(payload)
        .select("id, created_at, source, label, confidence, message_text")
        .execute()
    )
    if not response.data:
        raise RuntimeError("Supabase tidak mengembalikan baris setelah proses insert.")

    saved = _normalise_history_row(response.data[0])
    st.session_state.last_result = saved
    load_shared_history.clear()
    return saved


def get_history_admin_password() -> str:
    """Optional password that gates destructive actions on the shared history."""
    try:
        app_settings = st.secrets.get("app", {})
        return str(
            os.getenv("HISTORY_ADMIN_PASSWORD")
            or app_settings.get("history_admin_password", "")
            or ""
        )
    except Exception:
        return os.getenv("HISTORY_ADMIN_PASSWORD", "")


def delete_all_shared_history() -> None:
    """Delete the global history. The caller must verify the admin password first."""
    client = get_supabase_client()
    client.table(SUPABASE_TABLE).delete().gte("id", 1).execute()
    load_shared_history.clear()


# -----------------------------------------------------------------------------
# Shared interface components
# -----------------------------------------------------------------------------
def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-kicker">{kicker}</div><h1 class="page-title">{title}</h1>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon: str, foot: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-top"><div class="metric-label">{label}</div>'
        f'<div class="metric-icon">{icon}</div></div><div class="metric-value">{value}</div>'
        f'<div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def render_result(result: dict[str, Any]) -> None:
    label = result["label"]
    confidence = result.get("confidence")
    source = result.get("source", "Pemeriksaan")
    is_spam = label == "SPAM"
    confidence_text = f"{confidence:.2f}%" if confidence is not None else "Tidak tersedia"
    percent = confidence if confidence is not None else 0.0

    title = "Pesan terindikasi SPAM" if is_spam else "Pesan terindikasi HAM"
    description = (
        "Model menemukan pola yang sesuai dengan pesan spam atau mencurigakan. Periksa tautan dan permintaan data pribadi sebelum mengambil tindakan."
        if is_spam else
        "Model mengklasifikasikan pesan sebagai pesan normal. Tetap periksa identitas pengirim dan konteks pesan."
    )
    icon = "🚨" if is_spam else "✓"
    st.markdown(
        f'<div class="result-panel {"spam" if is_spam else "ham"}">'
        f'<div class="result-heading">{icon} &nbsp;{title}</div>'
        f'<div class="result-copy">{description}</div>'
        f'<div class="result-confidence">Keyakinan model: <b>{confidence_text}</b> &nbsp;·&nbsp; Sumber: {source}</div>'
        f'<div class="confidence-track"><div class="confidence-fill" style="width:{percent:.2f}%"></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if confidence is None:
        st.caption("Model ini tidak menyediakan predict_proba(), sehingga nilai confidence tidak tersedia.")
    if is_spam:
        st.warning("Waspada terhadap tautan tidak dikenal, permintaan OTP/PIN, transfer uang, atau permintaan data pribadi.", icon="⚠️")
    else:
        st.info("Hasil HAM bukan jaminan keamanan. Pastikan pengirim dan isi pesan tetap tepercaya.", icon="ℹ️")


def render_latest_result() -> None:
    result = st.session_state.get("last_result")
    if result:
        st.markdown('<div class="section-heading">Hasil analisis terakhir</div>', unsafe_allow_html=True)
        render_result(result)


def analyze_and_save(text: str, source: str, model: Any, vectorizer: Any) -> None:
    """Run prediction first, then persist it. A database outage never hides the prediction."""
    try:
        with st.spinner("Menganalisis pesan dengan model AI..."):
            result = classify_text(text, model, vectorizer)
    except ValueError as exc:
        st.warning(str(exc), icon="⚠️")
        return
    except Exception as exc:
        st.error(f"Analisis gagal dijalankan: {exc}")
        return

    # Keep the prediction visible even if the remote database is temporarily unavailable.
    st.session_state.last_result = {**result, "source": source}
    try:
        with st.spinner("Menyimpan hasil ke shared history..."):
            save_analysis(result, source)
        st.toast("Analisis selesai dan tersimpan di shared history.", icon="✅")
    except Exception as exc:
        st.error(
            "Prediksi berhasil, tetapi riwayat belum berhasil disimpan ke database. "
            "Periksa konfigurasi Supabase dan koneksi aplikasi."
        )
        st.caption(f"Detail teknis: {type(exc).__name__}")


def history_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in history:
        try:
            parsed = datetime.fromisoformat(str(item["created_at"]).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            timestamp = parsed.astimezone(DISPLAY_TIMEZONE).strftime("%d %b %Y, %H:%M:%S WIB")
        except (KeyError, TypeError, ValueError):
            timestamp = str(item.get("created_at", "-"))
        rows.append({
            "Waktu": timestamp,
            "Sumber": item.get("source", "-"),
            "Klasifikasi": item.get("label", "-"),
            "Confidence": f'{item["confidence"]:.2f}%' if item.get("confidence") is not None else "-",
            "Pesan": item.get("text", ""),
        })
    return pd.DataFrame(rows)


def build_csv(history: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=["created_at", "source", "label", "confidence_percent", "text"],
    )
    writer.writeheader()
    for item in reversed(history):
        writer.writerow({
            "created_at": item.get("created_at", ""),
            "source": item.get("source", ""),
            "label": item.get("label", ""),
            "confidence_percent": (
                f'{item["confidence"]:.2f}' if item.get("confidence") is not None else ""
            ),
            "text": item.get("text", ""),
        })
    return output.getvalue().encode("utf-8-sig")


def render_footer() -> None:
    st.markdown(
        '<div class="app-footer">SPAM DETECTOR AI &nbsp;|&nbsp; TF-IDF · Multinomial Naive Bayes · Tesseract OCR<br>'
        'Hasil merupakan prediksi model, bukan jaminan bahwa suatu pesan aman atau berbahaya.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def dashboard_page(history: list[dict[str, Any]]) -> None:
    page_header("OVERVIEW", "Security at a glance", "Pantau aktivitas pemeriksaan pesan dan pola spam dari seluruh pengguna aplikasi.")
    total = len(history)
    spam_count = sum(1 for item in history if item.get("label") == "SPAM")
    ham_count = total - spam_count
    spam_rate = (spam_count / total * 100) if total else 0.0

    if total == 0:
        st.markdown(
            '<div class="hero"><div class="hero-kicker">AI-POWERED MESSAGE PROTECTION</div>'
            '<div class="hero-title">Know what is<br>inside your inbox.</div>'
            '<p class="hero-copy">Analisis pesan teks atau screenshot untuk membantu mengenali pola spam. Mulai pemeriksaan pertama untuk mengisi ringkasan aktivitas Anda.</p>'
            '<div class="hero-pill">● Model siap digunakan · TF-IDF + Naive Bayes</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="hero"><div class="hero-kicker">AI-POWERED MESSAGE PROTECTION</div>'
            '<div class="hero-title">Your message<br>security overview.</div>'
            '<p class="hero-copy">Ringkasan aktivitas analisis dari seluruh pengguna. Gunakan insight ini untuk meninjau pesan yang perlu diperiksa lebih lanjut.</p>'
            '<div class="hero-pill">● Aktivitas analisis diperbarui secara langsung</div></div>',
            unsafe_allow_html=True,
        )

    cols = st.columns(4, gap="medium")
    with cols[0]:
        metric_card("Total pemeriksaan", f"{total:,}", "◉", "Seluruh pengguna aplikasi")
    with cols[1]:
        metric_card("Pesan terindikasi spam", f"{spam_count:,}", "⚠️", "Klasifikasi SPAM")
    with cols[2]:
        metric_card("Pesan terindikasi ham", f"{ham_count:,}", "✓", "Klasifikasi HAM")
    with cols[3]:
        metric_card("Spam rate", f"{spam_rate:.1f}%", "↗", "Proporsi SPAM dari pemeriksaan")

    st.markdown('<div class="section-heading">Analysis insights</div><div class="section-description">Visualisasi dari riwayat bersama yang tersimpan permanen di Supabase.</div>', unsafe_allow_html=True)
    chart_col, activity_col = st.columns([1.05, 1], gap="large")
    with chart_col:
        with st.container(border=True):
            st.markdown("**Distribusi klasifikasi**")
            if total:
                chart_data = pd.DataFrame({"Jumlah": [spam_count, ham_count]}, index=["SPAM", "HAM"])
                st.bar_chart(chart_data, height=250, color="#4d70df")
            else:
                st.markdown('<div class="empty-state"><div class="empty-state-icon">📊</div><div class="empty-state-title">Belum ada data analisis</div><div class="empty-state-copy">Grafik akan terisi setelah Anda memeriksa pesan.</div></div>', unsafe_allow_html=True)
    with activity_col:
        with st.container(border=True):
            st.markdown("**Aktivitas terbaru**")
            if total:
                recent = history[:5]
                recent_df = pd.DataFrame([
                    {
                        "Klasifikasi": item.get("label", "-"),
                        "Sumber": item.get("source", "-"),
                        "Pesan": (item.get("text", "").replace("\n", " ")[:62] + ("…" if len(item.get("text", "")) > 62 else "")),
                    }
                    for item in recent
                ])
                st.dataframe(recent_df, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="empty-state"><div class="empty-state-icon">🛡️</div><div class="empty-state-title">Siap untuk pemeriksaan pertama</div><div class="empty-state-copy">Aktivitas terbaru akan muncul di area ini.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Quick actions</div><div class="section-description">Pilih cara pemeriksaan yang paling sesuai dengan pesan Anda.</div>', unsafe_allow_html=True)
    q1, q2 = st.columns(2, gap="medium")
    with q1:
        st.markdown('<div class="soft-panel"><div class="feature-icon">⌨️</div><div class="feature-title">Text message analysis</div><div class="feature-copy">Tempel pesan dari SMS, email, atau aplikasi percakapan untuk mendapatkan klasifikasi.</div></div>', unsafe_allow_html=True)
        if st.button("Buka deteksi teks →", key="dash_to_text", use_container_width=True, type="primary"):
            st.session_state.navigation = "Spam Detection"
            st.session_state.sidebar_navigation = "Spam Detection"
            st.rerun()
    with q2:
        st.markdown('<div class="soft-panel"><div class="feature-icon">📷</div><div class="feature-title">Screenshot OCR</div><div class="feature-copy">Unggah screenshot, ekstrak teks dengan OCR, lalu koreksi hasil bacaan sebelum analisis.</div></div>', unsafe_allow_html=True)
        if st.button("Buka Screenshot OCR →", key="dash_to_ocr", use_container_width=True):
            st.session_state.navigation = "Screenshot OCR"
            st.session_state.sidebar_navigation = "Screenshot OCR"
            st.rerun()

    render_latest_result()


def spam_detection_page(model: Any, vectorizer: Any) -> None:
    page_header("MESSAGE SCANNER", "Spam detection", "Masukkan isi pesan untuk memeriksa apakah pola teks lebih menyerupai SPAM atau HAM.")
    st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">1</span> **Masukkan pesan yang ingin diperiksa**', unsafe_allow_html=True)
    st.caption("Dukung teks berbahasa Indonesia maupun Inggris. Maksimal 20.000 karakter per pemeriksaan.")
    st.markdown(
        '<div class="notice">🌐 Teks yang dianalisis dan hasil klasifikasinya disimpan pada history bersama dan dapat dilihat semua pengunjung aplikasi. Jangan gunakan pesan rahasia atau data pribadi.</div>',
        unsafe_allow_html=True,
    )
    consent_to_shared_history = st.checkbox(
        "Saya memahami teks pesan akan disimpan dan terlihat di history bersama.",
        key="consent_manual_shared_history",
    )

    reset_col, count_col = st.columns([1, 4])
    with reset_col:
        if st.button("↺ Reset teks", key="reset_manual", use_container_width=True):
            st.session_state.manual_input = ""
            st.session_state.last_result = None
            st.rerun()
    with count_col:
        current_len = len(st.session_state.get("manual_input", ""))
        st.markdown(f'<div style="text-align:right;color:#8996a8;font-size:.75rem;padding-top:.55rem">{current_len:,} / {MAX_TEXT_LENGTH:,} karakter</div>', unsafe_allow_html=True)

    with st.form("manual_detection_form", clear_on_submit=False):
        st.text_area(
            "Isi pesan",
            key="manual_input",
            height=210,
            max_chars=MAX_TEXT_LENGTH,
            placeholder="Contoh: Selamat! Anda mendapatkan hadiah spesial. Klik tautan berikut untuk mengklaim sekarang...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "🔍  Analisis pesan",
            type="primary",
            use_container_width=True,
            disabled=not consent_to_shared_history,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        analyze_and_save(st.session_state.manual_input, "Input teks", model, vectorizer)
    render_latest_result()


def screenshot_ocr_page(model: Any, vectorizer: Any) -> None:
    page_header("IMAGE INTELLIGENCE", "Screenshot OCR", "Ambil teks dari screenshot menggunakan Tesseract OCR. Periksa dan edit hasil pembacaan sebelum menjalankan deteksi spam.")

    st.markdown(
        '<div class="notice">🌐 <b>Bahasa OCR:</b> Indonesia + English (ind+eng). Teks OCR dan hasil klasifikasi akan disimpan pada history bersama dan dapat dilihat semua pengunjung aplikasi. Gambar screenshot tidak disimpan ke database. Jangan gunakan pesan rahasia atau data pribadi.</div>',
        unsafe_allow_html=True,
    )
    consent_to_shared_history = st.checkbox(
        "Saya memahami teks hasil OCR akan disimpan dan terlihat di history bersama.",
        key="consent_ocr_shared_history",
    )
    uploaded = st.file_uploader(
        "Unggah screenshot pesan",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        key="screenshot_upload",
        help="Format gambar: PNG, JPG, JPEG, WEBP, BMP, atau TIFF.",
    )

    if uploaded is None:
        st.markdown('<div class="empty-state"><div class="empty-state-icon">🖼️</div><div class="empty-state-title">Belum ada screenshot</div><div class="empty-state-copy">Unggah gambar untuk memulai proses ekstraksi teks.</div></div>', unsafe_allow_html=True)
        render_latest_result()
        return

    image_bytes = uploaded.getvalue()
    fingerprint = hashlib.sha256(image_bytes).hexdigest()
    if st.session_state.ocr_fingerprint != fingerprint:
        st.session_state.ocr_fingerprint = fingerprint
        st.session_state.ocr_editor = ""
        st.session_state.ocr_processed_image = None
        st.session_state.ocr_detected = False
        st.session_state.ocr_error = None
        st.session_state.last_result = None

    action_col, reset_col = st.columns([3, 1])
    with action_col:
        run_ocr = st.button("✦  Ekstrak teks dari screenshot", type="primary", use_container_width=True, key="run_ocr")
    with reset_col:
        if st.button("↺ Reset OCR", use_container_width=True, key="reset_ocr"):
            st.session_state.ocr_editor = ""
            st.session_state.ocr_processed_image = None
            st.session_state.ocr_detected = False
            st.session_state.ocr_error = None
            st.session_state.last_result = None
            st.rerun()

    if run_ocr:
        try:
            with st.spinner("Meningkatkan kualitas gambar dan membaca teks dengan Tesseract..."):
                processed_image, detected_text = extract_text(io.BytesIO(image_bytes))
            st.session_state.ocr_processed_image = image_to_png_bytes(processed_image)
            st.session_state.ocr_editor = detected_text
            st.session_state.ocr_detected = True
            st.session_state.ocr_error = None
        except pytesseract.pytesseract.TesseractNotFoundError:
            st.session_state.ocr_error = "Tesseract belum terpasang atau executable tidak ditemukan. Ikuti petunjuk instalasi pada README.md."
        except pytesseract.TesseractError as exc:
            st.session_state.ocr_error = f"Tesseract gagal memproses gambar. Pastikan language pack ind dan eng terpasang. Detail: {exc}"
        except (UnidentifiedImageError, OSError) as exc:
            st.session_state.ocr_error = f"File gambar tidak dapat dibaca: {exc}"
        except Exception as exc:
            st.session_state.ocr_error = f"OCR gagal dijalankan: {exc}"

    if st.session_state.ocr_error:
        st.error(st.session_state.ocr_error)

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.markdown('<div class="section-heading">Image preview</div>', unsafe_allow_html=True)
        st.image(image_bytes, caption=uploaded.name, use_container_width=True)
        if st.session_state.ocr_processed_image:
            with st.expander("Lihat gambar setelah preprocessing"):
                st.image(st.session_state.ocr_processed_image, caption="Grayscale · contrast · sharpness", use_container_width=True)
    with right:
        st.markdown('<div class="section-heading">Editable OCR result</div>', unsafe_allow_html=True)
        st.caption("Teks OCR dapat salah membaca angka, tanda baca, atau nama. Edit bagian yang keliru sebelum analisis.")
        st.text_area(
            "Hasil OCR (dapat diedit)",
            key="ocr_editor",
            height=250,
            max_chars=MAX_TEXT_LENGTH,
            placeholder="Klik tombol Ekstrak teks dari screenshot untuk memulai OCR...",
            label_visibility="collapsed",
        )
        if not st.session_state.ocr_detected:
            st.caption("Ekstraksi belum dijalankan untuk gambar ini.")
        elif not st.session_state.ocr_editor.strip():
            st.warning("Tidak ada teks yang berhasil dikenali. Coba gambar yang lebih jelas atau masukkan teks secara manual.")
        if st.button(
            "🔍  Analisis hasil OCR",
            type="primary",
            use_container_width=True,
            key="analyze_ocr",
            disabled=not consent_to_shared_history,
        ):
            analyze_and_save(st.session_state.ocr_editor, "Screenshot OCR", model, vectorizer)

    render_latest_result()


def history_page(history: list[dict[str, Any]]) -> None:
    page_header(
        "SHARED ACTIVITY LOG",
        "Analysis history",
        "Riwayat bersama seluruh pengguna. Data tersimpan di Supabase dan tetap tersedia setelah aplikasi dimulai ulang.",
    )
    st.markdown(
        '<div class="notice">🌐 <b>Shared history:</b> Semua pengunjung aplikasi dapat melihat riwayat yang sama. '
        'Jangan masukkan OTP, PIN, kata sandi, data kartu, atau informasi pribadi ke dalam pesan yang dianalisis.</div>',
        unsafe_allow_html=True,
    )

    total = len(history)
    spam_count = sum(1 for item in history if item.get("label") == "SPAM")
    ham_count = total - spam_count
    m1, m2, m3 = st.columns(3)
    m1.metric("Total analisis", f"{total:,}")
    m2.metric("SPAM", f"{spam_count:,}")
    m3.metric("HAM", f"{ham_count:,}")

    refresh_col, export_col = st.columns([1, 1], gap="medium")
    with refresh_col:
        if st.button("↻  Refresh shared history", use_container_width=True, key="refresh_shared_history"):
            load_shared_history.clear()
            st.rerun()
    with export_col:
        st.download_button(
            "⬇️  Export seluruh riwayat (CSV)",
            data=build_csv(history),
            file_name=f"spam_detector_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
            disabled=not history,
        )

    if not history:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">🗂️</div>'
            '<div class="empty-state-title">Riwayat masih kosong</div>'
            '<div class="empty-state-copy">Setiap hasil deteksi yang berhasil akan ditambahkan ke database bersama.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        top_left, top_right = st.columns([1, 1])
        with top_left:
            search_text = st.text_input("Cari dalam pesan", placeholder="Ketik kata kunci...", key="history_search")
        with top_right:
            filter_label = st.selectbox("Filter klasifikasi", ["Semua", "SPAM", "HAM"], key="history_filter")

        filtered = history
        if filter_label != "Semua":
            filtered = [item for item in filtered if item.get("label") == filter_label]
        if search_text.strip():
            term = search_text.strip().casefold()
            filtered = [item for item in filtered if term in item.get("text", "").casefold()]

        st.caption(f"Menampilkan {len(filtered):,} dari {total:,} analisis bersama.")
        if filtered:
            st.dataframe(
                history_dataframe(filtered),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Pesan": st.column_config.TextColumn("Pesan", width="large"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                },
            )
        else:
            st.info("Tidak ada riwayat yang cocok dengan pencarian atau filter tersebut.")

    # A public app must not expose a global-delete action to every visitor.
    admin_password = get_history_admin_password()
    st.markdown('<div class="section-heading">Administrator controls</div>', unsafe_allow_html=True)
    if not admin_password:
        st.caption(
            "Penghapusan seluruh riwayat dinonaktifkan. Admin dapat mengaktifkannya dengan "
            "menambahkan [app].history_admin_password pada Streamlit Secrets."
        )
    else:
        with st.expander("Hapus seluruh riwayat bersama (admin)"):
            entered_password = st.text_input(
                "Kata sandi admin",
                type="password",
                key="history_admin_password_input",
            )
            confirm_delete = st.checkbox(
                "Saya memahami bahwa tindakan ini menghapus riwayat untuk semua pengguna.",
                key="confirm_delete_shared_history",
            )
            if st.button(
                "🗑️  Hapus seluruh riwayat bersama",
                disabled=not (entered_password and confirm_delete),
                type="secondary",
                key="delete_shared_history",
            ):
                if hmac.compare_digest(entered_password.encode("utf-8"), admin_password.encode("utf-8")):
                    try:
                        with st.spinner("Menghapus seluruh riwayat bersama..."):
                            delete_all_shared_history()
                        st.session_state.last_result = None
                        st.success("Seluruh riwayat bersama berhasil dihapus.")
                        st.rerun()
                    except Exception as exc:
                        st.error("Riwayat tidak berhasil dihapus. Periksa koneksi dan hak akses database.")
                        st.caption(f"Detail teknis: {type(exc).__name__}")
                else:
                    st.error("Kata sandi admin tidak sesuai.")


def about_page() -> None:
    page_header("ABOUT SPAM DETECTOR AI", "Understand the technology", "Kenali proses di balik analisis pesan, komponen machine learning yang digunakan, dan batasan hasil prediksi.")
    st.markdown(
        '<div class="hero"><div class="hero-kicker">BUILT FOR EVERYDAY DIGITAL AWARENESS</div>'
        '<div class="hero-title">A smarter first look<br>at suspicious messages.</div>'
        '<p class="hero-copy">Spam Detector AI membantu mengidentifikasi pola pesan yang berpotensi spam melalui pemrosesan bahasa alami dan pembelajaran mesin. Aplikasi ini merupakan alat bantu pemeriksaan, bukan pengganti penilaian keamanan manusia.</p>'
        '<div class="hero-pill">🛡️ Text classification · Screenshot OCR</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Technology stack</div><div class="section-description">Model dan komponen utama yang dipertahankan dari aplikasi awal.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="soft-panel"><div class="feature-icon">✦</div><div class="feature-title">TF-IDF Vectorization</div><div class="feature-copy">Mengubah teks menjadi representasi fitur numerik berdasarkan bobot Term Frequency–Inverse Document Frequency. Representasi ini menjadi masukan bagi model klasifikasi.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="soft-panel" style="margin-top:1rem"><div class="feature-icon">🧠</div><div class="feature-title">Multinomial Naive Bayes</div><div class="feature-copy">Algoritma klasifikasi probabilistik yang umum digunakan untuk data berbasis frekuensi fitur, termasuk klasifikasi teks. Model terlatih dimuat dari spam_model.pkl.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="soft-panel"><div class="feature-icon">📷</div><div class="feature-title">Tesseract OCR</div><div class="feature-copy">Mengekstrak teks dari screenshot. Gambar dikonversi ke grayscale, diperbesar bila diperlukan, lalu ditingkatkan kontras dan ketajamannya sebelum diproses dengan bahasa Indonesia dan Inggris.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="soft-panel" style="margin-top:1rem"><div class="feature-icon">📈</div><div class="feature-title">Prediction confidence</div><div class="feature-copy">Saat model mendukung predict_proba(), aplikasi menampilkan probabilitas untuk kelas hasil prediksi. Nilai ini menunjukkan keyakinan model, bukan jaminan kebenaran.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">How the analysis works</div>', unsafe_allow_html=True)
    steps = [
        ("Input", "Pengguna mengetik pesan atau mengunggah screenshot."),
        ("OCR (screenshot only)", "Tesseract membaca teks dari gambar; hasilnya dapat dikoreksi sebelum klasifikasi."),
        ("Feature extraction", "TF-IDF vectorizer mengubah teks menjadi fitur numerik."),
        ("Classification", "Multinomial Naive Bayes memprediksi kelas SPAM atau HAM."),
        ("Result & shared history", "Aplikasi menampilkan label dan confidence bila tersedia, lalu menyimpan hasil ke database Supabase bersama."),
    ]
    for number, (step, description) in enumerate(steps, start=1):
        st.markdown(
            f'<div class="soft-panel" style="margin:.55rem 0;padding:.85rem 1rem"><span class="step-number">{number}</span>'
            f'<b style="color:#34445d">{step}</b><div style="color:#7a879a;font-size:.82rem;margin:.35rem 0 0 2.1rem">{description}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-heading">Important limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - Prediksi bergantung pada data dan proses pelatihan model yang sudah ada. Pesan baru, bahasa campuran, slang, atau pola serangan yang belum dikenal dapat menghasilkan klasifikasi yang keliru.
        - OCR dapat salah membaca teks, terutama pada gambar buram, teks kecil, latar ramai, atau screenshot yang terpotong. Selalu periksa dan edit hasil OCR.
        - Confidence adalah keluaran probabilitas model, bukan ukuran keamanan yang terkalibrasi atau bukti bahwa pesan aman.
        - Riwayat disimpan pada database Supabase bersama dan dapat dilihat oleh semua pengunjung aplikasi. Jangan masukkan pesan yang berisi informasi pribadi atau rahasia.
        - Jangan mengunggah atau memasukkan OTP, PIN, kata sandi, nomor kartu, atau informasi pribadi yang tidak diperlukan.
        """
    )
    st.markdown('<div class="notice">🔐 <b>Privacy note:</b> Isi pesan yang dianalisis disimpan pada database Supabase bersama dan dapat dilihat oleh semua pengunjung aplikasi. Jangan masukkan OTP, PIN, kata sandi, atau data pribadi.</div>', unsafe_allow_html=True)


def main() -> None:
    ensure_state()
    configure_tesseract()

    try:
        model, vectorizer = load_model()
        model_error = None
    except Exception as exc:
        model, vectorizer = None, None
        model_error = str(exc)

    try:
        shared_history = load_shared_history()
        database_error = None
    except Exception as exc:
        shared_history = []
        database_error = exc

    with st.sidebar:
        st.markdown(
            '<div class="brand-lockup"><div><div class="brand-name">Spam Detector AI</div><div class="brand-sub">Spam Intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="color:#8394b2;font-size:.68rem;font-weight:800;letter-spacing:.14em;margin:.2rem 0 .65rem .25rem">WORKSPACE</div>', unsafe_allow_html=True)
        pages = ["Dashboard", "Spam Detection", "Screenshot OCR", "History", "About"]
        selected_default = st.session_state.get("sidebar_navigation", st.session_state.get("navigation", "Dashboard"))
        if selected_default not in pages:
            selected_default = "Dashboard"
        selected_page = st.radio(
            "Navigation",
            pages,
            index=pages.index(selected_default),
            format_func=lambda item: {
                "Dashboard": "◫   Dashboard",
                "Spam Detection": "⌕   Spam Detection",
                "Screenshot OCR": "▧   Screenshot OCR",
                "History": "◷   History",
                "About": "ⓘ   About",
            }[item],
            label_visibility="collapsed",
            key="sidebar_navigation",
        )
        st.session_state.navigation = selected_page
        st.markdown("<hr>", unsafe_allow_html=True)
        if model_error:
            st.markdown('<div style="color:#ffb7bd;font-size:.77rem;font-weight:700">● Model tidak tersedia</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-note">Periksa keberadaan spam_model.pkl dan tfidf_vectorizer.pkl di folder aplikasi.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#8fe0c8;font-size:.77rem;font-weight:700">● Model siap digunakan</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-note">TF-IDF + Multinomial Naive Bayes<br>OCR: Indonesian + English</div>', unsafe_allow_html=True)
        if database_error:
            st.markdown('<div style="color:#ffcf91;font-size:.77rem;font-weight:700;margin-top:.5rem">● Database belum terhubung</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#8fe0c8;font-size:.77rem;font-weight:700;margin-top:.5rem">● Shared history connected</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-note">Spam Detector AI membantu pemeriksaan awal terhadap pesan mencurigakan. Hasil merupakan referensi, bukan keputusan keamanan otomatis.</div>', unsafe_allow_html=True)

    if model_error:
        page_header("SETUP REQUIRED", "Model files not found", "Aplikasi sudah berjalan, tetapi file model yang digunakan untuk prediksi belum tersedia.")
        st.error(model_error)
        st.markdown(
            "Pastikan struktur folder Anda seperti berikut, kemudian jalankan kembali aplikasi:\n\n"
            "```text\nspam_detector_ai/\n├── app.py\n├── spam_model.pkl\n├── tfidf_vectorizer.pkl\n├── requirements.txt\n└── .streamlit/\n    ├── config.toml\n    └── secrets.toml (lokal saja; jangan commit)\n```"
        )
        st.info("File `spam_model.pkl` dan `tfidf_vectorizer.pkl` tidak dibuat ulang atau dimodifikasi oleh aplikasi ini.")
        render_footer()
        return

    if database_error:
        st.warning(
            "Supabase belum terhubung. Prediksi tetap dapat dijalankan, tetapi hasil baru belum dapat disimpan "
            "ke history permanen. Ikuti bagian konfigurasi Supabase pada README.md."
        )

    if selected_page == "Dashboard":
        dashboard_page(shared_history)
    elif selected_page == "Spam Detection":
        spam_detection_page(model, vectorizer)
    elif selected_page == "Screenshot OCR":
        screenshot_ocr_page(model, vectorizer)
    elif selected_page == "History":
        history_page(shared_history)
    else:
        about_page()

    render_footer()


if __name__ == "__main__":
    main()
