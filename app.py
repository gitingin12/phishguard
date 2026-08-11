import re
import joblib
import tldextract
import streamlit as st
from difflib import SequenceMatcher
from urllib.parse import urlparse


# =========================================================
# KONFIGURASI
# =========================================================

st.set_page_config(
    page_title="PhishGuard - Banking URL Detector",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("phishing_rf_model.pkl")


try:
    model_bundle = load_model()

    rf = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    feature_columns = model_bundle["feature_columns"]
    bank_brands = model_bundle["bank_brands"]
    auth_keywords_dict = model_bundle["auth_keywords_dict"]
    official_domains = model_bundle["official_domains"]

except Exception as e:
    st.error("Model gagal dimuat.")
    st.code(str(e))
    st.stop()


# =========================================================
# VALIDASI FITUR
# =========================================================

EXPECTED_FEATURES = [
    "brand_detected",
    "auth_detected",
    "brand_domain_similarity",
    "subdomain_count"
]

if feature_columns != EXPECTED_FEATURES:
    st.error(
        "Urutan fitur pada model tidak sesuai dengan konfigurasi aplikasi."
    )
    st.write("Fitur pada model:", feature_columns)
    st.stop()


# =========================================================
# EXTRACT DOMAIN
# =========================================================

extractor = tldextract.TLDExtract(suffix_list_urls=())


def get_domain_parts(url):
    """
    Memisahkan root domain dan subdomain dari URL.
    """

    url_raw = str(url).strip().lower()

    if not url_raw.startswith(("http://", "https://")):
        url_raw = "http://" + url_raw

    ext = extractor(url_raw)

    return ext.domain, ext.subdomain


# =========================================================
# SUBDOMAIN COUNT
# =========================================================

def count_subdomain(subdomain_raw):

    if not subdomain_raw:
        return 0

    subdomain_raw = str(subdomain_raw).strip()

    return len(subdomain_raw.split("."))


# =========================================================
# BRAND NORMALIZATION
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(text).lower()
    )


def get_brand_names():

    brands = []

    if isinstance(bank_brands, dict):

        for key, value in bank_brands.items():
            brands.append(str(key))

            if isinstance(value, (list, tuple, set)):
                brands.extend([str(x) for x in value])

            else:
                brands.append(str(value))

    elif isinstance(bank_brands, (list, tuple, set)):

        brands = [str(x) for x in bank_brands]

    else:

        brands = [str(bank_brands)]

    return list(set(brands))


# =========================================================
# BRAND DETECTION
# =========================================================

def detect_brand(url):

    url_lower = str(url).lower()

    domain, _ = get_domain_parts(url)

    domain_normalized = normalize_text(domain)

    for brand in get_brand_names():

        brand_normalized = normalize_text(brand)

        if not brand_normalized:
            continue

        if brand_normalized in domain_normalized:
            return True, brand

        if brand.lower() in url_lower:
            return True, brand

    return False, None


# =========================================================
# AUTH DETECTION
# =========================================================

def detect_auth(url):

    url_lower = str(url).lower()

    if isinstance(auth_keywords_dict, dict):

        keywords = []

        for key, value in auth_keywords_dict.items():

            keywords.append(str(key))

            if isinstance(value, (list, tuple, set)):
                keywords.extend([str(x) for x in value])

            else:
                keywords.append(str(value))

    elif isinstance(auth_keywords_dict, (list, tuple, set)):

        keywords = [str(x) for x in auth_keywords_dict]

    else:

        keywords = [str(auth_keywords_dict)]

    for keyword in set(keywords):

        keyword = keyword.strip().lower()

        if keyword and keyword in url_lower:
            return True, keyword

    return False, None


# =========================================================
# BRAND DOMAIN SIMILARITY
# =========================================================

def compute_brand_domain_similarity(brand_name, root_domain):

    if not brand_name or not root_domain:
        return 0.0

    return round(
        SequenceMatcher(
            None,
            str(brand_name).lower().strip(),
            str(root_domain).lower().strip()
        ).ratio(),
        4
    )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(url):

    url_raw = str(url).strip()

    root_domain, subdomain_raw = get_domain_parts(url_raw)

    brand_detected, detected_brand = detect_brand(url_raw)

    auth_detected, detected_auth = detect_auth(url_raw)

    brand_similarity = compute_brand_domain_similarity(
        detected_brand,
        root_domain
    )

    subdomain_count = count_subdomain(subdomain_raw)

    features = {
        "brand_detected": int(brand_detected),
        "auth_detected": int(auth_detected),
        "brand_domain_similarity": brand_similarity,
        "subdomain_count": subdomain_count
    }

    return features, {
        "root_domain": root_domain,
        "subdomain": subdomain_raw,
        "detected_brand": detected_brand,
        "detected_auth": detected_auth
    }


# =========================================================
# PREDICTION
# =========================================================

def predict_url(url):

    features, details = extract_features(url)

    feature_values = [
        features[col]
        for col in feature_columns
    ]

    prediction = rf.predict([feature_values])[0]

    probabilities = rf.predict_proba([feature_values])[0]

    confidence = float(probabilities.max() * 100)

    # Menentukan label berdasarkan label encoder
    try:
        predicted_label = label_encoder.inverse_transform(
            [prediction]
        )[0]

    except Exception:
        predicted_label = (
            "Phishing"
            if int(prediction) == 1
            else "Legitimate"
        )

    return (
        predicted_label,
        confidence,
        probabilities,
        features,
        details
    )


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 25px;
        border-radius: 16px;
        margin-top: 20px;
        border: 1px solid #ddd;
    }

    .feature-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ PhishGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deteksi URL phishing yang meniru layanan perbankan '
    'menggunakan Random Forest.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUT
# =========================================================

st.subheader("🔍 Analisis URL")

url_input = st.text_input(
    "Masukkan URL yang ingin dianalisis",
    placeholder="Contoh: https://example.com"
)

analyze = st.button(
    "🔎 Analisis URL",
    use_container_width=True
)


# =========================================================
# PROCESS
# =========================================================

if analyze:

    if not url_input.strip():

        st.warning("Silakan masukkan URL terlebih dahulu.")

    else:

        with st.spinner("Menganalisis URL..."):

            try:

                (
                    prediction,
                    confidence,
                    probabilities,
                    features,
                    details
                ) = predict_url(url_input)

                # ==========================================
                # HASIL
                # ==========================================

                st.divider()

                st.subheader("📊 Hasil Analisis")

                if str(prediction).lower() == "phishing":

                    st.error(
                        "⚠️ PHISHING TERDETEKSI"
                    )

                else:

                    st.success(
                        "✅ URL TERKLASIFIKASI LEGITIMATE"
                    )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                # ==========================================
                # PROBABILITAS
                # ==========================================

                st.subheader("Probabilitas Kelas")

                class_names = list(label_encoder.classes_)

                probability_data = {}

                for class_name, probability in zip(
                    class_names,
                    probabilities
                ):

                    probability_data[class_name] = (
                        float(probability) * 100
                    )

                cols = st.columns(len(probability_data))

                for col, (name, value) in zip(
                    cols,
                    probability_data.items()
                ):

                    with col:

                        st.metric(
                            name,
                            f"{value:.2f}%"
                        )

                # ==========================================
                # INFORMASI URL
                # ==========================================

                st.subheader("🌐 Informasi URL")

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Root Domain:**",
                        details["root_domain"]
                    )

                    st.write(
                        "**Subdomain:**",
                        details["subdomain"]
                        if details["subdomain"]
                        else "-"
                    )

                with col2:

                    st.write(
                        "**Brand Terdeteksi:**",
                        details["detected_brand"]
                        if details["detected_brand"]
                        else "Tidak"
                    )

                    st.write(
                        "**Auth Keyword:**",
                        details["detected_auth"]
                        if details["detected_auth"]
                        else "Tidak"
                    )

                # ==========================================
                # 4 FITUR MODEL
                # ==========================================

                st.subheader("🧩 Fitur yang Digunakan Model")

                feature_cols = st.columns(4)

                feature_display = {
                    "brand_detected": (
                        "Brand Detected",
                        "Ya" if features["brand_detected"] else "Tidak"
                    ),

                    "auth_detected": (
                        "Auth Detected",
                        "Ya" if features["auth_detected"] else "Tidak"
                    ),

                    "brand_domain_similarity": (
                        "Brand-Domain Similarity",
                        f"{features['brand_domain_similarity']:.4f}"
                    ),

                    "subdomain_count": (
                        "Subdomain Count",
                        str(features["subdomain_count"])
                    )
                }

                for col, feature_name in zip(
                    feature_cols,
                    feature_columns
                ):

                    title, value = feature_display[feature_name]

                    with col:

                        st.markdown(
                            f"""
                            <div class="feature-card">
                                <b>{title}</b>
                                <br>
                                <span style="font-size:24px;">
                                    {value}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # ==========================================
                # CATATAN
                # ==========================================

                st.info(
                    "Confidence merupakan probabilitas prediksi "
                    "terbesar dari model Random Forest dan bukan "
                    "jaminan keamanan absolut terhadap URL."
                )

            except Exception as e:

                st.error(
                    "Terjadi kesalahan saat melakukan prediksi."
                )

                st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PhishGuard — Random Forest Banking URL Phishing Detection"
)