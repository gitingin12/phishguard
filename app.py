import os
import re
import joblib
import tldextract
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher


# =========================================================
# KONFIGURASI
# =========================================================
st.set_page_config(
    page_title="PhishGuard - Deteksi URL Phishing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CSS / UI
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(49,51,63,.06), rgba(255,255,255,.02));
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        opacity: .78;
        margin-top: .55rem;
        line-height: 1.55;
    }

    .about-box {
        padding: 1rem 1.2rem;
        border-left: 4px solid #4b7bec;
        border-radius: 8px;
        background: rgba(75,123,236,.06);
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin: 1.25rem 0 .65rem;
    }

    .result-card {
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.22);
        margin-top: .5rem;
    }

    .result-phishing {
        background: rgba(255, 75, 75, .08);
        border-color: rgba(255,75,75,.35);
    }

    .result-legitimate {
        background: rgba(35, 180, 105, .08);
        border-color: rgba(35,180,105,.35);
    }

    .result-label {
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: .25rem;
    }

    .result-desc {
        opacity: .78;
        line-height: 1.5;
    }

    .mini-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: 1rem;
        height: 100%;
        min-height: 108px;
    }

    .mini-title {
        font-weight: 750;
        margin-bottom: .4rem;
    }

    .mini-value {
        font-size: 1.3rem;
        font-weight: 750;
    }

    .reason-card {
        border-left: 3px solid rgba(128,128,128,.35);
        padding: .7rem .9rem;
        margin: .35rem 0;
        line-height: 1.5;
    }

    .reason-title {
        font-weight: 750;
        margin-bottom: .15rem;
    }

    .step {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        height: 100%;
    }

    .step-number {
        font-size: 1.5rem;
        font-weight: 800;
    }

    .step-title {
        font-weight: 750;
        margin-top: .25rem;
    }

    .muted {
        opacity: .72;
    }

    .footer {
        text-align: center;
        opacity: .6;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


EXPECTED_FEATURES = [
    "brand_detected",
    "auth_detected",
    "brand_domain_similarity",
    "subdomain_count",
]

if feature_columns != EXPECTED_FEATURES:
    st.error("Urutan fitur pada model tidak sesuai dengan konfigurasi aplikasi.")
    st.write("Fitur pada model:", feature_columns)
    st.stop()

extractor = tldextract.TLDExtract(suffix_list_urls=())


@st.cache_data
def load_url_dataset(path="url_dataset.csv"):
    # Prefer explicit filenames, else try to discover a CSV in the app folder.
    candidates = ["url_dataset.csv", "banking_dataset.csv", path]
    # Add any other CSV files in the folder as fallback.
    try:
        for entry in os.listdir('.'):
            if entry.lower().endswith('.csv') and entry not in candidates:
                candidates.append(entry)
    except Exception:
        pass

    for fname in candidates:
        if not fname:
            continue
        if not os.path.exists(fname):
            continue
        try:
            df = pd.read_csv(fname)
            # Ensure 'url' column exists or try to infer it
            if "url" not in df.columns:
                # try lowercase variants
                for c in df.columns:
                    if c.strip().lower() == "url":
                        df = df.rename(columns={c: "url"})
                        break
            if "url" in df.columns:
                df["url"] = df["url"].astype(str)
            # attach source filename for UI reference
            df.attrs["_source_file"] = fname
            return df
        except Exception:
            continue

    return None


url_dataset = load_url_dataset()


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def get_domain_parts(url):
    url_raw = str(url).strip().lower()
    if not url_raw.startswith(("http://", "https://")):
        url_raw = "http://" + url_raw
    ext = extractor(url_raw)
    return ext.domain, ext.subdomain


def count_subdomain(subdomain_raw):
    if not subdomain_raw:
        return 0
    return len(str(subdomain_raw).strip().split("."))


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def get_brand_names():
    brands = []
    if isinstance(bank_brands, dict):
        for key, value in bank_brands.items():
            brands.append(str(key))
            if isinstance(value, (list, tuple, set)):
                brands.extend(str(x) for x in value)
            else:
                brands.append(str(value))
    elif isinstance(bank_brands, (list, tuple, set)):
        brands = [str(x) for x in bank_brands]
    else:
        brands = [str(bank_brands)]
    return list(set(brands))


def detect_brand(url):
    """Mengembalikan status, brand, dan alasan deteksi."""
    url_lower = str(url).lower()
    domain, _ = get_domain_parts(url)
    domain_normalized = normalize_text(domain)

    for brand in get_brand_names():
        brand_normalized = normalize_text(brand)
        if not brand_normalized:
            continue

        if brand_normalized in domain_normalized:
            return True, brand, "root domain"

        if brand.lower() in url_lower:
            return True, brand, "URL"

    return False, None, None


def detect_auth(url):
    """Mengembalikan status, keyword, dan alasan deteksi."""
    url_lower = str(url).lower()

    if isinstance(auth_keywords_dict, dict):
        keywords = []
        for key, value in auth_keywords_dict.items():
            keywords.append(str(key))
            if isinstance(value, (list, tuple, set)):
                keywords.extend(str(x) for x in value)
            else:
                keywords.append(str(value))
    elif isinstance(auth_keywords_dict, (list, tuple, set)):
        keywords = [str(x) for x in auth_keywords_dict]
    else:
        keywords = [str(auth_keywords_dict)]

    for keyword in set(keywords):
        keyword = keyword.strip().lower()
        if keyword and keyword in url_lower:
            return True, keyword, keyword

    return False, None, None


def compute_brand_domain_similarity(brand_name, root_domain):
    if not brand_name or not root_domain:
        return 0.0

    return round(
        SequenceMatcher(
            None,
            str(brand_name).lower().strip(),
            str(root_domain).lower().strip(),
        ).ratio(),
        4,
    )


def extract_features(url):
    url_raw = str(url).strip()
    root_domain, subdomain_raw = get_domain_parts(url_raw)

    brand_detected, detected_brand, brand_source = detect_brand(url_raw)
    auth_detected, detected_auth, auth_match = detect_auth(url_raw)

    brand_similarity = compute_brand_domain_similarity(
        detected_brand,
        root_domain,
    )

    subdomain_count = count_subdomain(subdomain_raw)

    features = {
        "brand_detected": int(brand_detected),
        "auth_detected": int(auth_detected),
        "brand_domain_similarity": brand_similarity,
        "subdomain_count": subdomain_count,
    }

    details = {
        "root_domain": root_domain,
        "subdomain": subdomain_raw,
        "detected_brand": detected_brand,
        "brand_source": brand_source,
        "detected_auth": detected_auth,
        "auth_match": auth_match,
    }

    return features, details


# =========================================================
# PREDICTION
# =========================================================
def predict_url(url):
    features, details = extract_features(url)

    # Gunakan DataFrame dengan nama kolom yang sama seperti saat training.
    # Ini mencegah warning scikit-learn tentang feature names.
    model_input = pd.DataFrame(
        [[features[col] for col in feature_columns]],
        columns=feature_columns,
    )

    prediction = rf.predict(model_input)[0]
    probabilities = rf.predict_proba(model_input)[0]
    confidence = float(probabilities.max() * 100)

    try:
        predicted_label = label_encoder.inverse_transform([prediction])[0]
    except Exception:
        predicted_label = "Phishing" if int(prediction) == 1 else "Legitimate"

    return predicted_label, confidence, probabilities, features, details


# =========================================================
# HEADER + TENTANG SISTEM
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ PhishGuard</div>
        <div class="hero-subtitle">Banking URL Phishing Detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Tentang Sistem")
st.markdown(
    """
    <div class="about-box">
        PhishGuard mengklasifikasikan URL terkait layanan perbankan menjadi
        <b>Phishing</b> atau <b>Legitimate</b> menggunakan model
        <b>Random Forest</b>.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# INPUT
# =========================================================
st.markdown('<div class="section-title">🔍 Analisis URL</div>', unsafe_allow_html=True)
st.write(
    "Masukkan URL yang ingin diperiksa. Sistem akan mengekstraksi 4 fitur "
    "sesuai model sebelum melakukan klasifikasi."
)

url_input = st.text_input(
    "URL",
    placeholder="Contoh: https://contoh-bank.com/login",
    label_visibility="collapsed",
)

analyze = st.button(
    "🔎 Analisis URL",
    type="primary",
    use_container_width=True,
)


# =========================================================
# PROCESS
# =========================================================
if analyze:
    if not url_input.strip():
        st.warning("Silakan masukkan URL terlebih dahulu.")
    else:
        with st.spinner("Mengekstraksi fitur dan melakukan klasifikasi..."):
            try:
                (
                    prediction,
                    confidence,
                    probabilities,
                    features,
                    details,
                ) = predict_url(url_input)

                is_phishing = str(prediction).lower() == "phishing"
                result_class = "result-phishing" if is_phishing else "result-legitimate"
                result_icon = "⚠️" if is_phishing else "✅"
                result_title = (
                    "PHISHING TERDETEKSI"
                    if is_phishing
                    else "URL TERKLASIFIKASI LEGITIMATE"
                )
                result_desc = (
                    "Model menghasilkan klasifikasi phishing berdasarkan nilai fitur "
                    "yang diekstraksi dari URL aktual yang dimasukkan."
                    if is_phishing
                    else
                    "Model menghasilkan klasifikasi legitimate berdasarkan nilai fitur "
                    "yang diekstraksi dari URL aktual yang dimasukkan."
                )

                st.divider()
                st.markdown(
                    '<div class="section-title">📊 Hasil Analisis</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="result-card {result_class}">
                        <div class="result-label">{result_icon} {result_title}</div>
                        <div class="result-desc">{result_desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("### Probabilitas Prediksi")
                st.progress(min(confidence / 100, 1.0))
                st.metric("Probabilitas Prediksi Tertinggi",
                         f"{confidence:.2f}%"
                         )

                # =====================================================
                # PROBABILITAS
                # =====================================================
                st.markdown(
                    '<div class="section-title">📈 Probabilitas Kelas</div>',
                    unsafe_allow_html=True,
                )

                class_names = list(label_encoder.classes_)
                probability_data = {
                    name: float(probability) * 100
                    for name, probability in zip(class_names, probabilities)
                }

                prob_cols = st.columns(len(probability_data))
                for col, (name, value) in zip(prob_cols, probability_data.items()):
                    with col:
                        st.metric(name, f"{value:.2f}%")
                        st.progress(min(value / 100, 1.0))

                # =====================================================
                # INFORMASI URL
                # =====================================================
                st.markdown(
                    '<div class="section-title">🌐 Informasi URL</div>',
                    unsafe_allow_html=True,
                )

                info_cols = st.columns(4)
                info_items = [
                    ("Root Domain", details["root_domain"] or "-"),
                    ("Subdomain", details["subdomain"] or "Tidak ada"),
                    ("Brand Terdeteksi", details["detected_brand"] or "Tidak"),
                    ("Auth Keyword", details["detected_auth"] or "Tidak"),
                ]

                for col, (title, value) in zip(info_cols, info_items):
                    with col:
                        st.markdown(
                            f"""
                            <div class="mini-card">
                                <div class="mini-title">{title}</div>
                                <div class="mini-value">{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # =====================================================
                # FITUR MODEL
                # =====================================================
                st.markdown(
                    '<div class="section-title">🧩 Fitur yang Digunakan Model</div>',
                    unsafe_allow_html=True,
                )

                feature_display = {
                    "brand_detected": (
                        "Brand Detected",
                        "Ya" if features["brand_detected"] else "Tidak",
                    ),
                    "auth_detected": (
                        "Auth Detected",
                        "Ya" if features["auth_detected"] else "Tidak",
                    ),
                    "brand_domain_similarity": (
                        "Brand-Domain Similarity",
                        f"{features['brand_domain_similarity']:.4f}",
                    ),
                    "subdomain_count": (
                        "Subdomain Count",
                        str(features["subdomain_count"]),
                    ),
                }

                feature_cols = st.columns(4)
                for col, feature_name in zip(feature_cols, feature_columns):
                    title, value = feature_display[feature_name]
                    with col:
                        st.markdown(
                            f"""
                            <div class="mini-card">
                                <div class="mini-title">{title}</div>
                                <div class="mini-value">{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # =====================================================
                # PENJELASAN DINAMIS BERDASARKAN URL YANG DIANALISIS
                # =====================================================
                st.markdown(
                    '<div class="section-title">💡 Mengapa Nilai 4 Fitur Seperti Ini?</div>',
                    unsafe_allow_html=True,
                )

                # Brand Detected
                if features["brand_detected"]:
                    brand = details["detected_brand"] or "brand"
                    source = details["brand_source"] or "URL"
                    brand_reason = (
                        f"Bernilai <b>Ya</b> karena indikasi brand <b>{brand}</b> "
                        f"ditemukan pada {source} dari URL aktual yang dianalisis. "
                        "Deteksi ini mengikuti daftar brand yang tersimpan bersama model."
                    )
                else:
                    brand_reason = (
                        "Bernilai <b>Tidak</b> karena tidak ditemukan indikasi nama brand "
                        "yang sesuai dengan daftar brand pada URL yang dianalisis."
                    )

                # Auth Detected
                if features["auth_detected"]:
                    auth_keyword = details["detected_auth"] or "keyword autentikasi"
                    auth_reason = (
                        f"Bernilai <b>Ya</b> karena keyword autentikasi "
                        f"<b>{auth_keyword}</b> ditemukan pada URL aktual yang dianalisis."
                    )
                else:
                    auth_reason = (
                        "Bernilai <b>Tidak</b> karena tidak ada keyword autentikasi "
                        "yang tersimpan dalam konfigurasi model yang ditemukan pada URL."
                    )

                # Brand-Domain Similarity
                similarity = features["brand_domain_similarity"]
                brand = details["detected_brand"]
                root_domain = details["root_domain"]

                if not brand or not root_domain:
                    similarity_reason = (
                        f"Bernilai <b>0.0000</b> karena tidak terdapat brand yang "
                        f"berhasil dideteksi untuk dibandingkan dengan root domain "
                        f"<b>{root_domain or '-'}</b>. Dalam kondisi ini, sistem "
                        "menghasilkan nilai similarity 0."
                    )
                elif similarity == 1.0:
                    similarity_reason = (
                        f"Bernilai <b>1.0000</b> karena teks brand "
                        f"<b>{brand}</b> dan teks root domain <b>{root_domain}</b> "
                        "identik setelah diseragamkan menjadi huruf kecil. "
                        "Nilai 1.0000 merupakan tingkat kemiripan teks maksimum "
                        "yang dihasilkan oleh SequenceMatcher."
                    )
                else:
                    similarity_reason = (
                        f"Bernilai <b>{similarity:.4f}</b> karena nilai tersebut merupakan "
                        f"tingkat kemiripan teks antara brand <b>{brand}</b> dan root domain "
                        f"<b>{root_domain}</b> yang dihitung menggunakan SequenceMatcher. "
                        "Nilai semakin mendekati 1 menunjukkan kemiripan teks yang semakin tinggi."
                    )

                # Subdomain Count
                subdomain = details["subdomain"]
                if features["subdomain_count"] == 0:
                    subdomain_reason = (
                        "Bernilai <b>0</b> karena tidak terdapat subdomain pada URL; "
                        "bagian subdomain yang diekstraksi dari URL kosong."
                    )
                else:
                    subdomain_reason = (
                        f"Bernilai <b>{features['subdomain_count']}</b> karena subdomain "
                        f"<b>{subdomain}</b> terdiri dari "
                        f"{features['subdomain_count']} bagian yang dipisahkan oleh titik."
                    )

                reason_items = [
                    ("Brand Detected", brand_reason),
                    ("Auth Detected", auth_reason),
                    ("Brand-Domain Similarity", similarity_reason),
                    ("Subdomain Count", subdomain_reason),
                ]

                for title, reason in reason_items:
                    st.markdown(
                        f"""
                        <div class="reason-card">
                            <div class="reason-title">{title}</div>
                            <div>{reason}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # =====================================================
                # ALUR PREDIKSI
                # =====================================================
                st.markdown(
                    '<div class="section-title">🔄 Alur Prediksi</div>',
                    unsafe_allow_html=True,
                )

                step_cols = st.columns(4)
                steps = [
                    ("1", "URL", "URL dimasukkan pengguna"),
                    ("2", "Ekstraksi Fitur", "4 fitur dihitung dari URL"),
                    ("3", "Random Forest", "Model memproses nilai fitur"),
                    ("4", "Prediksi", "Phishing atau Legitimate"),
                ]

                for col, (num, title, desc) in zip(step_cols, steps):
                    with col:
                        st.markdown(
                            f"""
                            <div class="step">
                                <div class="step-number">{num}</div>
                                <div class="step-title">{title}</div>
                                <div class="muted">{desc}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.info(
                    "Catatan: probabilitas prediksi menunjukkan probabilitas kelas "
                    "yang dihasilkan model Random Forest berdasarkan fitur URL yang "
                    "dianalisis. Nilai ini bukan jaminan bahwa URL aman atau berbahaya "
                    "secara absolut."
                )

            except Exception as e:
                st.error("Terjadi kesalahan saat melakukan prediksi.")
                st.exception(e)


# =========================================================
# CARA KERJA SAAT BELUM ADA ANALISIS
# =========================================================
if not analyze:
    st.divider()
    st.markdown(
        '<div class="section-title">📚 Cara Kerja PhishGuard</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "URL yang dimasukkan akan diproses menjadi empat fitur, kemudian "
        "keempat fitur tersebut diberikan kepada model Random Forest untuk "
        "menghasilkan klasifikasi Phishing atau Legitimate."
    )

    step_cols = st.columns(4)
    steps = [
        ("1", "Masukkan URL", "Masukkan alamat URL yang ingin dianalisis."),
        ("2", "Ekstraksi Fitur", "Sistem menghitung 4 fitur dari URL."),
        ("3", "Klasifikasi", "Random Forest memproses nilai fitur tersebut."),
        ("4", "Hasil", "Sistem menampilkan Phishing atau Legitimate."),
    ]

    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="step">
                    <div class="step-number">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="muted">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">🧩 Empat Fitur Model</div>',
        unsafe_allow_html=True,
    )

    feature_cols = st.columns(4)
    feature_descriptions = [
        ("Brand Detected", "Apakah indikasi nama brand ditemukan pada URL."),
        ("Auth Detected", "Apakah keyword autentikasi ditemukan pada URL."),
        ("Brand-Domain Similarity", "Kemiripan nama brand dengan root domain."),
        ("Subdomain Count", "Jumlah bagian subdomain pada URL."),
    ]

    for col, (title, desc) in zip(feature_cols, feature_descriptions):
        with col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-title">{title}</div>
                    <div class="muted">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info(
        "Catatan: hasil klasifikasi merupakan keluaran model Random Forest "
        "dan bukan jaminan keamanan absolut terhadap suatu URL."
    )


# =========================================================
# DAFTAR URL / DATASET
# =========================================================

st.markdown(
    '<div class="section-title">📚 Contoh Sampel URL (20 baris)</div>',
    unsafe_allow_html=True,
)

with st.expander("Lihat sampel URL dari dataset"):
    if url_dataset is not None:
        src = url_dataset.attrs.get("_source_file", "url_dataset.csv")
        # Tampilkan langsung 20 URL sampel tanpa penjelasan tambahan
        # Ensure we explicitly take the first 20 rows
        sample_df = url_dataset.iloc[:20].copy().reset_index(drop=True)
        # Tampilkan numbering mulai dari 1 sampai 20
        sample_df.index = sample_df.index + 1
        st.dataframe(sample_df)

        if "url" in sample_df.columns:
            sample_text = "\n".join(sample_df["url"].astype(str).tolist())
            st.text_area("Salin 20 URL sampel", value=sample_text, height=220)
    else:
        st.write(
            "Tidak ditemukan file `url_dataset.csv` di direktori aplikasi. "
            "Untuk menampilkan daftar URL yang sebenarnya, siapkan file CSV dengan kolom `url` "
            "dan opsional kolom `label`, lalu letakkan di folder aplikasi."
        )
        st.write(
            "Jika Anda hanya ingin menunjukkan data contoh untuk sidang, beri label ini sebagai "
            "Contoh Sampel URL atau Sample URL. "
            "Ini menunjukkan bahwa yang ditampilkan adalah subset, bukan keseluruhan data."
        )

        example_rows = []
        if isinstance(official_domains, dict):
            example_rows.extend(list(official_domains.keys())[:10])
        if isinstance(bank_brands, (list, tuple, set)):
            example_rows.extend([str(x) for x in list(bank_brands)[:10]])
        elif isinstance(bank_brands, dict):
            example_rows.extend([str(x) for x in list(bank_brands.keys())[:10]])

        if example_rows:
            st.write("Contoh domain atau brand yang digunakan model (bukan URL lengkap):")
            for item in example_rows[:20]:
                st.markdown(f"- {item}")


st.markdown(
    '<div class="footer">PhishGuard • Banking URL Phishing Detection • Random Forest</div>',
    unsafe_allow_html=True,
)
