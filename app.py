import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fuzzy Logic", layout="wide")

st.title("Fuzzy Logic - Aplikasi Fuzzy dengan Streamlit")
st.sidebar.header("Menu Navigasi")
section = st.sidebar.selectbox("Pilih Bagian", ["Lalu Lintas", "IPK"])


# Common utility

def defuzzify_centroid(x: np.ndarray, membership: np.ndarray) -> float:
    if np.sum(membership) == 0:
        return 0.0
    return np.sum(x * membership) / np.sum(membership)


# Traffic fuzzy functions

def membership_sedikit(x: float) -> float:
    if x <= 0:
        return 1.0
    if x < 80:
        return (80 - x) / 80
    return 0.0


def membership_sedang(x: float) -> float:
    if x <= 40 or x >= 160:
        return 0.0
    if x <= 100:
        return (x - 40) / 60
    return (160 - x) / 60


def membership_banyak(x: float) -> float:
    if x <= 120:
        return 0.0
    if x < 200:
        return (x - 120) / 80
    return 1.0


def output_lancar(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.minimum((50 - x) / 50, 1.0), 0.0)


def output_padat(x: np.ndarray) -> np.ndarray:
    left = np.maximum(np.minimum((x - 30) / 20, 1.0), 0.0)
    right = np.maximum(np.minimum((70 - x) / 20, 1.0), 0.0)
    return np.minimum(left + right, 1.0)


def output_macet(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.minimum((x - 50) / 50, 1.0), 0.0)


# IPK fuzzy functions

def membership_rendah(x: float) -> float:
    if x <= 0:
        return 1.0
    if x < 2.5:
        return (2.5 - x) / 2.5
    return 0.0


def membership_sedang_ipk(x: float) -> float:
    if x <= 2.0 or x >= 3.5:
        return 0.0
    if x <= 2.75:
        return (x - 2.0) / 0.75
    return (3.5 - x) / 0.75


def membership_tinggi(x: float) -> float:
    if x <= 3.0:
        return 0.0
    if x < 4.0:
        return (x - 3.0) / 1.0
    return 1.0


def output_tidak_layak(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.minimum((40 - x) / 40, 1.0), 0.0)


def output_dipertimbangkan(x: np.ndarray) -> np.ndarray:
    left = np.maximum(np.minimum((x - 20) / 20, 1.0), 0.0)
    right = np.maximum(np.minimum((60 - x) / 20, 1.0), 0.0)
    return np.minimum(left + right, 1.0)


def output_layak(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.minimum((x - 40) / 30, 1.0), 0.0)


# Rendering functions

def render_traffic_section():
    st.header("Prediksi Kondisi Lalu Lintas")
    st.write(
        "Masukkan `Jumlah Kendaraan` untuk memprediksi kondisi jalan berdasarkan logika fuzzy: `Lancar`, `Padat`, dan `Macet`."
    )

    jumlah = st.slider("Jumlah Kendaraan", min_value=0, max_value=200, value=60, step=1)

    mu_sedikit = membership_sedikit(jumlah)
    mu_sedang = membership_sedang(jumlah)
    mu_banyak = membership_banyak(jumlah)

    st.subheader("Derajat Keanggotaan Input")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sedikit", f"{mu_sedikit:.2f}")
    col2.metric("Sedang", f"{mu_sedang:.2f}")
    col3.metric("Banyak", f"{mu_banyak:.2f}")

    rule_lancar = mu_sedikit
    rule_padat = mu_sedang
    rule_macet = mu_banyak

    x_output = np.linspace(0, 100, 501)
    base_lancar = output_lancar(x_output)
    base_padat = output_padat(x_output)
    base_macet = output_macet(x_output)

    agg_lancar = np.minimum(rule_lancar, base_lancar)
    agg_padat = np.minimum(rule_padat, base_padat)
    agg_macet = np.minimum(rule_macet, base_macet)
    aggregated = np.maximum(np.maximum(agg_lancar, agg_padat), agg_macet)

    hasil_krisp = defuzzify_centroid(x_output, aggregated)

    label_terkuat = "Lancar"
    if max(mu_sedikit, mu_sedang, mu_banyak) == mu_sedang:
        label_terkuat = "Padat"
    elif max(mu_sedikit, mu_sedang, mu_banyak) == mu_banyak:
        label_terkuat = "Macet"

    st.subheader("Hasil Fuzzy")
    
    # Create results table
    hasil_data = {
        "Kategori": ["Lancar", "Padat", "Macet"],
        "Derajat Keanggotaan Input": [f"{mu_sedikit:.2f}", f"{mu_sedang:.2f}", f"{mu_banyak:.2f}"],
        "Rule": [f"{rule_lancar:.2f}", f"{rule_padat:.2f}", f"{rule_macet:.2f}"]
    }
    df_hasil = pd.DataFrame(hasil_data)
    st.table(df_hasil)
    
    st.write(f"- **Kondisi dominan**: {label_terkuat}")
    st.write(f"- **Nilai defuzzifikasi**: {hasil_krisp:.1f}")

    if hasil_krisp < 40:
        st.success("Hasil: Lancar")
    elif hasil_krisp < 70:
        st.warning("Hasil: Padat")
    else:
        st.error("Hasil: Macet")

    x_input = np.linspace(0, 200, 501)
    input_sedikit = np.array([membership_sedikit(x) for x in x_input])
    input_sedang = np.array([membership_sedang(x) for x in x_input])
    input_banyak = np.array([membership_banyak(x) for x in x_input])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(x_input, input_sedikit, label="Sedikit", color="#2ca02c")
    axes[0].plot(x_input, input_sedang, label="Sedang", color="#ff7f0e")
    axes[0].plot(x_input, input_banyak, label="Banyak", color="#d62728")
    axes[0].axvline(jumlah, color="black", linestyle="--", label=f"Jumlah = {jumlah}")
    axes[0].set_title("Fungsi Keanggotaan Input: Jumlah Kendaraan")
    axes[0].set_xlabel("Jumlah Kendaraan")
    axes[0].set_ylabel("Derajat Keanggotaan")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(x_output, base_lancar, label="Lancar", color="#2ca02c")
    axes[1].plot(x_output, base_padat, label="Padat", color="#ff7f0e")
    axes[1].plot(x_output, base_macet, label="Macet", color="#d62728")
    axes[1].fill_between(x_output, aggregated, color="#9467bd", alpha=0.4, label="Agregasi Output")
    axes[1].axvline(hasil_krisp, color="black", linestyle="--", label=f"Defuzzifikasi = {hasil_krisp:.1f}")
    axes[1].set_title("Fungsi Keanggotaan Output dan Agregasi")
    axes[1].set_xlabel("Skala Kondisi (0=Lancar, 100=Macet)")
    axes[1].set_ylabel("Derajat Keanggotaan")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    st.pyplot(fig)

    st.markdown(
        "---\n"
        "### Interpretasi Hasil Lalu Lintas:\n"
        "- `Lancar`: Jalan probabel bebas hambatan, kecepatan kendaraan stabil, dan antrian minimal.\n"
        "- `Padat`: Kondisi jalan mulai sesak, kendaraan melaju lambat, dan kemungkinan antrean di persimpangan meningkat.\n"
        "- `Macet`: Kondisi lalu lintas sangat padat, kecepatan rendah, dan kemungkinan kemacetan panjang tinggi.\n"
        "- Nilai defuzzifikasi di bawah 40 menunjukkan kondisi lancar kuat, 40-70 menunjukkan padat, dan di atas 70 menunjukkan macet.\n"
    )


def render_ipk_section():
    st.header("Prediksi Kelayakan IPK")
    st.write(
        "Masukkan `IPK` untuk menentukan status kelayakan: `Tidak Layak`, `Dipertimbangkan`, dan `Layak`."
    )

    ipk = st.slider("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)

    mu_tidak = membership_rendah(ipk)
    mu_dipertimbangkan = membership_sedang_ipk(ipk)
    mu_layak = membership_tinggi(ipk)

    st.subheader("Derajat Keanggotaan Input")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tidak Layak", f"{mu_tidak:.2f}")
    col2.metric("Dipertimbangkan", f"{mu_dipertimbangkan:.2f}")
    col3.metric("Layak", f"{mu_layak:.2f}")

    rule_tidak = mu_tidak
    rule_dipertimbangkan = mu_dipertimbangkan
    rule_layak = mu_layak

    x_output = np.linspace(0, 100, 501)
    base_tidak = output_tidak_layak(x_output)
    base_dipertimbangkan = output_dipertimbangkan(x_output)
    base_layak = output_layak(x_output)

    agg_tidak = np.minimum(rule_tidak, base_tidak)
    agg_dipertimbangkan = np.minimum(rule_dipertimbangkan, base_dipertimbangkan)
    agg_layak = np.minimum(rule_layak, base_layak)
    aggregated = np.maximum(np.maximum(agg_tidak, agg_dipertimbangkan), agg_layak)

    hasil_krisp = defuzzify_centroid(x_output, aggregated)

    label_terkuat = "Tidak Layak"
    if max(mu_tidak, mu_dipertimbangkan, mu_layak) == mu_dipertimbangkan:
        label_terkuat = "Dipertimbangkan"
    elif max(mu_tidak, mu_dipertimbangkan, mu_layak) == mu_layak:
        label_terkuat = "Layak"

    st.subheader("Hasil Fuzzy")
    
    # Create results table
    hasil_data = {
        "Kategori": ["Tidak Layak", "Dipertimbangkan", "Layak"],
        "Derajat Keanggotaan Input": [f"{mu_tidak:.2f}", f"{mu_dipertimbangkan:.2f}", f"{mu_layak:.2f}"],
        "Rule": [f"{rule_tidak:.2f}", f"{rule_dipertimbangkan:.2f}", f"{rule_layak:.2f}"]
    }
    df_hasil = pd.DataFrame(hasil_data)
    st.table(df_hasil)
    
    st.write(f"- **Kelayakan dominan**: {label_terkuat}")
    st.write(f"- **Nilai defuzzifikasi**: {hasil_krisp:.1f}")

    if hasil_krisp < 35:
        st.error("Hasil: Tidak Layak")
    elif hasil_krisp < 65:
        st.warning("Hasil: Dipertimbangkan")
    else:
        st.success("Hasil: Layak")

    x_input = np.linspace(0, 4, 501)
    input_tidak = np.array([membership_rendah(x) for x in x_input])
    input_dipertimbangkan = np.array([membership_sedang_ipk(x) for x in x_input])
    input_layak = np.array([membership_tinggi(x) for x in x_input])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(x_input, input_tidak, label="Tidak Layak", color="#d62728")
    axes[0].plot(x_input, input_dipertimbangkan, label="Dipertimbangkan", color="#ff7f0e")
    axes[0].plot(x_input, input_layak, label="Layak", color="#2ca02c")
    axes[0].axvline(ipk, color="black", linestyle="--", label=f"IPK = {ipk:.2f}")
    axes[0].set_title("Fungsi Keanggotaan Input: IPK")
    axes[0].set_xlabel("IPK")
    axes[0].set_ylabel("Derajat Keanggotaan")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(x_output, base_tidak, label="Tidak Layak", color="#d62728")
    axes[1].plot(x_output, base_dipertimbangkan, label="Dipertimbangkan", color="#ff7f0e")
    axes[1].plot(x_output, base_layak, label="Layak", color="#2ca02c")
    axes[1].fill_between(x_output, aggregated, color="#9467bd", alpha=0.4, label="Agregasi Output")
    axes[1].axvline(hasil_krisp, color="black", linestyle="--", label=f"Defuzzifikasi = {hasil_krisp:.1f}")
    axes[1].set_title("Fungsi Keanggotaan Output dan Agregasi")
    axes[1].set_xlabel("Skala Kelayakan (0=Tidak Layak, 100=Layak)")
    axes[1].set_ylabel("Derajat Keanggotaan")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    st.pyplot(fig)

    st.markdown(
        "---\n"
        "### Interpretasi Hasil IPK:\n"
        "- `Tidak Layak`: IPK terlalu rendah untuk diterima secara aman, perlu peningkatan sebelum masuk pertimbangan.\n"
        "- `Dipertimbangkan`: IPK berada di kisaran moderat; pelamar layak dievaluasi lebih lanjut dengan faktor lain.\n"
        "- `Layak`: IPK cukup baik dan menunjukkan potensi tinggi untuk diterima atau melanjutkan ke tahap selanjutnya.\n"
        "- Nilai defuzzifikasi di bawah 35 menunjukkan hasil tidak layak kuat, 35-65 menunjukkan status dipertimbangkan, dan di atas 65 menunjukkan layak.\n"
    )


if section == "Lalu Lintas":
    render_traffic_section()
else:
    render_ipk_section()
