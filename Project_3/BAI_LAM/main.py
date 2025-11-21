import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* --- Chủ đề màu xanh tổng thể --- */
    :root {
        --main-blue: #1e88e5;
        --light-blue: #e3f2fd;
        --dark-blue: #1565c0;
    }

    /* --- Nền tổng thể --- */
    .stApp {
        background-color: var(--light-blue);
    }

    /* --- Tiêu đề chính --- */
    h1, h2, h3 {
        color: var(--dark-blue) !important;
    }

    /* --- Sidebar style --- */
    section[data-testid="stSidebar"] {
        background-color: #f4faff;
        border-right: 2px solid #cfe1f8;
    }

    /* --- Card trắng cho khối nội dung --- */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        border-radius: 10px;
        background-color: #ffffffcc;
        backdrop-filter: blur(3px);
        padding-left: 2rem;
        padding-right: 2rem;
        box-shadow: 0 0 15px rgba(0,0,0,0.05);
    }

    /* --- Button màu xanh dương --- */
    div.stButton > button {
        background-color: var(--main-blue);
        color: white;
        border-radius: 6px;
        border: 1px solid var(--dark-blue);
        padding: 0.5rem 1.5rem;
        font-size: 1.05rem;
    }
    div.stButton > button:hover {
        background-color: var(--dark-blue);
        color: white;
        border-color: var(--dark-blue);
    }

    /* --- SelectBox & NumberInput border sáng --- */
    .stSelectbox, .stNumberInput {
        border-radius: 6px !important;
    }

    /* --- Dataframe border đẹp hơn --- */
    .stDataFrame, .stTable {
        border: 1px solid #bbdefb !important;
        border-radius: 8px !important;
    }

    /* --- Thẻ success/error đẹp hơn --- */
    .stSuccess {
        background-color: #d0ebff !important;
        color: #084c8d !important;
        border-left: 6px solid var(--main-blue);
    }
    .stError {
        background-color: #ffdce0 !important;
        color: #7a0000 !important;
        border-left: 6px solid #d32f2f;
    }

    </style>
""", unsafe_allow_html=True)




# ======================= PATHS =======================

MODEL_PATH = "linear_model.pkl"   # model dự đoán giá
CLUSTER_MODEL_PATH = "cluster.pkl"  # model phát hiện bất thường
ENCODERS_PATH = "encoders.pkl"   # dict encoders cho các cột category

# ======================= LOAD MODELS & ENCODERS =======================

@st.cache_resource
def load_models_and_encoders():
    with open(MODEL_PATH, "rb") as f:
        price_model = pickle.load(f)
    with open(CLUSTER_MODEL_PATH, "rb") as f:
        cluster_model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return price_model, cluster_model, encoders

try:
    loaded_model, cluster_model, encoders = load_models_and_encoders()
except Exception as e:
    loaded_model, cluster_model, encoders = None, None, None
    st.error(f"Không thể load model hoặc encoders: {e}")

# Các cột category & cột bắt buộc
cat_cols = ["Thương hiệu", "Dòng xe", "Loại xe", "Xuất xứ"]

required_columns_price = [
    "Thương hiệu",
    "Dòng xe",
    "Loại xe",
    "Xuất xứ",
    "Khoảng giá min",
    "Khoảng giá max",
    "Năm đăng ký",
    "Số Km đã đi",
]

required_columns_cluster = [
    "Thương hiệu",
    "Dòng xe",
    "Loại xe",
    "Xuất xứ",
    "Khoảng giá min",
    "Khoảng giá max",
    "Năm đăng ký",
    "Số Km đã đi",
    "Giá",
]


def predict_car_price(input_df):
    """input_df đã encode, chứa đủ required_columns_price."""
    missing_cols = set(required_columns_price) - set(input_df.columns)
    if missing_cols:
        raise ValueError(f"Thiếu cột: {missing_cols}")
    preds = loaded_model.predict(input_df[required_columns_price])
    return preds.flatten()


def predict_anomaly(input_df):
    """input_df đã encode, chứa đủ required_columns_cluster.
       Trả về 1: bất thường, 0: bình thường.
    """
    missing_cols = set(required_columns_cluster) - set(input_df.columns)
    if missing_cols:
        raise ValueError(f"Thiếu cột: {missing_cols}")
    preds = cluster_model.predict(input_df[required_columns_cluster])
    return preds.flatten()


# ======================= SIDEBAR =======================

st.sidebar.title("Hệ thống")

page = st.sidebar.radio(
    "Chọn chức năng",
    ("Dự đoán giá xe", "Phát hiện bất thường")
)

st.sidebar.title("Dữ liệu xe máy")
uploaded_file = st.sidebar.file_uploader(
    "Upload file dữ liệu (CSV / Excel)",
    type=["csv", "xlsx", "xls"]
)

if loaded_model is None or cluster_model is None or encoders is None:
    st.warning("Model hoặc encoders chưa load được, vui lòng kiểm tra 'linear_model.pkt', 'cluster.pkl' và 'encoders.pkl'.")
    st.stop()

# ======================= LOAD FILE CHUNG =======================

df = None
if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

# ======================= PAGE 1: DỰ ĐOÁN GIÁ XE =======================

if page == "Dự đoán giá xe":
    st.title("Hệ thống dự đoán giá xe máy")

    if df is None:
        st.info("Hãy upload dữ liệu ở sidebar để bắt đầu.")
        st.stop()

    st.subheader("Dữ liệu gốc (dùng để lấy lựa chọn cho người dùng)")
    st.dataframe(df)

    required_cols_for_ui = ["Thương hiệu", "Dòng xe", "Loại xe", "Xuất xứ"]
    missing = [c for c in required_cols_for_ui if c not in df.columns]

    if missing:
        st.error(f"Thiếu các cột trong dữ liệu upload: {', '.join(missing)}")
    else:
        # ========== BƯỚC 1: CHỌN THƯƠNG HIỆU ==========
        st.subheader("Bước 1: Chọn thương hiệu")
        brand_list = df["Thương hiệu"].dropna().unique()
        selected_brand = st.selectbox("Thương hiệu", options=brand_list)

        df_brand = df[df["Thương hiệu"] == selected_brand]

        # ========== BƯỚC 2: CHỌN DÒNG XE ==========
        st.subheader("Bước 2: Chọn Dòng xe")
        dong_xe_list = df_brand["Dòng xe"].dropna().unique()
        selected_dong_xe = st.selectbox("Dòng xe", options=dong_xe_list)

        df_dong = df_brand[df_brand["Dòng xe"] == selected_dong_xe]

        # ========== BƯỚC 3: CHỌN LOẠI XE ==========
        st.subheader("Bước 3: Chọn Loại xe")
        loai_xe_list = df_dong["Loại xe"].dropna().unique()
        selected_loai_xe = st.selectbox("Loại xe", options=loai_xe_list)

        df_loai = df_dong[df_dong["Loại xe"] == selected_loai_xe]

        # ====== BƯỚC 4: CHỌN XUẤT XỨ ======
        st.subheader("Bước 4: Chọn Xuất xứ")
        xuat_xu_list = df_loai["Xuất xứ"].dropna().unique()
        xuat_xu_list = [x for x in xuat_xu_list if x != "Đang cập nhật"]

        selected_xuat_xu = st.selectbox("Xuất xứ", options=xuat_xu_list)
        df_xuatxu = df_loai[df_loai["Xuất xứ"] == selected_xuat_xu]

        # ====== BƯỚC 5: CHỌN KHOẢNG GIÁ ======
        st.subheader("Bước 5: Chọn khoảng giá (triệu VND)")
        if "Giá" in df_xuatxu.columns:
            try:
                min_price = float(df_xuatxu["Giá"].min() / 1_000_000)
                max_price = float(df_xuatxu["Giá"].max() / 1_000_000)
            except Exception:
                min_price, max_price = 0.0, 500.0
        else:
            min_price, max_price = 0.0, 500.0

        gia_min = st.number_input(
            "Giá tối thiểu (triệu)",
            min_value=0.0,
            max_value=max_price if max_price > 0 else 9999.0,
            value=min_price if min_price >= 0 else 0.0,
            step=0.1
        )

        gia_max = st.number_input(
            "Giá tối đa (triệu)",
            min_value=gia_min,
            max_value=max_price if max_price > 0 else 9999.0,
            value=max_price if max_price >= gia_min else gia_min,
            step=0.1
        )

        # ====== BƯỚC 6: NĂM ĐĂNG KÝ ======
        st.subheader("Bước 6: Nhập Năm đăng ký")
        nam_dang_ky = st.number_input(
            "Năm đăng ký",
            min_value=2000,
            max_value=2025,
            value=2020,
            step=1
        )

        # ====== BƯỚC 7: SỐ KM ĐÃ ĐI ======
        st.subheader("Bước 7: Nhập Số Km đã đi")
        so_km_da_di = st.number_input(
            "Số Km đã đi",
            min_value=0,
            max_value=500_000,
            value=50_000,
            step=1_000
        )

        st.success(
            f"Đã chọn: "
            f"Thương hiệu **{selected_brand}**, "
            f"Dòng xe **{selected_dong_xe}**, "
            f"Loại xe **{selected_loai_xe}**, "
            f"Xuất xứ **{selected_xuat_xu}**, "
            f"Khoảng giá **{gia_min} – {gia_max} triệu**, "
            f"Năm đăng ký **{nam_dang_ky}**, "
            f"Số Km đã đi **{so_km_da_di}**"
        )

        # ====== INPUT RAW ======
        input_df = pd.DataFrame({
            "Thương hiệu":    [selected_brand],
            "Dòng xe":        [selected_dong_xe],
            "Loại xe":        [selected_loai_xe],
            "Xuất xứ":        [selected_xuat_xu],
            "Khoảng giá min": [gia_min],
            "Khoảng giá max": [gia_max],
            "Năm đăng ký":    [nam_dang_ky],
            "Số Km đã đi":    [so_km_da_di]
        })

        st.subheader("DataFrame đầu vào (chưa encode) - debug")
        st.dataframe(input_df)

        # ====== ENCODE CATEGORY ======
        encoded_input = input_df.copy()
        for col in cat_cols:
            if col not in encoders:
                st.error(f"Không tìm thấy encoder cho cột '{col}' trong encoders.pkl")
                st.stop()

            le = encoders[col]
            val = encoded_input[col].iloc[0]

            if val not in le.classes_:
                st.error(
                    f"Giá trị '{val}' của cột '{col}' không tồn tại trong dữ liệu huấn luyện.\n"
                    "Vui lòng chọn giá trị khác hoặc cập nhật lại model/encoders."
                )
                st.stop()

            encoded_input[col] = le.transform(encoded_input[col])

        st.subheader("DataFrame sau khi encode (debug)")
        st.dataframe(encoded_input)

        # ====== DỰ ĐOÁN GIÁ ======
        if st.button("Dự đoán giá"):
            try:
                y_pred = predict_car_price(encoded_input)
                predicted_price = float(y_pred[0])
                st.success(
                    f"Giá dự đoán: **{predicted_price:,.0f}** "
                    f"triệu"
                )
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")

# ======================= PAGE 2: PHÁT HIỆN BẤT THƯỜNG =======================

elif page == "Phát hiện bất thường":
    st.title("Phát hiện bất thường trong dữ liệu xe")

    if df is None:
        st.info("Hãy upload dữ liệu ở sidebar để bắt đầu.")
        st.stop()

    st.subheader("Dữ liệu gốc")
    st.dataframe(df)

    required_cols_for_ui = ["Thương hiệu", "Dòng xe", "Loại xe", "Xuất xứ"]
    missing = [c for c in required_cols_for_ui if c not in df.columns]

    if missing:
        st.error(f"Thiếu các cột trong dữ liệu upload: {', '.join(missing)}")
    else:
        # ========== BƯỚC 1: CHỌN THƯƠNG HIỆU ==========
        st.subheader("Bước 1: Chọn thương hiệu")
        brand_list = df["Thương hiệu"].dropna().unique()
        selected_brand = st.selectbox("Thương hiệu", options=brand_list)

        df_brand = df[df["Thương hiệu"] == selected_brand]

        # ========== BƯỚC 2: CHỌN DÒNG XE ==========
        st.subheader("Bước 2: Chọn Dòng xe")
        dong_xe_list = df_brand["Dòng xe"].dropna().unique()
        selected_dong_xe = st.selectbox("Dòng xe", options=dong_xe_list)

        df_dong = df_brand[df_brand["Dòng xe"] == selected_dong_xe]

        # ========== BƯỚC 3: CHỌN LOẠI XE ==========
        st.subheader("Bước 3: Chọn Loại xe")
        loai_xe_list = df_dong["Loại xe"].dropna().unique()
        selected_loai_xe = st.selectbox("Loại xe", options=loai_xe_list)

        df_loai = df_dong[df_dong["Loại xe"] == selected_loai_xe]

        # ====== BƯỚC 4: CHỌN XUẤT XỨ ======
        st.subheader("Bước 4: Chọn Xuất xứ")
        xuat_xu_list = df_loai["Xuất xứ"].dropna().unique()
        xuat_xu_list = [x for x in xuat_xu_list if x != "Đang cập nhật"]

        selected_xuat_xu = st.selectbox("Xuất xứ", options=xuat_xu_list)

        # ====== BƯỚC 5: NĂM ĐĂNG KÝ ======
        st.subheader("Bước 5: Nhập Năm đăng ký")
        nam_dang_ky = st.number_input(
            "Năm đăng ký",
            min_value=2000,
            max_value=2025,
            value=2020,
            step=1
        )

        # ====== BƯỚC 6: SỐ KM ĐÃ ĐI ======
        st.subheader("Bước 6: Nhập Số Km đã đi")
        so_km_da_di = st.number_input(
            "Số Km đã đi",
            min_value=0,
            max_value=500_000,
            value=50_000,
            step=1_000
        )

        # ====== BƯỚC 7: NHẬP GIÁ HIỆN TẠI ======
        st.subheader("Bước 7: Nhập Giá hiện tại của xe (triệu VND)")
        gia_input = st.number_input(
            "Giá hiện tại (triệu)",
            min_value=0.0,
            max_value=9999.0,
            value=50.0,
            step=0.1
        )

        st.info("Trong mô hình bất thường, 'Khoảng giá min' = 30 và 'Khoảng giá max' = 100 (triệu) được gán mặc định.")

        # ====== TẠO INPUT RAW CHO MODEL CLUSTER ======
        input_cluster_df = pd.DataFrame({
            "Thương hiệu":    [selected_brand],
            "Dòng xe":        [selected_dong_xe],
            "Loại xe":        [selected_loai_xe],
            "Xuất xứ":        [selected_xuat_xu],
            "Khoảng giá min": [30.0],   # default
            "Khoảng giá max": [100.0],  # default
            "Năm đăng ký":    [nam_dang_ky],
            "Số Km đã đi":    [so_km_da_di],
            "Giá":            [gia_input]   # cũng đang ở đơn vị "triệu" nếu model train vậy
        })

        st.subheader("DataFrame đầu vào (chưa encode) - debug")
        st.dataframe(input_cluster_df)

        # ====== ENCODE CATEGORY ======
        encoded_cluster_input = input_cluster_df.copy()
        for col in cat_cols:
            if col not in encoders:
                st.error(f"Không tìm thấy encoder cho cột '{col}' trong encoders.pkl")
                st.stop()

            le = encoders[col]
            val = encoded_cluster_input[col].iloc[0]

            if val not in le.classes_:
                st.error(
                    f"Giá trị '{val}' của cột '{col}' không tồn tại trong dữ liệu huấn luyện.\n"
                    "Vui lòng chọn giá trị khác hoặc cập nhật lại model/encoders."
                )
                st.stop()

            encoded_cluster_input[col] = le.transform(encoded_cluster_input[col])

        st.subheader("DataFrame sau khi encode (debug)")
        st.dataframe(encoded_cluster_input)

        # ====== DỰ ĐOÁN BẤT THƯỜNG ======
        if st.button("Kiểm tra bất thường"):
            try:
                y_pred = predict_anomaly(encoded_cluster_input)
                result = int(y_pred[0])

                if result == 1:
                    st.error("⚠️ Mô hình phát hiện: **BẤT THƯỜNG** (anomaly). Giá này có thể không phù hợp với mẫu dữ liệu thông thường.")
                else:
                    st.success("✅ Mô hình phát hiện: **BÌNH THƯỜNG** (normal). Giá này không bị xem là bất thường theo mô hình.")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán bất thường: {e}")
