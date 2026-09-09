import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 리포지토리 내부 폰트 파일 경로 지정
font_path = "fonts/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)

# 전역 기본 폰트 적용 및 마이너스 기호 깨짐 방지
plt.rc("font", family=font_prop.get_name())
plt.rcParams["axes.unicode_minus"] = False


# ── 1. 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="가죽 이상 탐지 시스템",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 가죽 이상 탐지 시스템")
st.caption("업로드된 이미지 또는 웹캠 촬영 이미지를 분석하여 가죽의 결함 여부를 판정합니다.")

# ── 설정 상수 ──────────────────────────────────────────────────────
MODEL_PATH     = "./weights/leather_model.keras"
INPUT_IMG_SIZE = (224, 224)
CLASSES        = ["정상", "불량"]


# ── 2. 모델 로드 (@st.cache_resource로 캐싱) ────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# ── 기존 전처리 및 추론 로직 (그대로 유지) ──────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model, pil_img):
    arr   = preprocess(pil_img)
    prob  = float(model.predict(arr, verbose=0)[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ── 3. 이미지 입력 선택 UI ─────────────────────────────────────────
input_mode = st.radio(
    "이미지 입력 방식을 선택하세요",
    ["파일 업로드", "카메라 촬영"],
    horizontal=True
)

pil_img = None

if input_mode == "파일 업로드":
    uploaded_file = st.file_uploader(
        "가죽 이미지를 선택하세요",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)

elif input_mode == "카메라 촬영":
    camera_file = st.camera_input("카메라로 가죽을 촬영하세요")
    if camera_file is not None:
        pil_img = Image.open(camera_file)

# 입력 이미지 미리보기 표시
if pil_img is not None:
    st.image(pil_img, caption="입력된 이미지", use_column_width=True)

    # ── 4. 검사 실행 ───────────────────────────────────────────────
    if st.button("검사 시작", type="primary"):
        model = load_model()
        
        if model is not None:
            with st.spinner("이미지 분석 중..."):
                label, defect_prob = predict(model, pil_img)
                normal_prob = 1.0 - defect_prob

            st.divider()

            # ── 5. 결과 표시 ───────────────────────────────────────
            if label == "정상":
                st.success(f"판정 결과: **{label}**")
            else:
                st.error(f"판정 결과: **{label}**")

            # 수치 표시
            col1, col2 = st.columns(2)
            col1.metric("정상 확률", f"{normal_prob:.1%}")
            col2.metric("불량 확률", f"{defect_prob:.1%}")

            # 불량 확률 시각화
            st.write("**불량 확률 그래프**")
            st.progress(defect_prob)
            
            # 비교 막대 차트
            chart_data = pd.DataFrame(
                {"확률": [normal_prob, defect_prob]},
                index=["정상", "불량"]
            )
            st.bar_chart(chart_data)