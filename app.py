# app.py — Streamlit application สำหรับทำนายการยกเลิกบัตรเครดิต

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ===== การตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="ระบบทำนายการยกเลิกบัตรเครดิต",
    page_icon="💳",          
    layout="centered",        
    initial_sidebar_state="expanded"
)

# ===== โหลดโมเดล =====
@st.cache_resource
def load_model():
    """โหลดโมเดล Random Forest ที่เราเทรนไว้"""
    model = joblib.load("model_artifacts/credit_card_churn_model.pkl")
    return model

with st.spinner("กำลังโหลดสมอง AI..."):
    model = load_model()

# ===== Sidebar: ข้อมูลเกี่ยวกับโมเดล =====
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดลนี้")
    st.write("**ประเภท:** Random Forest Classifier")
    st.write("**ความแม่นยำ (Accuracy):** ~95.8%")
    st.write("**เป้าหมาย:** ทำนายว่าลูกค้าจะยกเลิกบัตรหรือไม่")

    st.divider() 

    st.subheader("⚠️ ข้อควรระวัง")
    st.warning(
        "ผลลัพธ์นี้เป็นการประเมินความน่าจะเป็นเบื้องต้นจาก AI เท่านั้น "
        "เพื่อช่วยให้ฝ่ายการตลาดตัดสินใจออกโปรโมชั่นรักษาลูกค้า"
    )

# ===== ส่วนหลัก: Header =====
st.title("💳 ระบบทำนายการยกเลิกบัตรเครดิต")
st.markdown("""
กรอกข้อมูลพฤติกรรมการใช้จ่ายของลูกค้าด้านล่าง ระบบจะคำนวณความเสี่ยงว่า
**"ลูกค้าคนนี้มีแนวโน้มจะยกเลิกบัตรเครดิตของเราหรือไม่"**
""")

st.divider()

# ===== ส่วนรับ Input =====
st.subheader("📋 ข้อมูลลูกค้าและการใช้จ่าย")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**ข้อมูลทั่วไป**")
    customer_age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=45)
    gender = st.selectbox("เพศ",["Male", "Female"])