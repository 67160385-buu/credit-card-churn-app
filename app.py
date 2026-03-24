# app.py — Streamlit application สำหรับคาดการณ์ลูกค้าบัตรเครดิต (Credit Card Churn/Segmentation)

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===== การตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="Credit Card Customer Prediction",
    page_icon="💳",          
    layout="centered",        
    initial_sidebar_state="expanded"
)

# ===== โหลดโมเดลและข้อมูล =====
@st.cache_resource
def load_model():
    """โหลด Model, Scaler และ Columns"""
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

# โหลดโมเดล
with st.spinner("กำลังโหลดระบบประเมินลูกค้า..."):
    model, scaler, model_columns = load_model()

# ===== Sidebar: ข้อมูลเกี่ยวกับโมเดล =====
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับระบบนี้")
    st.write("**โปรเจค:** Credit Card Customer Analytics")
    st.write("**เป้าหมาย:** ทำนายความเสี่ยงที่ลูกค้าจะยกเลิกบัตร (Churn) หรือการจัดกลุ่มลูกค้า")
    
    st.divider()
    st.subheader("💡 คำแนะนำการใช้งาน")
    st.write("ลองปรับเปลี่ยนค่า **ยอดใช้จ่ายรวม (Total Transaction Amount)** หรือ **จำนวนครั้งที่รูดบัตร** เพื่อดูว่าความเสี่ยงของลูกค้าเปลี่ยนไปอย่างไร")

# ===== ส่วนหลัก: Header =====
st.title("💳 ระบบวิเคราะห์พฤติกรรมลูกค้าบัตรเครดิต")
st.markdown("""
กรอกข้อมูลพฤติกรรมการใช้จ่ายของลูกค้าด้านล่าง ระบบ AI จะทำการประเมินว่าลูกค้าท่านนี้ 
**มีแนวโน้มที่จะใช้งานต่อ** หรือ **มีความเสี่ยงที่จะยกเลิกบัตร (Churn)**
""")

st.divider()

# ===== ส่วนรับ Input =====
st.subheader("📋 ข้อมูลและพฤติกรรมของลูกค้า")

col1, col2 = st.columns(2)

with col1:
    customer_age = st.number_input("อายุลูกค้า (Age)", min_value=18, max_value=100, value=35)
    
    gender = st.selectbox("เพศ (Gender)", ["M", "F"])
    
    income_category = st.selectbox(
        "รายได้ต่อปี (Income Category)", 
        ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"]
    )
    
    card_category = st.selectbox("ประเภทบัตร (Card Category)", ["Blue", "Silver", "Gold", "Platinum"])

with col2:
    months_on_book = st.number_input("ระยะเวลาที่เป็นลูกค้า (เดือน)", min_value=0, value=36)
    
    total_trans_amt = st.number_input(
        "ยอดใช้จ่ายรวม 12 เดือนล่าสุด ($)", 
        min_value=0, value=4500, step=500
    )
    
    total_trans_ct = st.number_input(
        "จำนวนครั้งที่รูดบัตร (ครั้ง)", 
        min_value=0, value=65, step=5
    )
    
    revolving_bal = st.number_input(
        "ยอดหนี้คงค้าง (Revolving Balance)", 
        min_value=0, value=1500, step=100
    )

st.divider()

# ===== ปุ่มทำนายและแสดงผล =====
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_button = st.button("🔍 วิเคราะห์ความเสี่ยงลูกค้า", use_container_width=True, type="primary")

if predict_button:
    # 1. รวบรวมข้อมูลที่กรอก
    input_df = pd.DataFrame({
        'Customer_Age': [customer_age],
        'Gender': [gender],
        'Income_Category': [income_category],
        'Card_Category': [card_category],
        'Months_on_book': [months_on_book],
        'Total_Trans_Amt': [total_trans_amt],
        'Total_Trans_Ct': [total_trans_ct],
        'Total_Revolving_Bal': [revolving_bal]
    })

    with st.spinner("กำลังประมวลผลด้วย AI..."):
        # 2. แปลงข้อมูล Categorical เป็น One-Hot
        input_encoded = pd.get_dummies(input_df)
        
        # 3. จัดคอลัมน์ให้ตรงกับตอน Train
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 4. สเกลข้อมูล
        input_scaled = scaler.transform(input_aligned)

        # 5. ทำนายผล (สมมติ 1 = เสี่ยงยกเลิก, 0 = ใช้งานต่อ)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_
