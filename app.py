import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Customer Prediction", page_icon="💳", layout="centered")

# ===== โหลดโมเดล =====
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl") 
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

with st.spinner("กำลังโหลดระบบประเมินลูกค้า..."):
    model, scaler, model_columns = load_model()

# ===== ส่วนหน้าเว็บ =====
st.title("💳 ระบบคาดการณ์ลูกค้าบัตรเครดิต")
st.markdown("กรอกข้อมูลพฤติกรรมการใช้จ่ายของลูกค้า เพื่อประเมินแนวโน้ม")
st.divider()

col1, col2 = st.columns(2)

with col1:
    customer_age = st.number_input("อายุลูกค้า (Age)", min_value=18, max_value=100, value=35)
    gender = st.selectbox("เพศ (Gender)", ["M", "F"])
    income_category = st.selectbox("รายได้ต่อปี", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"])
    card_category = st.selectbox("ประเภทบัตร", ["Blue", "Silver", "Gold", "Platinum"])

with col2:
    months_on_book = st.number_input("ระยะเวลาที่เป็นลูกค้า (เดือน)", min_value=0, value=36)
    total_trans_amt = st.number_input("ยอดใช้จ่ายรวม 12 เดือน ($)", min_value=0, value=4500, step=500)
    total_trans_ct = st.number_input("จำนวนครั้งที่รูดบัตร", min_value=0, value=65, step=5)
    revolving_bal = st.number_input("ยอดหนี้คงค้าง ($)", min_value=0, value=1500, step=100)

predict_button = st.button("🔍 วิเคราะห์ข้อมูลลูกค้า", use_container_width=True, type="primary")

# ===== เมื่อกดปุ่มทำนาย =====
if predict_button:
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

    # เตรียมข้อมูลให้ตรงกับตอน Train
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
    input_scaled = scaler.transform(input_aligned)

    # ทำนายผล
    prediction = model.predict(input_scaled)[0]
    
    # ระบบป้องกัน Error กรณีโมเดลไม่มี predict_proba
    has_prob = False
    try:
        probabilities = model.predict_proba(input_scaled)[0]
        prob_stay = probabilities[0]
        prob_churn = probabilities[1]
        has_prob = True
    except AttributeError:
        pass # ถ้า Error ให้ข้ามไปเลย ไม่ต้องแสดงเปอร์เซ็นต์

    st.subheader("📊 ผลการวิเคราะห์ลูกค้า")

    # แสดงผลลัพธ์
    if str(prediction) == "1":
        st.error("### ⚠️ ลูกค้ากลุ่มเสี่ยง (High Risk)")
        st.write("ระบบประเมินว่าลูกค้ารายนี้จัดอยู่ในกลุ่มเสี่ยง มีแนวโน้มที่จะยกเลิกบัตรเครดิต")
        if has_prob:
            st.progress(float(prob_churn), text=f"โอกาสยกเลิกบัตร: {prob_churn*100:.1f}%")
            
    elif str(prediction) == "0":
        st.success("### ✅ ลูกค้าปกติ (Loyal Customer)")
        st.write("ระบบประเมินว่าลูกค้ารายนี้มีแนวโน้มที่จะใช้งานบัตรต่อไปตามปกติ")
        if has_prob:
            st.progress(float(prob_stay), text=f"โอกาสใช้งานต่อ: {prob_stay*100:.1f}%")
            
    else:
        # กรณีโมเดลเป็นแบบจัดกลุ่ม (Segmentation) ไม่ใช่ 0 กับ 1
        st.info(f"### 🎯 ผลการจัดกลุ่ม: {prediction}")
        st.write("ระบบจัดลูกค้าท่านนี้อยู่ในกลุ่มตามที่แสดงด้านบน")

    with st.expander("📋 ดูข้อมูลที่ใช้คำนวณ"):
        st.dataframe(input_df, use_container_width=True)
