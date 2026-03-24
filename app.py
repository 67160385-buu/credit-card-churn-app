import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===== การตั้งค่าหน้าเว็บแบบ Wide =====
st.set_page_config(
    page_title="Smart Credit Card Analytics",
    page_icon="💳",
    layout="wide" # ปรับเป็นจอ กว้างเพื่อให้ดูเป็น Dashboard
)

# Custom CSS เพื่อให้ปุ่มและตัวอักษรสวยขึ้น
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl") 
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

model, scaler, model_columns = load_model()

# ===== Sidebar Design =====
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.title("Customer Insights")
    st.info("กรอกข้อมูลลูกค้าในส่วนหลักเพื่อเริ่มการวิเคราะห์")
    st.divider()
    st.write("📌 **Model Version:** 1.2.0 (Stable)")
    st.write("📊 **Algorithm:** Gradient Boosting")

# ===== Main UI =====
st.title("💳 ระบบวิเคราะห์ความเสี่ยงลูกค้าบัตรเครดิต")
st.write("---")

# แบ่งส่วนกรอกข้อมูลเป็น 2 ฝั่ง
tab1, tab2 = st.tabs(["📝 กรอกข้อมูลลูกค้า", "📊 ผลการวิเคราะห์"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 ข้อมูลพื้นฐาน")
        customer_age = st.slider("อายุลูกค้า", 18, 80, 35)
        gender = st.selectbox("เพศ", ["M", "F"])
        income = st.selectbox("รายได้ต่อปี", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"])
        card = st.selectbox("ประเภทหน้าบัตร", ["Blue", "Silver", "Gold", "Platinum"])
    
    with col2:
        st.subheader("💰 พฤติกรรมการใช้จ่าย")
        months = st.number_input("เป็นลูกค้ามาแล้ว (เดือน)", 0, 120, 36)
        trans_amt = st.number_input("ยอดรูดรวม 12 เดือน ($)", 0, 50000, 4500)
        trans_ct = st.number_input("จำนวนครั้งที่รูดบัตร (ครั้ง)", 0, 500, 65)
        rev_bal = st.number_input("ยอดหนี้ค้างชำระ ($)", 0, 10000, 1500)

    predict_button = st.button("🚀 เริ่มการวิเคราะห์เดี๋ยวนี้")

if predict_button:
    with tab2:
        # Prepare Data
        input_df = pd.DataFrame({
            'Customer_Age': [customer_age], 'Gender': [gender],
            'Income_Category': [income], 'Card_Category': [card],
            'Months_on_book': [months], 'Total_Trans_Amt': [trans_amt],
            'Total_Trans_Ct': [trans_ct], 'Total_Revolving_Bal': [rev_bal]
        })
        
        input_encoded = pd.get_dummies(input_df)
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_aligned)
        
        prediction = model.predict(input_scaled)[0]
        
        # แสดง Metrics สำคัญก่อน
        m1, m2, m3 = st.columns(3)
        m1.metric("ยอดใช้จ่ายรวม", f"${trans_amt:,.0f}")
        m2.metric("ความถี่การใช้งาน", f"{trans_ct} ครั้ง/ปี")
        m3.metric("หนี้คงค้าง", f"${rev_bal:,.0f}", delta_color="inverse")

        st.divider()

        # แสดงผลลัพธ์แบบเน้นสี
        if str(prediction) == "1":
            st.error("## ⚠️ผลลัพธ์: มีความเสี่ยงที่จะยกเลิกบัตร (Churn Risk)")
            st.markdown("""
                **💡 คำแนะนำสำหรับฝ่ายการตลาด:**
                * ควรโทรติดต่อเพื่อสอบถามความพึงพอใจ
                * เสนอโปรโมชั่นลดค่าธรรมเนียมรายปี
                * เพิ่มสิทธิประโยชน์ในการแลกแต้ม
            """)
        else:
            st.success("## ✅ผลลัพธ์: ลูกค้ากลุ่มภักดี (Loyal Customer)")
            st.markdown("""
                **💡 คำแนะนำสำหรับฝ่ายการตลาด:**
                * เสนอการอัปเกรดหน้าบัตร (เช่น จาก Blue เป็น Gold)
                * ส่งคำเชิญเข้าร่วมกิจกรรมพิเศษสำหรับ Exclusive Member
                * มอบของขวัญพิเศษในเดือนเกิด
            """)
        
        # เพิ่มชาร์ตเล็กๆ ให้ดูว้าว
        st.write("🔍 **เปรียบเทียบยอดใช้จ่ายกับค่าเฉลี่ย**")
        chart_data = pd.DataFrame({
            'Category': ['ลูกค้าคนนี้', 'ค่าเฉลี่ยลูกค้าทั่วไป'],
            'ยอดใช้จ่าย ($)': [trans_amt, 4400]
        })
        st.bar_chart(chart_data.set_index('Category'))

else:
    with tab2:
        st.info("กรุณากดปุ่มเริ่มการวิเคราะห์")
