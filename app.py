import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Smart Credit Card Analytics",
    page_icon="💳",
    layout="wide" 
)

# 2. Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
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

# 3. Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.title("Customer Insights")
    st.info("กรอกข้อมูลลูกค้าในส่วนหลักเพื่อเริ่มการวิเคราะห์")
    st.divider()
    st.write("📌**Model Version:** 1.2.0")
    st.write("📊**Algorithm:** Gradient Boosting")

# 4. Main UI
st.title("💳 ระบบวิเคราะห์ความเสี่ยงลูกค้าบัตรเครดิต")
st.write("---")

tab1, tab2 = st.tabs(["📝กรอกข้อมูลลูกค้า", "📊ผลการวิเคราะห์"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤ข้อมูลพื้นฐาน")
        customer_age = st.slider("อายุลูกค้า", 18, 80, 35)
        gender = st.selectbox("เพศ", ["M", "F"])
        # Map รายได้ไทย กลับเป็นค่าที่โมเดลรู้จัก (English)
        income_display = st.selectbox("รายได้ต่อปี (โดยประมาณ)", [
            "น้อยกว่า 1.4 ล้านบาท", "1.4 - 2.1 ล้านบาท", 
            "2.1 - 2.8 ล้านบาท", "2.8 - 4.2 ล้านบาท", "4.2 ล้านบาทขึ้นไป"
        ])
        income_map = {
            "น้อยกว่า 1.4 ล้านบาท": "Less than $40K",
            "1.4 - 2.1 ล้านบาท": "$40K - $60K",
            "2.1 - 2.8 ล้านบาท": "$60K - $80K",
            "2.8 - 4.2 ล้านบาท": "$80K - $120K",
            "4.2 ล้านบาทขึ้นไป": "$120K +"
        }
        income = income_map[income_display]
        card = st.selectbox("ประเภทหน้าบัตร", ["Blue", "Silver", "Gold", "Platinum"])
    
    with col2:
        st.subheader("💰พฤติกรรมการใช้จ่าย")
        months = st.number_input("เป็นลูกค้ามาแล้ว (เดือน)", 0, 120, 36)
        trans_amt_thb = st.number_input("ยอดรูดรวม 12 เดือน (บาท)", 0, 2000000, 150000, step=10000)
        trans_amt = trans_amt_thb / 35 
        trans_ct = st.number_input("จำนวนครั้งที่รูดบัตร (ครั้ง)", 0, 500, 65)
        rev_bal_thb = st.number_input("ยอดหนี้ค้างชำระ (บาท)", 0, 500000, 50000, step=5000)
        rev_bal = rev_bal_thb / 35

    predict_button = st.button("เริ่มการวิเคราะห์เดี๋ยวนี้")

# 5. การประมวลผล
if predict_button:
    with tab2:
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
        
        m1, m2, m3 = st.columns(3)
        m1.metric("ยอดใช้จ่ายรวม", f"{trans_amt_thb:,.0f} ฿")
        m2.metric("ความถี่การใช้งาน", f"{trans_ct} ครั้ง/ปี")
        m3.metric("หนี้คงค้าง", f"{rev_bal_thb:,.0f} ฿", delta_color="inverse")

        st.divider()

        if str(prediction) == "1":
            st.error("⚠️ผลลัพธ์:มีความเสี่ยงที่จะยกเลิกบัตร (Churn Risk)")
            st.markdown("*💡คำแนะนำ:ควรโทรติดต่อเพื่อเสนอโปรโมชั่นพิเศษ")
        else:
            st.success("✅ผลลัพธ์: ลูกค้ากลุ่มภักดี (Loyal Customer)")
            st.markdown("*💡คำแนะนำ:เสนอการอัปเกรดหน้าบัตรหรือสิทธิประโยชน์เพิ่ม")
        
        st.write("---")
        st.write("🔍 **เปรียบเทียบยอดใช้จ่ายกับค่าเฉลี่ย**")

        chart_data = pd.DataFrame({
            'กลุ่มลูกค้า': ['ลูกค้าคนนี้', 'ค่าเฉลี่ยลูกค้าทั่วไป'],
            'ยอดใช้จ่าย (฿)': [trans_amt_thb, 4400 * 35]
        })

        fig = px.bar(
            chart_data, x='กลุ่มลูกค้า', y='ยอดใช้จ่าย (฿)',
            color='กลุ่มลูกค้า',
            color_discrete_map={'ลูกค้าคนนี้': '#007bff', 'ค่าเฉลี่ยลูกค้าทั่วไป': '#ced4da'},
            text_auto='.2s'
        )
        fig.update_layout(showlegend=False, height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

else:
    with tab2:
        st.info("👈กรุณากดปุ่มเริ่มการวิเคราะห์เพื่อดูผลลัพธ์")
