import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. ตั้งค่าหน้าเว็บให้ดูเรียบหรู ---
st.set_page_config(
    page_title="Credit Card Churn Predictor | อุ่นใจด้านการเงิน", 
    page_icon="💳", 
    layout="centered"
)

# --- 2. สไตล์ CSS แบบเรียบหรู ---
st.markdown("""
<style>
    /* ตั้งค่าฟอนต์หลักและสีพื้นหลัง */
    .stApp {
        background-color: #f8f9fb; /* สีขาวอมฟ้าอ่อนๆ ดูสะอาดตา */
        color: #333333;
        font-family: 'Inter', sans-serif; /* ใช้ฟอนต์สไตล์โมเดิร์น */
    }

    /* ปรับแต่งหัวข้อ Title */
    .stTitle {
        font-family: 'Poppins', sans-serif;
        color: #1a73e8; /* สีน้ำเงินของ Google/ธนาคาร ดูน่าเชื่อถือ */
        text-align: center;
    }

    /* ปรับแต่งปุ่มทำนายผล */
    .stButton>button {
        background-color: #1a73e8; 
        color: white; 
        border-radius: 20px; /* ขอบมน น่ารักขึ้น */
        padding: 10px 20px; 
        width: 100%; /* ปุ่มเต็มความกว้าง */
        border: none;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* เพิ่มเงาให้ดูมีมิติ */
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1557b0; /* สีเข้มขึ้นตอนเอาเมาส์ชี้ */
        border: none;
    }

    /* ปรับแต่งช่องกรอกข้อมูลให้ดูสะอาดตา */
    .stNumberInput input, .stSelectbox select {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    
    /* ปรับแต่งข้อความ Subheader */
    .stSubheader {
        color: #5f6368;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล AI ---
model_path = os.path.join("model_artifacts", "credit_card_churn_model.pkl")

try:
    model = joblib.load(model_path)
except:
    st.error("❌ ไม่พบไฟล์โมเดล! กรุณาเช็คว่ามีไฟล์ในโฟลเดอร์ model_artifacts หรือยัง")
    st.stop() # หยุดการทำงานหากหาโมเดลไม่เจอ

# --- 4. ส่วนหัวของหน้าเว็บ ---
st.image("https://img.icons8.com/clouds/200/credit-card.png", width=120) # เพิ่มไอคอนน่ารักๆ
st.title("💖 ระบบดูแลลูกค้าน่ารักของเรา 🛡️")
st.subheader("วิเคราะห์พฤติกรรมการใช้งานบัตรเครดิตล่วงหน้า")
st.write("สวัสดีครับ! กรุณากรอกข้อมูลลูกค้าด้านล่าง เพื่อช่วยกันดูแลลูกค้าคนสำคัญครับ")
st.divider()

# --- 5. สร้างฟอร์มรับข้อมูลลูกค้า ---
st.write("#### 👨‍💼 ข้อมูลพื้นฐานลูกค้า")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("อายุ (Age) 🎂", min_value=18, max_value=100, value=40)
    gender = st.selectbox("เพศ (Gender) 🚻", ["ชาย", "หญิง"])
    dependent_count = st.number_input("จำนวนผู้รอการดูแล (Dependents) 👨‍👩‍👧", min_value=0, max_value=10, value=2)

with col2:
    months_on_book = st.number_input("ระยะเวลาที่เป็นลูกค้า (เดือน) 🗓️", min_value=1, value=36)
    total_relationship_count = st.number_input("จำนวนผลิตภัณฑ์ธนาคารที่ใช้ (Products) 🏦", min_value=1, max_value=10, value=3)

st.divider()
st.write("#### 💰 ข้อมูลการเงินล่าสุด")
col3, col4 = st.columns(2)

with col3:
    education = st.selectbox("ระดับการศึกษา (Education) 🎓", ["High School", "Graduate", "Uneducated", "College", "Post-Graduate", "Doctorate", "Unknown"])
    credit_limit = st.number_input("วงเงินบัตรเครดิต (Credit Limit) 💳", min_value=500.0, value=5000.0)

with col4:
    total_trans_ct = st.number_input("จำนวนครั้งที่ทำรายการ (ปีล่าสุด) 🛍️", min_value=1, value=50)

# --- 6. ปุ่มทำนายผลแบบเรียบหรู ---
st.write("###") # เพิ่มช่องว่างนิดนึง
if st.button("📊 เริ่มวิเคราะห์ข้อมูลกันเลย!"):
    # เตรียมข้อมูลให้โมเดล
    input_data = pd.DataFrame({
        'Customer_Age': [age],
        'Gender': [1 if gender == "ชาย" else 0],
        'Dependent_count': [dependent_count],
        'Education_Level': [education],
        'Months_on_book': [months_on_book],
        'Total_Relationship_Count': [total_relationship_count],
        'Credit_Limit': [credit_limit],
        'Total_Trans_Ct': [total_trans_ct]
    })
    
    # ทำนายผล
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    
    # --- 7. แสดงผลลัพธ์แบบน่ารักและชัดเจน ---
    if prediction[0] == 1:
        st.error(f"⚠️ **แจ้งเตือนความเสี่ยง!** ลูกค้ามีแนวโน้มจะยกเลิกบัตร 😢")
        st.write(f"โอกาสที่ AI มั่นใจ: **{probability:.2%}**")
        st.write("💡 *คำแนะนำ:* ลองเสนอโปรโมชั่นพิเศษ หรือติดต่อสอบถามเพื่อดูแลเป็นพิเศษนะครับ")
    else:
        st.success(f"✅ **สบายใจได้!** ลูกค้ามีแนวโน้มใช้งานต่อครับ 🥰")
        st.write(f"โอกาสที่ AI มั่นใจ: **{(1-probability):.2%}**")
        st.write("💡 *คำแนะนำ:* ส่งเสริมการใช้งานอย่างต่อเนื่อง ด้วยสิทธิประโยชน์ที่หลากหลายครับ")

# --- ส่วนท้าย ---
st.divider()
st.caption("จัดทำโดย: [ใส่ชื่อของคุณ] ✨")
st.caption("อ้างอิง: ข้อมูลพฤติกรรมการยกเลิกบัตรเครดิต")