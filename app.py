import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. โหลดไฟล์ที่บันทึกไว้ (Model, Scaler, Columns)
# ==========================================
# ตรวจสอบให้แน่ใจว่าไฟล์ทั้ง 3 อยู่ในโฟลเดอร์เดียวกับ app.py นะครับ
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

# ==========================================
# 2. ส่วนหน้าตาเว็บ (UI)
# ==========================================
st.title("📊 Profit Prediction App")
st.write("แอปพลิเคชันสำหรับทำนายว่าออเดอร์นี้จะได้กำไรหรือขาดทุน")
st.markdown("---")

# สร้างฟอร์มรับข้อมูล (คุณสามารถเพิ่ม/ลดช่องกรอกให้ตรงกับข้อมูลจริงของคุณได้เลย)
col1, col2 = st.columns(2)

with col1:
    sales = st.number_input("ยอดขาย (Sales)", min_value=0.0, value=100.0)
    quantity = st.number_input("จำนวน (Quantity)", min_value=1, value=1)
    
with col2:
    discount = st.number_input("ส่วนลด (Discount)", min_value=0.0, max_value=1.0, value=0.0)
    # สมมติว่ามีคอลัมน์ Category ให้เลือก (คุณต้องแก้ชื่อใน List ให้ตรงกับข้อมูลของคุณนะ)
    category = st.selectbox("หมวดหมู่สินค้า (Category)", ["Furniture", "Office Supplies", "Technology"])

st.markdown("---")

# ==========================================
# 3. เมื่อกดปุ่มทำนาย
# ==========================================
if st.button("🔮 ทำนายผล"):
    
    # 3.1 นำข้อมูลที่ผู้ใช้กรอกมาใส่เป็น DataFrame 1 แถว
    # ชื่อ Key ด้านซ้ายมือ ต้องพิมพ์ให้ตรงกับชื่อคอลัมน์ใน Dataset เป๊ะๆ (ตัวพิมพ์เล็ก/ใหญ่)
    input_df = pd.DataFrame({
        'Sales': [sales],
        'Quantity': [quantity],
        'Discount': [discount],
        'Category': [category]
        # ⚠️ ถ้าตอน Train คุณมีคอลัมน์อื่นอีก (เช่น Shipping_Cost, Sub-Category) 
        # ต้องเพิ่มช่องรับข้อมูลด้านบน และเอามาใส่ใน Dictionary นี้ให้ครบนะครับ
    })
    
    # 3.2 แปลงข้อมูลตัวหนังสือเป็นตัวเลข (One-Hot Encoding)
    input_df = pd.get_dummies(input_df)
    
    # 3.3 🔥 จุดสำคัญที่สุดที่แก้ Error! 🔥
    # บังคับโครงสร้างคอลัมน์ให้ตรงกับตอน Train เป๊ะๆ ถ้าคอลัมน์ไหนขาดหายไปให้เติม 0
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # 3.4 สเกลข้อมูลให้เป็นมาตรฐานเดียวกับตอน Train
    input_scaled = scaler.transform(input_df)
    
    # 3.5 ให้โมเดลทำนาย
    prediction = model.predict(input_scaled)
    
    # 3.6 แสดงผลลัพธ์
    if prediction[0] == 1:
        st.success("✅ จากข้อมูลนี้ คาดว่าออเดอร์นี้จะมี **กำไร (Profit)**")
    else:
        st.error("❌ จากข้อมูลนี้ คาดว่าออเดอร์นี้จะ **ขาดทุน (Loss)**")