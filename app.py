# ในไฟล์ app.py ส่วนที่รับข้อมูล
revenue = st.number_input("Revenue", value=100.0)
quantity = st.number_input("Quantity", value=1)
discount = st.number_input("Discount", value=0.0)
shipping_cost = st.number_input("Shipping Cost", value=10.0)
category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])

# ตอนสร้าง DataFrame เพื่อทำนาย
input_df = pd.DataFrame({
    'Revenue': [revenue],
    'Quantity': [quantity],
    'Discount': [discount],
    'Shipping_Cost': [shipping_cost],
    'Category': [category]
})