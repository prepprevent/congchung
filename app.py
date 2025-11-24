import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import io
import base64

st.title("🔎 OCR Công chứng → Xuất Excel (10 ảnh/lần)")

uploaded_files = st.file_uploader(
    "Tải lên tối đa 10 ảnh",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

def ocr_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(img, lang='vie')
    return text

def convert_df(df):
    return df.to_excel(index=False, engine='openpyxl')

if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("❌ Chỉ được import tối đa 10 ảnh mỗi lần!")
    else:
        st.success(f"✔ Bạn đã tải {len(uploaded_files)} ảnh")

    if st.button("📄 Create Excel"):
        all_rows = []

        for file in uploaded_files:
            text = ocr_image(file.read())
            all_rows.append({
                "filename": file.name,
                "content": text
            })

        df = pd.DataFrame(all_rows)
        excel_bytes = convert_df(df)

        st.download_button(
            label="⬇ Tải Excel",
            data=excel_bytes,
            file_name="ocr_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
