# app.py
import streamlit as st
from PIL import Image
import io
import os
import shutil
import requests
import pandas as pd
import pytesseract

# Optional: kiểm tra xem tesseract có trên PATH không
def tesseract_available():
    return shutil.which('tesseract') is not None

# OCR bằng pytesseract (local tesseract)
def ocr_with_tesseract(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    text = pytesseract.image_to_string(img, lang='vie+eng', config='--psm 6')
    return text

# OCR fallback: OCR.space API
def ocr_with_ocr_space(img_bytes, api_key=None, language='vie'):
    """
    Uses OCR.space free API. It's simple, but has rate limits.
    Provide API key via Streamlit secrets or environment variable.
    """
    if api_key is None:
        # Try streamlit secrets first (on Streamlit Cloud)
        try:
            api_key = st.secrets["OCR_SPACE_API_KEY"]
        except Exception:
            api_key = os.environ.get("OCR_SPACE_API_KEY")

    if not api_key:
        raise ValueError("No OCR.space API key provided. Set OCR_SPACE_API_KEY in Streamlit secrets or env.")

    files = {'file': ('image.jpg', img_bytes)}
    data = {
        'apikey': api_key,
        'language': language,
        'isOverlayRequired': False,
        'OCREngine': 2
    }
    r = requests.post('https://api.ocr.space/parse/image', files=files, data=data, timeout=60)
    r.raise_for_status()
    result = r.json()
    # Parse the response
    parsed = []
    if 'ParsedResults' in result and result['ParsedResults']:
        for pr in result['ParsedResults']:
            parsed.append(pr.get('ParsedText', ''))
    return "\n\n".join(parsed)

# Wrapper: tự chọn phương pháp
def ocr_image(img_bytes):
    # prefer local tesseract if available
    if tesseract_available():
        try:
            return ocr_with_tesseract(img_bytes)
        except Exception as e:
            st.warning(f"Local tesseract failed: {e}. Trying OCR.space fallback.")
    # fallback to OCR.space
    try:
        return ocr_with_ocr_space(img_bytes)
    except Exception as e:
        # trả lỗi rõ ràng cho người deployer, nhưng không crash app
        st.error("Không thể OCR: cả Tesseract trên server lẫn OCR.space fallback đều không thành công.")
        st.write("Lỗi chi tiết:", e)
        return ""

# Streamlit UI
st.title("OCR bảng / công chứng → Xuất Excel (10 ảnh/lần)")

uploaded_files = st.file_uploader("Chọn tối đa 10 ảnh (jpg/png)", accept_multiple_files=True, type=['jpg','jpeg','png'])

if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("Chỉ upload tối đa 10 ảnh mỗi lần.")
    else:
        st.info(f"Bạn đã chọn {len(uploaded_files)} ảnh.")
        if st.button("📄 OCR & Tạo Excel"):
            rows = []
            for f in uploaded_files:
                st.write("Xử lý:", f.name)
                img_bytes = f.read()
                text = ocr_image(img_bytes)
                rows.append({'filename': f.name, 'ocr_text': text})
            df = pd.DataFrame(rows)
            # convert to excel bytes
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='ocr')
            towrite.seek(0)
            st.download_button("⬇ Tải file Excel", data=towrite, file_name="ocr_output.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.write("Chưa có ảnh nào được tải lên. Bạn có thể dùng 2 ảnh mẫu trong workspace để thử: `/mnt/data/1.jpg` và `/mnt/data/2.jpg`")
