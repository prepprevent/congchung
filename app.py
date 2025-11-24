import streamlit as st
from PIL import Image
import io, re, shutil, os
import pandas as pd
import pytesseract

# ---------------------------------------------------------
# 8 CỘT CHUẨN THEO YÊU CẦU
# ---------------------------------------------------------
TEMPLATE_COLS = [
    "Số DM Công chứng",
    "Ngày công chứng",
    "Ngày thụ lý",
    "Họ tên, nơi cư trú người yêu cầu công chứng",
    "Loại việc công chứng",
    "Tóm tắt nội dung",
    "Họ tên người ký công chứng",
    "Ghi chú"
]

# ---------------------------------------------------------
#  REGEX trích theo pattern đúng với trang bạn upload
# ---------------------------------------------------------
PATTERNS = {
    "Số DM Công chứng": r"(?i)(Số DM Công chứng|Số DM|Số)\s*[:\-]?\s*([0-9]{4,})",
    "Ngày công chứng": r"(?i)Ngày công chứng\s*[:\-]?\s*([0-9]{1,2}\/[0-9]{1,2}\/[0-9]{4})",
    "Ngày thụ lý": r"(?i)Ngày thụ lý\s*[:\-]?\s*([0-9]{1,2}\/[0-9]{1,2}\/[0-9]{4})",
    "Họ tên, nơi cư trú người yêu cầu công chứng":
        r"(?i)(Bên yêu cầu|Bên A|Họ tên.*?yêu cầu)\s*[:\-]?\s*(.+?)(?=(Loại việc|Tóm tắt|Họ tên người ký|$))",
    "Loại việc công chứng":
        r"(?i)(Loại việc|Loại việc công chứng|Nhóm)\s*[:\-]?\s*(.+?)(?=(Tóm tắt|Họ tên người ký|$))",
    "Tóm tắt nội dung":
        r"(?i)(Tóm tắt nội dung|Nội dung)\s*[:\-]?\s*(.+?)(?=(Họ tên người ký|$))",
    "Họ tên người ký công chứng":
        r"(?i)(Họ tên người ký|Công chứng viên)\s*[:\-]?\s*(.+?)(?=(Ghi chú|$))",
    "Ghi chú": r"(?i)(Ghi chú)\s*[:\-]?\s*(.+)$"
}


def tesseract_available():
    return shutil.which("tesseract") is not None


def ocr(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return pytesseract.image_to_string(img, lang="vie+eng", config="--psm 6")


# ---------------------------------------------------------
# HÀM PARSE OCR → 8 CỘT
# ---------------------------------------------------------
def parse_text(full_text):
    data = {c: "" for c in TEMPLATE_COLS}

    for col, patt in PATTERNS.items():
        m = re.search(patt, full_text, re.DOTALL)
        if m:
            data[col] = m.group(len(m.groups()))  # lấy group cuối
            data[col] = re.sub(r"\s+", " ", data[col]).strip()

    return data


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("OCR Công chứng → Xuất Excel theo 8 cột chuẩn")

uploaded_files = st.file_uploader("Tải lên ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Tạo file Excel"):
        records = []

        for f in uploaded_files:
            st.write(f"Đang xử lý: {f.name}")
            img_bytes = f.read()

            text = ocr(img_bytes)
            row = parse_text(text)

            row["Tên file"] = f.name  # thêm để kiểm tra nguồn
            records.append(row)

        df = pd.DataFrame(records)

        # đảm bảo đúng thứ tự cột
        df = df[TEMPLATE_COLS + ["Tên file"]]

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        buf.seek(0)

        st.download_button(
            "⬇ Tải Excel kết quả",
            data=buf,
            file_name="hinh_anh_cong_chung.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
