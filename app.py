import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import pandas as pd


# =============================
# 1. OCR utility
# =============================
def ocr_text(img):
    return pytesseract.image_to_string(img, lang="vie+eng", config="--psm 6")


# =============================
# 2. Detect header position
# =============================
def find_header_y(gray):
    """
    Tìm vị trí dòng chứa 'Số DM Công chứng'
    → giúp crop bảng nhanh và chính xác
    """
    h, w = gray.shape
    crop = gray[0:400, :]  # vùng trên
    text = pytesseract.image_to_string(crop, lang="vie+eng", config="--psm 6")

    if "Số DM" not in text and "Công chứng" not in text:
        return 120  # fallback

    d = pytesseract.image_to_data(crop, lang="vie+eng", config="--psm 6", output_type=pytesseract.Output.DICT)
    ys = []
    for i, word in enumerate(d["text"]):
        if "DM" in word or "Công" in word:
            ys.append(d["top"][i])

    if len(ys) == 0:
        return 120

    return min(ys)


# =============================
# 3. Crop table region
# =============================
def crop_table_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y0 = find_header_y(gray)

    # Trong ảnh bạn, bảng luôn kéo dài đến ~95% chiều cao
    h = img.shape[0]
    return img[y0:h - 50, 50:-50]  # crop lề trái-phải nhẹ


# =============================
# 4. Detect grid and extract cells
# =============================
def extract_table_cells(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(blur, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV,
                                15, 10)

    # detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # combine
    grid = cv2.add(v_lines, h_lines)

    # find contours (each cell)
    cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 20:  # loại noise
            boxes.append((x, y, w, h))

    # sort top→down rồi left→right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # group theo hàng
    rows = []
    current = []
    last_y = None

    for b in boxes:
        x, y, w, h = b
        if last_y is None:
            current.append(b)
            last_y = y
            continue

        if abs(y - last_y) < 20:
            current.append(b)
        else:
            if len(current) >= 8:
                rows.append(sorted(current, key=lambda b: b[0]))
            current = [b]
            last_y = y

    if len(current) >= 8:
        rows.append(sorted(current, key=lambda b: b[0]))

    return rows


# =============================
# 5. Extract text per cell
# =============================
def extract_row_data(rows, crop):
    data = []

    for r in rows:
        row_text = []
        for (x, y, w, h) in r[:8]:  # đúng 8 cột
            cell = crop[y:y + h, x:x + w]
            txt = pytesseract.image_to_string(cell, lang="vie+eng",
                                              config="--psm 6").strip()
            row_text.append(txt)
        data.append(row_text)

    return data


# =============================
# STREAMLIT APP
# =============================
st.title("FAST OCR Công chứng — Chỉ lấy dữ liệu trong bảng")

files = st.file_uploader("Chọn ảnh bảng công chứng", type=["jpg", "jpeg", "png"],
                         accept_multiple_files=True)

if files:
    if st.button("Xử lý tất cả ảnh và xuất Excel"):
        ALL = []

        for f in files:
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # crop bảng
            crop = crop_table_region(img)

            # detect grid
            rows = extract_table_cells(crop)

            # OCR từng cell
            table = extract_row_data(rows, crop)

            for r in table:
                ALL.append(r)

        df = pd.DataFrame(ALL, columns=[
            "Số DM Công chứng",
            "Ngày công chứng",
            "Ngày thụ lý",
            "Họ tên người yêu cầu công chứng",
            "Loại việc công chứng",
            "Tóm tắt nội dung",
            "Họ tên người ký",
            "Ghi chú"
        ])

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False)

        st.download_button("Tải Excel", buf.getvalue(),
                           file_name="ket_qua_ocr.xlsx")
