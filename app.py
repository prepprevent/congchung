import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
import io

# -------------------------------------------------------
# 8 cột cố định
# -------------------------------------------------------
COLS = [
    "Số DM Công chứng",
    "Ngày công chứng",
    "Ngày thụ lý",
    "Họ tên, nơi cư trú người yêu cầu công chứng",
    "Loại việc công chứng",
    "Tóm tắt nội dung",
    "Họ tên người ký công chứng",
    "Ghi chú"
]


# -------------------------------------------------------
# OCR 1 cell
# -------------------------------------------------------
def ocr_cell(img):
    text = pytesseract.image_to_string(img, lang="vie+eng", config="--psm 6")
    text = text.replace("\n", " ").strip()
    return text


# -------------------------------------------------------
# 1) Cắt CHỈ phần bảng trong ảnh (vì template cố định)
# -------------------------------------------------------
def crop_table_region(gray):
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 4
    )

    # tìm line ngang
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=2)

    contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ys = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 500:
            ys.append(y)

    ys = sorted(ys)

    # bảng bắt đầu dưới header → dòng 1
    table_top = ys[1]
    table_bottom = ys[-1]

    return gray[table_top:table_bottom, :]


# -------------------------------------------------------
# 2) tách grid (các ô)
# -------------------------------------------------------
def extract_cells(table_img):

    th = cv2.adaptiveThreshold(
        table_img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 4
    )

    # detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    v_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # combine to get grid intersections
    grid = cv2.add(v_lines, h_lines)

    # detect contours as cell borders
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 and h > 20:
            boxes.append((x, y, w, h))

    # sort top → bottom, left → right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # group into rows
    rows = []
    current_row = []
    last_y = None

    for b in boxes:
        x, y, w, h = b
        if last_y is None:
            current_row.append(b)
            last_y = y
            continue

        if abs(y - last_y) < 20:  # same row
            current_row.append(b)
        else:
            # finish row
            if len(current_row) >= 8:   # hàng hợp lệ
                rows.append(sorted(current_row, key=lambda x: x[0]))
            current_row = [b]
        last_y = y

    # add last row
    if len(current_row) >= 8:
        rows.append(sorted(current_row, key=lambda x: x[0]))

    return rows


# -------------------------------------------------------
# 3) OCR toàn bộ bảng
# -------------------------------------------------------
def ocr_table(gray):

    table_img = crop_table_region(gray)
    rows = extract_cells(table_img)

    results = []

    for r in rows:
        if len(r) < 8:
            continue

        row_data = []

        for (x, y, w, h) in r[:8]:   # đúng 8 cột
            cell = table_img[y:y+h, x:x+w]
            cell_rgb = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
            txt = ocr_cell(cell_rgb)
            row_data.append(txt)

        results.append(row_data)

    df = pd.DataFrame(results, columns=COLS)
    return df


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.title("Ứng dụng di động để xuất excel từ hình ảnh - Ms. Kim Hiền")
st.write("Hãy kiểm tra thử ngẫu nhiên 1 vài dòng nhé, vì chất lượng hình ảnh là rất quan trọng")

files = st.file_uploader("Tải lên ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if files:
    if st.button("Xử lý & Xuất Excel"):
        all_data = []

        for f in files:
            st.write(f"Đang xử lý: {f.name}")

            img = Image.open(f).convert("RGB")
            img = np.array(img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            df = ocr_table(gray)
            df["Nguồn ảnh"] = f.name
            all_data.append(df)

        final_df = pd.concat(all_data, ignore_index=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            final_df.to_excel(writer, index=False, sheet_name="data")
        buf.seek(0)

        st.download_button(
            "⬇ Tải Excel kết quả",
            data=buf,
            file_name="ket_qua_ocr_cong_chung.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
