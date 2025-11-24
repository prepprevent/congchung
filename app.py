# extract_template.py
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import os
import pandas as pd

# Template columns (the order will be used when exporting)
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

# --------- Helpers ----------
def load_image_bgr(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def preprocess_for_lines(img_bgr, scale_width=1400):
    h,w = img_bgr.shape[:2]
    if w != scale_width:
        scale = scale_width / w
        img_bgr = cv2.resize(img_bgr, (scale_width, int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # adaptive threshold -> binary
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 9)
    return img_bgr, thr

def detect_table_grid(thr):
    """
    Return vertical lines x positions and horizontal lines y positions (as sorted lists)
    """
    h, w = thr.shape[:2]

    # vertical kernel to detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//50)))
    vertical_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # horizontal kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//50), 1))
    horizontal_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # sum columns/rows to get strong line positions
    v_sum = np.sum(vertical_lines, axis=0)
    h_sum = np.sum(horizontal_lines, axis=1)

    # find peaks in v_sum -> x positions
    x_positions = []
    threshold_v = max(50, int(0.3 * np.max(v_sum)))  # heuristic threshold
    in_line = False
    acc = []
    for x, val in enumerate(v_sum):
        if val > threshold_v:
            acc.append(x)
            in_line = True
        else:
            if in_line:
                x_positions.append(int(np.mean(acc)))
                acc = []
            in_line = False
    if acc:
        x_positions.append(int(np.mean(acc)))

    # find peaks in h_sum -> y positions
    y_positions = []
    threshold_h = max(50, int(0.3 * np.max(h_sum)))
    in_line = False
    acc = []
    for y, val in enumerate(h_sum):
        if val > threshold_h:
            acc.append(y)
            in_line = True
        else:
            if in_line:
                y_positions.append(int(np.mean(acc)))
                acc = []
            in_line = False
    if acc:
        y_positions.append(int(np.mean(acc)))

    # Include image borders to ensure full coverage
    if 0 not in x_positions:
        x_positions = [0] + x_positions
    if w-1 not in x_positions:
        x_positions.append(w-1)
    if 0 not in y_positions:
        y_positions = [0] + y_positions
    if h-1 not in y_positions:
        y_positions.append(h-1)

    # sort & unique (in case)
    x_positions = sorted(list(dict.fromkeys(x_positions)))
    y_positions = sorted(list(dict.fromkeys(y_positions)))

    return x_positions, y_positions, vertical_lines, horizontal_lines

def crop_cell(img_bgr, x0, y0, x1, y1, pad=4):
    h,w = img_bgr.shape[:2]
    xa = max(0, x0 - pad)
    ya = max(0, y0 - pad)
    xb = min(w, x1 + pad)
    yb = min(h, y1 + pad)
    return img_bgr[ya:yb, xa:xb]

def ocr_image_bytes_from_array(img_array):
    # img_array is grayscale or binary or BGR; convert to PIL RGB for pytesseract
    if len(img_array.shape)==2:
        pil = Image.fromarray(img_array)
    else:
        pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    # optionally enhance - convert to RGB & use pytesseract
    txt = pytesseract.image_to_string(pil, lang='vie+eng', config='--psm 6')
    return txt.strip()

# --------- Main extractor ----------
def extract_table_template_from_path(image_path, debug=False):
    """
    Returns DataFrame with columns TEMPLATE_COLS and one row per table row found.
    If detection fails, it will try a fallback: OCR entire page and do heuristic splitting.
    """
    img_bgr = load_image_bgr(image_path)
    img_bgr, thr = preprocess_for_lines(img_bgr, scale_width=1400)
    x_pos, y_pos, v_lines, h_lines = detect_table_grid(thr)

    if debug:
        print("Detected x positions:", x_pos)
        print("Detected y positions:", y_pos)

    # Build column ranges from x positions: pairwise adjacent
    col_ranges = []
    for i in range(len(x_pos)-1):
        x0 = x_pos[i]
        x1 = x_pos[i+1]
        # ignore too small columns
        if x1 - x0 < 40:
            continue
        col_ranges.append((x0, x1))

    # Build row ranges from y positions: pairwise adjacent
    row_ranges = []
    for i in range(len(y_pos)-1):
        y0 = y_pos[i]
        y1 = y_pos[i+1]
        if y1 - y0 < 20:
            continue
        row_ranges.append((y0, y1))

    # Heuristic: header row often near top; we want actual data rows (skip header area if it seems like one)
    # Try to detect header by checking if first few rows have thicker horizontal lines (small heuristic)
    # For simplicity we will treat all row_ranges as table rows and later clean empty rows.
    records = []
    for r_idx, (y0, y1) in enumerate(row_ranges):
        # For each row, extract cells according to col_ranges
        row_cells = []
        for c_idx, (x0,x1) in enumerate(col_ranges):
            cell_img = crop_cell(img_bgr, x0, y0, x1, y1, pad=3)
            # Preprocess cell for OCR
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            # apply threshold to improve OCR
            thr_cell = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,8)
            # enlarge small texts
            thr_cell = cv2.resize(thr_cell, (max(200, thr_cell.shape[1]*2), max(50, thr_cell.shape[0]*2)), interpolation=cv2.INTER_LINEAR)
            txt = ocr_image_bytes_from_array(thr_cell)
            row_cells.append(txt)
        # If the row has any non-empty significant content, consider it a data row
        joined = " ".join([c.strip() for c in row_cells])
        if len(joined.strip()) < 3:
            continue
        # Map columns to the 8-template columns:
        # If detected columns count matches exactly 8 -> direct map
        # Otherwise: we do best-effort mapping: left columns map to first template cols etc., and merge remaining middle into "Tóm tắt nội dung" if needed.
        rec = {col: "" for col in TEMPLATE_COLS}
        n_detected_cols = len(row_cells)
        if n_detected_cols >= 8:
            # assign first 8
            for i, col_name in enumerate(TEMPLATE_COLS):
                rec[col_name] = row_cells[i].strip()
        else:
            # heuristic map:
            # assume typical layout in scanned pages: col0=STT or Số DM, col1=Ngày công chứng, col2=Ngày thụ lý, col3=Họ tên..., col4=Loại việc, col5=Tóm tắt nội dung, col6=Họ tên ký, col7=Ghi chú
            # We'll map available cols left->right and if fewer columns, we merge middle columns into Tóm tắt nội dung.
            # Example: if 6 detected cols -> map 0->0,1->1,2->2,3->3,4->4,5->5 ; leave last blank
            for i in range(n_detected_cols):
                if i < len(TEMPLATE_COLS):
                    rec[TEMPLATE_COLS[i]] = row_cells[i].strip()
                else:
                    # if more than template, append to tóm tắt nội dung
                    rec["Tóm tắt nội dung"] += (" " + row_cells[i].strip())
        # add source info if desired
        rec["_row_y_range"] = (y0, y1)
        records.append(rec)

    if not records:
        # fallback: OCR whole page and try to parse blocks by known labels (less accurate)
        pil = Image.open(image_path).convert("RGB")
        full_text = pytesseract.image_to_string(pil, lang='vie+eng', config='--psm 3')
        # simple fallback: one row with full_text in Tóm tắt nội dung
        rec = {col: "" for col in TEMPLATE_COLS}
        rec["Tóm tắt nội dung"] = full_text
        records.append(rec)

    df = pd.DataFrame(records)
    # Clean up helper column
    if "_row_y_range" in df.columns:
        df = df.drop(columns=["_row_y_range"])
    # Trim whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Reorder to template columns
    df = df[TEMPLATE_COLS]
    return df

# --------- Example usage ----------
if __name__ == "__main__":
    image_path = r"/mnt/data/4.jpg"   # <- path to your sample scanned page
    df = extract_table_template_from_path(image_path, debug=True)
    print(df.head())
    # Save to excel
    out = "ocr_template_output_from_4.xlsx"
    df.to_excel(out, index=False)
    print("Saved:", out)
