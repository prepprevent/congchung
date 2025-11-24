import streamlit as st
from PIL import Image
import io, os, shutil, re
import numpy as np
import cv2
import pytesseract
import pandas as pd

# ---------------------- Cấu hình ----------------------
# 8 cột theo yêu cầu
COLUMNS = [
    "Số DM Công chứng",
    "Ngày công chứng",
    "Ngày thụ lý",
    "Họ tên, nơi cư trú người yêu cầu công chứng",
    "Loại việc công chứng",
    "Tóm tắt nội dung",
    "Họ tên người ký công chứng",
    "Ghi chú"
]

# sample path (developer đã upload file)
SAMPLE_PATH = "/mnt/data/3.jpg"

# ---------------------- Helpers ----------------------
def tesseract_available():
    return shutil.which("tesseract") is not None

def ocr_image(pil_img):
    # pil_img: PIL Image (RGB or L)
    config = "--psm 6"
    try:
        txt = pytesseract.image_to_string(pil_img, lang="vie+eng", config=config)
    except Exception:
        txt = pytesseract.image_to_string(pil_img, lang="eng", config=config)
    # normalize whitespace
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def sort_coords(vals, tol=15):
    """Merge close coordinates and return sorted unique ints."""
    vals = sorted([int(v) for v in vals])
    if not vals:
        return []
    merged = [vals[0]]
    for v in vals[1:]:
        if v - merged[-1] <= tol:
            # merge by averaging
            merged[-1] = int((merged[-1] + v) / 2)
        else:
            merged.append(v)
    return merged

# ---------------------- Grid detection ----------------------
def detect_table_grid(img_bgr, debug=False):
    """
    Return xs (vertical grid x positions) and ys (horizontal grid y positions)
    or (None, None) if fail
    """
    # Convert to gray and adaptive threshold
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # increase contrast / denoise a bit
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thr = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,15,9)

    h, w = thr.shape

    # Create structure elements for lines detection
    horiz_size = max(10, w // 40)
    vert_size = max(10, h // 40)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))

    horizontal = cv2.erode(thr, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=2)

    vertical = cv2.erode(thr, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=2)

    # intersections
    mask = cv2.bitwise_and(horizontal, vertical)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    pts = []
    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        cx = x + wc//2
        cy = y + hc//2
        pts.append((cx,cy))

    if not pts:
        return None, None

    xs = sorted(set([p[0] for p in pts]))
    ys = sorted(set([p[1] for p in pts]))

    xs = sort_coords(xs, tol=max(8, w//200))
    ys = sort_coords(ys, tol=max(8, h//200))

    # Very important: ensure leftmost and rightmost borders included
    # Add image border coordinates if not present
    if xs[0] > 5:
        xs.insert(0, 0)
    if xs[-1] < w-5:
        xs.append(w)
    if ys[0] > 5:
        ys.insert(0, 0)
    if ys[-1] < h-5:
        ys.append(h)

    # debug option: show grid overlay
    if debug:
        debug_img = img_bgr.copy()
        for x in xs:
            cv2.line(debug_img, (x,0), (x,h-1), (0,255,0), 1)
        for y in ys:
            cv2.line(debug_img, (0,y), (w-1,y), (255,0,0), 1)
        return xs, ys, debug_img

    return xs, ys

# ---------------------- Build cells & OCR ----------------------
def extract_cells_from_grid(img_bgr, xs, ys, expected_cols=8):
    """
    Given xs and ys (grid lines), build cell boxes between adjacent coords
    and OCR each cell. Return list of rows; each row is list of cell texts.
    If number of columns not equal expected_cols, try to merge/split to get expected_cols.
    """
    h, w = img_bgr.shape[:2]
    # cell boxes from adjacent pairs
    rows = []
    for i in range(len(ys)-1):
        y0 = ys[i]
        y1 = ys[i+1]
        # ignore tiny rows
        if y1 - y0 < 10:
            continue
        row_cells = []
        for j in range(len(xs)-1):
            x0 = xs[j]; x1 = xs[j+1]
            if x1 - x0 < 8:
                continue
            # crop with small margin
            mx = max(0, x0+2)
            Mx = min(w, x1-2)
            my = max(0, y0+2)
            My = min(h, y1-2)
            if Mx - mx <= 3 or My - my <= 3:
                row_cells.append("")
                continue
            crop = img_bgr[my:My, mx:Mx]
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            txt = ocr_image(pil)
            row_cells.append(txt)
        # skip empty rows
        if all(len(s.strip())==0 for s in row_cells):
            continue
        rows.append(row_cells)

    # Normalize columns: many images grid detection returns number of cols might be >8
    # We'll merge adjacent narrow columns to reach expected_cols if needed.
    if not rows:
        return []

    ncols = max(len(r) for r in rows)
    if ncols == expected_cols:
        # good
        # ensure every row has exactly expected_cols by padding
        final_rows = []
        for r in rows:
            r2 = r + [""]*(expected_cols - len(r))
            final_rows.append(r2[:expected_cols])
        return final_rows

    # If ncols > expected_cols: merge adjacent columns by minimal text length heuristic
    if ncols > expected_cols:
        # compute column widths from xs
        widths = []
        for j in range(ncols):
            # approx width using consecutive xs differences
            try:
                wj = xs[j+1] - xs[j]
            except:
                wj = 10
            widths.append(wj)
        # merge iteratively smallest width with neighbor until len == expected_cols
        col_indices = [[i] for i in range(ncols)]
        while len(col_indices) > expected_cols:
            # find index with smallest total width
            col_widths = [sum(widths[idx] for idx in group) for group in col_indices]
            idx_min = int(np.argmin(col_widths))
            # decide merge with left or right neighbor (prefer right if exists)
            if idx_min < len(col_indices)-1:
                # merge with right
                col_indices[idx_min] = col_indices[idx_min] + col_indices[idx_min+1]
                del col_indices[idx_min+1]
            else:
                # merge with left
                col_indices[idx_min-1] = col_indices[idx_min-1] + col_indices[idx_min]
                del col_indices[idx_min]
        # now build final rows by joining texts of merged columns with separator " | "
        final_rows = []
        for r in rows:
            newr = []
            for grp in col_indices:
                parts = []
                for k in grp:
                    if k < len(r):
                        parts.append(r[k])
                newr.append(" | ".join([p for p in parts if p]))
            # pad
            newr = newr + [""]*(expected_cols - len(newr))
            final_rows.append(newr[:expected_cols])
        return final_rows

    # If ncols < expected_cols: try to split big columns heuristically by newline markers
    if ncols < expected_cols:
        final_rows = []
        for r in rows:
            # try to split central big column (likely big text areas)
            newr = r.copy()
            # if last cell contains many separators and can be split, attempt greedy split
            # fallback: pad to expected_cols
            if len(newr) < expected_cols:
                newr = newr + [""]*(expected_cols - len(newr))
            final_rows.append(newr[:expected_cols])
        return final_rows

    return []

# ---------------------- Map extracted row (list of 8 texts) -> required columns ----------------------
def map_row_to_schema(row_cells):
    """
    row_cells: list length 8 (texts)
    We'll map columns by position according to typical layout in your image:
    [col0, col1, col2, col3, col4, col5, col6, col7] -> map to COLUMNS
    This mapping may be adjusted if needed.
    """
    # default simple mapping: assume the scanned table column order is:
    # Số DM | Ngày công chứng | Ngày thụ lý | Họ tên/địa chỉ yêu cầu | Loại việc | Tóm tắt nội dung | Họ tên ký | Ghi chú
    mapped = {}
    for i, name in enumerate(COLUMNS):
        if i < len(row_cells):
            mapped[name] = row_cells[i].strip()
        else:
            mapped[name] = ""
    return mapped

# ---------------------- Main processing per image ----------------------
def process_image_bytes(img_bytes, debug=False):
    # load image as BGR
    arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        # fallback: open via PIL then convert
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    detect = detect_table_grid(img_bgr, debug=False)
    if detect is None:
        return [], None
    xs, ys = detect

    rows_cells = extract_cells_from_grid(img_bgr, xs, ys, expected_cols=8)
    records = []
    for rc in rows_cells:
        mapped = map_row_to_schema(rc)
        records.append(mapped)

    debug_img = None
    if debug:
        # return debug image with overlay lines
        _, _, debug_img = detect_table_grid(img_bgr, debug=True)
    return records, debug_img

# ---------------------- Streamlit UI ----------------------
st.set_page_config(layout="wide")
st.title("OCR Bảng công chứng → Xuất Excel (mỗi ảnh nhiều dòng)")

col1, col2 = st.columns([2,1])
with col2:
    use_sample = st.checkbox("Dùng ảnh mẫu /mnt/data/3.jpg", value=True)
    show_debug = st.checkbox("Hiển thị ảnh debug (overlay grid)", value=False)

uploaded = st.file_uploader("Upload ảnh (jpg/png). Mỗi ảnh có thể chứa 5-25 dòng", accept_multiple_files=True, type=["jpg","jpeg","png"])

# build list of files to process
files_to_process = []
if use_sample:
    if os.path.exists(SAMPLE_PATH):
        files_to_process.append(("sample_3.jpg", open(SAMPLE_PATH, "rb").read()))
    else:
        st.warning(f"Ảnh mẫu không tồn tại ở {SAMPLE_PATH}")

if uploaded:
    for f in uploaded:
        files_to_process.append((f.name, f.read()))

if not files_to_process:
    st.info("Chưa có file để xử lý. Upload hoặc tích dùng ảnh mẫu.")
else:
    if st.button("Chạy OCR và Xuất Excel"):
        all_records = []
        # optional debug gallery
        debug_cols = []
        for fname, b in files_to_process:
            st.write(f"Xử lý: {fname}")
            try:
                recs, debug_img = process_image_bytes(b, debug=show_debug)
            except Exception as e:
                st.error(f"Lỗi khi xử lý {fname}: {e}")
                recs = []
                debug_img = None

            if not recs:
                st.warning(f"Không tìm thấy hàng nào trong {fname} (có thể lưới bị mờ). Thử tăng chất lượng ảnh/phóng to vùng bảng.")
            else:
                for r in recs:
                    r["__source_file"] = fname
                    all_records.append(r)

            if show_debug and debug_img is not None:
                # convert BGR -> RGB
                rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption=f"Grid overlay: {fname}", use_column_width=True)

        if not all_records:
            st.error("Không thu được bản ghi nào. Hãy kiểm tra ảnh, đảm bảo lưới rõ.")
        else:
            df = pd.DataFrame(all_records)
            # ensure columns order: requested COLUMNS + source
            out_cols = COLUMNS + ["__source_file"]
            for c in out_cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[out_cols]
            st.dataframe(df.head(50))

            # export excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="data")
            buf.seek(0)
            st.download_button("⬇ Tải Excel kết quả", data=buf,
                               file_name="ocr_cong_chung_multirow.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
