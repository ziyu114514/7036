import os
import re

import fitz  # PyMuPDF
from PIL import Image

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

# =========================
# 1. 初始化 Surya 模型（默认硬件配置）
# =========================

foundation = FoundationPredictor()
detector = DetectionPredictor()
recognizer = RecognitionPredictor(foundation)
layout_predictor = LayoutPredictor(
    FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
)

# =========================
# 2. PDF 渲染（PyMuPDF）
# =========================

def pdf_to_images_fast(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        try:
            pix = page.get_pixmap(dpi=dpi, annots=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
        except Exception as e:
            print(f"[警告] 渲染失败，跳过该页: {e}")
            continue
    return images

# =========================
# 3. Layout 区域判断（整页 OCR 后过滤）
# =========================

def line_in_bbox(line_bbox, region_bbox):
    x1, y1, x2, y2 = line_bbox
    rx1, ry1, rx2, ry2 = region_bbox
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)

def is_in_text_region(line, layout_page):
    for region in layout_page.bboxes:
        if region.label in ["Text", "SectionHeader", "ListItem"]:
            if line_in_bbox(line.bbox, region.bbox):
                return True
    return False

# =========================
# 4. blank-aware 行过滤
# =========================

def is_blank_line(line, threshold=0.85):
    """
    利用 Surya 的 blank_ratio 过滤空白区域。
    threshold 越高越严格（0.85~0.95 推荐）
    """
    if hasattr(line, "blank_ratio"):
        return line.blank_ratio >= threshold

    if not line.text or line.text.strip() == "":
        return True

    return False

# =========================
# 5. Table-aware 区域过滤（关键优化）
# =========================

def in_any_region(line_bbox, region_list):
    for (x1, y1, x2, y2) in region_list:
        if not (line_bbox[2] < x1 or line_bbox[0] > x2 or line_bbox[3] < y1 or line_bbox[1] > y2):
            return True
    return False

# =========================
# 6. 文本清洗（基础 + 严格）
# =========================

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
PHONE_RE = re.compile(r"\d{3,4}-\d{7,8}")

def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)

def is_contact_info(text: str) -> bool:
    if EMAIL_RE.search(text): return True
    if URL_RE.search(text): return True
    if PHONE_RE.search(text): return True
    if any(k in text for k in ["电话", "邮箱", "研究员", "SAC"]): return True
    return False

def is_footnote(text: str) -> bool:
    return text.startswith(("注", "备注", "*"))

def is_page_number(text: str) -> bool:
    return re.fullmatch(r"\d{1,3}", text) is not None

def is_toc_line(text: str) -> bool:
    if re.search(r"\.{3,}", text): return True
    if re.search(r"\d+$", text) and " " in text: return True
    return False

def is_header_footer(text: str) -> bool:
    keywords = ["请务必阅读", "重要声明", "证券研究报告", "研究所"]
    return any(k in text for k in keywords)

def is_fragment(text: str) -> bool:
    if is_contact_info(text): return True
    if any(k in text for k in ["当前价", "目标价", "买入评级", "维持评级", "上调评级"]): return True
    if re.fullmatch(r"[0-9\.\-]+(元|倍)?", text): return True
    if re.fullmatch(r"\d+(\.\d+)?\s*(元|倍|万股|亿元|万股)", text): return True
    return False

def is_short_fragment(text: str) -> bool:
    return len(text.strip()) <= 2

def is_meta_info(text: str) -> bool:
    meta_keywords = [
        "资料来源", "来源", "Wind", "WIND",
        "分析师", "执业证书", "免责声明",
        "请阅读", "重要声明",
        "风险提示",
        "证券有限责任公司",
        "投资评级说明", "行业的投资评级说明",
        "研究发展部",
        "地址:", "网址:", "邮编:",
    ]
    return any(k in text for k in meta_keywords)

def clean_line_basic(line: str):
    line = line.strip()
    if not line: return None
    if is_page_number(line): return None
    if is_toc_line(line): return None
    if is_header_footer(line): return None
    return line

def clean_text_strict(text: str):
    text = remove_html(text).strip()
    if not text: return None
    if is_footnote(text): return None
    if is_contact_info(text): return None
    if is_fragment(text): return None
    if is_meta_info(text): return None
    if is_short_fragment(text): return None
    return text

# =========================
# 7. 单个 PDF 的完整处理（整页 OCR → Layout → blank + table + 清洗）
# =========================

def pdf_to_text_fast(pdf_path: str) -> str:
    images = pdf_to_images_fast(pdf_path, dpi=150)
    if not images:
        return ""

    # 整页 OCR
    ocrs = recognizer(images, det_predictor=detector)

    # 整页 Layout
    layouts = layout_predictor(images)

    output_lines = []

    for i in range(len(images)):
        ocr_page = ocrs[i]
        layout_page = layouts[i]

        # 提取表格区域（关键）
        table_regions = [r.bbox for r in layout_page.bboxes if r.label == "Table"]

        for line in ocr_page.text_lines:

            # 0. blank 过滤
            if is_blank_line(line):
                continue

            # 1. 表格区域过滤（解决“主要股东 27.55%”混入正文）
            if in_any_region(line.bbox, table_regions):
                continue

            raw = line.text
            if not raw:
                continue

            # 2. Layout 过滤
            if not is_in_text_region(line, layout_page):
                continue

            # 3. 基础清洗
            basic = clean_line_basic(raw)
            if not basic:
                continue

            # 4. 严格清洗
            strict = clean_text_strict(basic)
            if not strict:
                continue

            output_lines.append(strict)

    return "\n".join(output_lines)

# =========================
# 8. 批量处理（默认单进程）
# =========================

def build_output_path(pdf_path: str, pdf_dir: str, txt_dir: str) -> str:
    rel = os.path.relpath(pdf_path, pdf_dir)
    parts = rel.split(os.sep)

    is_deep = "深度报告" in parts

    if is_deep:
        idx = parts.index("深度报告")
        stock = parts[idx - 1]
        out_dir = os.path.join(txt_dir, stock, "深度报告")
    else:
        stock = parts[0]
        out_dir = os.path.join(txt_dir, stock)

    os.makedirs(out_dir, exist_ok=True)
    txt_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
    return os.path.join(out_dir, txt_filename)

# def batch_convert(pdf_dir: str, txt_dir: str):
#     pdf_paths = []
#     for root, _, files in os.walk(pdf_dir):
#         for file in files:
#             if file.lower().endswith(".pdf"):
#                 pdf_paths.append(os.path.join(root, file))

#     print(f"共找到 {len(pdf_paths)} 个 PDF，单进程顺序处理。")

#     for pdf_path in pdf_paths:
#         txt_path = build_output_path(pdf_path, pdf_dir, txt_dir)
#         print(f"转换中: {pdf_path} → {txt_path}")
#         text = pdf_to_text_fast(pdf_path)
#         with open(txt_path, "w", encoding="utf-8") as f:
#             f.write(text)
def batch_convert(pdf_dir: str, txt_dir: str):
    pdf_paths = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))

    print(f"共找到 {len(pdf_paths)} 个 PDF，单进程顺序处理。")

    for pdf_path in pdf_paths:
        txt_path = build_output_path(pdf_path, pdf_dir, txt_dir)

        # -----------------------------
        # 如果 txt 已存在 → 跳过
        # -----------------------------
        if os.path.exists(txt_path):
            print(f"跳过（已存在）: {txt_path}")
            continue

        print(f"转换中: {pdf_path} → {txt_path}")
        text = pdf_to_text_fast(pdf_path)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    print("全部处理完成！")

if __name__ == "__main__":
    pdf_root = r"East_money_research_report_download\reports_pdf"
    txt_root = r"reports_txt"
    batch_convert(pdf_root, txt_root)
