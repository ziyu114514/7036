import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================================================
# 1. 轻量 LLM 过滤器（正文 vs 噪声）
# ============================================================

class LightLLMFilter:
    def __init__(self, model_name="uer/roberta-base-finetuned-chinanews-chinese"):
        print("Loading lightweight LLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.prompt = (
            "判断下面这句话是否属于研报正文内容，而不是表格、字段名、股东列表、财务数据、"
            "百分比行、碎片行或噪声：\n\n"
            "句子：\"{line}\"\n\n"
            "回答：正文 或 噪声"
        )

    def is_clean(self, line: str) -> bool:
        if not line.strip():
            return False

        text = self.prompt.format(line=line)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        pred = logits.argmax().item()
        return pred == 1


# ============================================================
# 2. 完整 LLM-only 清洗 Pipeline
# ============================================================

class TextCleanerPipeline:
    def __init__(self):
        self.llm_filter = LightLLMFilter()

    def clean(self, lines):
        cleaned = []
        for line in lines:
            if self.llm_filter.is_clean(line):
                cleaned.append(line)

        # fallback：如果全被删光，保留原始文本
        if not cleaned:
            print("Warning: LLM filtered out all lines, fallback to original text.")
            return lines

        return cleaned




# ============================================================
# 5. 主程序：输入 txt → 输出清洗 txt
# ============================================================

def main():
    

    input_path = r"D:\MFIN\7036\reports_txt\TCL中环\20190626_上海证券_TCL中环_董事长增持彰显信心，光伏、半导体双轮驱动.txt"
    output_path = r"D:\MFIN\7036\reports_txt\TCL中环\20190626_上海证券_TCL中环_董事长增持彰显信心，光伏、半导体双轮驱动_clean.txt"

    print(f"读取文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    cleaner = TextCleanerPipeline()
    cleaned = cleaner.clean(lines)

    print(f"写入清洗结果: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned))

    print("清洗完成！")


if __name__ == "__main__":
    main()
