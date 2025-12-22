import sys
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ============================================================
# 1. 加载 bge-m3 模型
# ============================================================

class EmbeddingClassifier:
    def __init__(self):
        print("Loading BGE-m3 embedding model...")
        self.model = SentenceTransformer("BAAI/bge-m3")

        # 正文示例
        self.pos_examples = [
            "公司发布公告，2019年一季度实现营业收入同比增长，经营性现金流持续改善。",
            "光伏单晶硅片行业景气度回升，公司产能扩张顺利推进，盈利能力显著提升。",
            "公司半导体硅片业务进入放量阶段，8英寸和12英寸硅片需求持续增长。",
            "公司通过严格成本控制和精益化管理，有效降低经营成本，提升盈利能力。",
            "我们预计公司未来三年净利润将保持高速增长，维持对公司的增持评级。",
            "公司与地方政府签署合作协议，拟投资建设新的单晶硅项目，进一步扩大产能。",
            "随着光伏政策落地和海外需求旺盛，公司硅片价格企稳回升，盈利拐点明确。",
            "公司半导体材料业务毛利率提升，受益于产品结构升级和大尺寸硅片放量。",
            "公司发布员工持股计划，有助于绑定核心员工利益，提升公司治理水平。",
            "公司预计未来将迎来光伏与半导体双轮驱动的增长新时代。"
        ]

        # 噪声示例
        self.neg_examples = [
            "主要股东（2019Q1）",
            "基本数据（截至2019年06月20日）",
            "收入结构（2018A）",
            "资产负债表",
            "现金流量表",
            "预测财务报表（单位：百万元）",
            "联系人：陈瑶",
            "当前股价：8.24元",
            "报告日期：2017年11月29日",
            "行业：电气设备和新能源",
            "总市值（亿元）",
            "流通A股（亿股）",
            "52周内股价区间（元）",
            "近3月换手率",
            "股价表现（最近一年）",
            "相对指数表现",
            "5.2/11.96",
            "27.55%",
            "0.39",
            "10.86亿",
            "单位：百万元",
            "指标",
            "投资评级体系与评级定义",
            "市场有风险，投资需谨慎",
            "免责条款",
            "公司基本情况（最新）",
            "《【联讯电新公司点评】中环股",
            "份(002129): 非公开发行助",
            "投资评级的说明:",
            "买入: 预期未来 6-12 个月内上涨幅度在 15%以上;",
            "增持: 预期未来 6-12 个月内上涨幅度在 5%-15%;",
            "中性: 预期未来 6-12 个月内变动幅度在 -5%-5%;",
            "减持: 预期未来 6-12 个月内下跌幅度在 5%以上。"

        ]


        # 预计算 embedding
        self.pos_emb = self.model.encode(self.pos_examples, convert_to_tensor=True)
        self.neg_emb = self.model.encode(self.neg_examples, convert_to_tensor=True)

    # ============================================================
    # classify：加入短句增强逻辑
    # ============================================================
    def classify(self, line: str) -> str:
        text = line.strip()
        if not text:
            return "噪声"

        length = len(text)

        # -------------------------
        # 1. 强规则：短句噪声优先
        # -------------------------
        if length <= 20:
            if re.match(r"^[\d\.\-/%]+$", text) or \
               re.search(r"(亿|万|元|%|同比|环比|YOY|《|》|【|】)", text):
                return "噪声"

        # -------------------------
        # 2. embedding 相似度
        # -------------------------
        emb = self.model.encode(text, convert_to_tensor=True)
        pos_sim = util.cos_sim(emb, self.pos_emb).max().item()
        neg_sim = util.cos_sim(emb, self.neg_emb).max().item()

        # -------------------------
        # 3. 短句惩罚（弱规则）
        # -------------------------
        if length < 20:
            # 如果短句包含明显正文词，则不惩罚
            if not re.search(r"(收入|净利润|增长|业务|产能)", text):
                neg_sim += 0.3  # 轻微惩罚

        # -------------------------
        # 4. 最终决策
        # -------------------------
        NEG_THRESHOLD = 0.3

        if neg_sim > NEG_THRESHOLD and neg_sim > pos_sim:
            return "噪声"
        else:
            return "正文"


# ============================================================
# 2. recheck：上下文增强二次判断
# ============================================================

def recheck_with_context(lines, labels, clf, i):
    line = lines[i].strip()

    if len(line.strip()) < 10:
        return "噪声"

    if not line:
        return "噪声"

    prev1 = lines[i-1] if i > 0 else ""
    next1 = lines[i+1] if i < len(lines)-1 else ""

    prev_pos = ""
    for j in range(i-1, -1, -1):
        if labels[j] == "正文":
            prev_pos = lines[j]
            break

    next_pos = ""
    for j in range(i+1, len(lines)):
        if labels[j] == "正文":
            next_pos = lines[j]
            break

    candidates = [
        line,
        prev1 + " " + line,
        line + " " + next1
        # prev_pos + " " + line + " " + next_pos,
    ]

    for text in candidates:
        if clf.classify(text) == "正文":
            return "正文"

    return "噪声"


# ============================================================
# 3. 主程序
# ============================================================

def main():
    input_path = r"D:\MFIN\7036\reports_txt\TCL中环\20190401_粤开证券_TCL中环_【联讯电新公司点评】：中环股份：业绩符合预期，产能顺利释放助公司成长.txt"
    output_path = r"D:\MFIN\7036\reports_txt\TCL中环\20190401_粤开证券_TCL中环_【联讯电新公司点评】：中环股份：业绩符合预期，产能顺利释放助公司成长_clean.txt"

    print(f"读取文件: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    clf = EmbeddingClassifier()

    # 第一次分类
    labels = [clf.classify(line) for line in lines]

    # 第二次 recheck
    for i in range(len(lines)):
        if labels[i] == "噪声":
            labels[i] = recheck_with_context(lines, labels, clf, i)

    print(f"写入标签结果: {output_path}")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for label, line in zip(labels, lines):
            out_f.write(f"{label}\t{line}\n")

    print("标注完成！")


if __name__ == "__main__":
    main()
