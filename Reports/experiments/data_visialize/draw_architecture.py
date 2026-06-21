import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
import os, argparse

LANG = "en"

FIG_W, FIG_H = 6.0, 5.0
SUB_W_L, SUB_W_R = 2.70, 2.80
GAP_X = 0.22
SUB_H = 4.55
LEFT_X = 0.08
RIGHT_X = LEFT_X + SUB_W_L + GAP_X

PAD_T = 0.30
PAD_B = 0.25
BOFF_T = 0.42   # inside sub_box top margin
BOFF_B = 0.42   # inside sub_box bottom margin

BOX_W = 1.60
BOX_H = 0.28

L_LAYERS = 4   # offline: doc, chunk, 3-mid, 3-out
R_LAYERS = 8   # online: query, rewriter, 4-path, rrf, pool, rerank, topk, gen_ans

LAYOUT_BASE = {
    "sub_left_label":   {"en": "Offline Phase: Index Construction",      "zh": "离线阶段：索引构建"},
    "sub_right_label":  {"en": "Online Phase: Retrieval & Generation",   "zh": "在线阶段：检索与生成"},
    "doc":              {"en": "Knowledge Base\nDocuments",              "zh": "知识库文档"},
    "chunk":            {"en": "Document Chunking\n(Text Blocks)",       "zh": "文档分块\n（文本块）"},
    "gen_meta":         {"en": "LLM: Generate\nMeta-Questions",          "zh": "LLM：生成\n元问题"},
    "gen_kw":           {"en": "LLM: Extract\nKeywords",                 "zh": "LLM：提取\n关键词"},
    "idx_vec":          {"en": "Chunk\nVector Index",                    "zh": "文本块\n向量索引"},
    "idx_meta":         {"en": "Meta-Question\nVector Index",            "zh": "元问题\n向量索引"},
    "idx_kw":           {"en": "Keyword\nVector Index",                  "zh": "关键词\n向量索引"},
    "idx_bm25":         {"en": "Chunk Lexical\nIndex: BM25",             "zh": "文本块词法\n索引：BM25"},
    "query":            {"en": "User Query",                             "zh": "用户查询"},
    "rewrite":          {"en": "Query Rewriter\n(Q_sem, Q_broad, Q_keyw)", "zh": "查询重写器\n（Q_sem, Q_broad, Q_keyw）"},
    "rrf":              {"en": "RRF\nFusion",                            "zh": "RRF\n融合"},
    "pool":             {"en": "Candidate Pooling\n& Dedup",             "zh": "候选池化\n与去重"},
    "rerank":           {"en": "Cross-Encoder\nReranking",               "zh": "Cross-Encoder\n重排序"},
    "topk":             {"en": "Top-K_final\nFiltering",                 "zh": "Top-K_final\n过滤"},
    "gen_ans":          {"en": "Answer Generation\n(Qwen3-30B)",         "zh": "生成回答\n（Qwen3-30B）"},
}

def t(key):
    return LAYOUT_BASE[key][LANG]

colors = {
    "input":   ("#f8fafc", "#94a3b8", "#0f172a"),
    "process": ("#eff6ff", "#60a5fa", "#1e40af"),
    "llm":     ("#f3e8ff", "#c084fc", "#6b21a8"),
    "index":   ("#f0fdf4", "#4ade80", "#166534"),
    "fusion":  ("#fff7ed", "#fb923c", "#9a3412"),
    "output":  ("#1e293b", "#0f172a", "#ffffff"),
}

def draw_box(ax, cx, cy, w, h, text, face, edge, fontcol, fs=6.5):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.05", facecolor=face,
                       edgecolor=edge, linewidth=1.0, zorder=3)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=fontcol, fontweight="bold", linespacing=1.1, zorder=4)

def draw_sub_box(ax, x, y, w, h, label):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor="#fafbfc", edgecolor="#cbd5e1",
                       linewidth=1.2, linestyle="--", zorder=1)
    ax.add_patch(r)
    ax.text(x + w/2, y + h + 0.05, label, ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#0f172a", zorder=2)

def arrow(ax, x1, y1, x2, y2, x_src=None, x_dst=None):
    """Straight arrow from box bottom to box top."""
    margin = 0.03
    ys = y1 - BOX_H/2 - margin
    yd = y2 + BOX_H/2 + margin
    xs = x_src if x_src else x1
    xd = x_dst if x_dst else x2
    ax.annotate("", xy=(xd, yd), xytext=(xs, ys),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#64748b"), zorder=2)

def arrow_curved_elbow(ax, x0, y0, x1, turn_y, y2):
    """Quadratic bezier from Chunk bottom to (x1,turn_y), then vertical down to index top.
    Control point is (x1, y0) so curve leaves horizontally and arrives vertically."""
    margin = 0.03
    p0 = (x0, y0 - BOX_H/2 - margin)
    ctrl = (x1, p0[1])
    p1 = (x1, turn_y)
    p2 = (x1, y2 + BOX_H/2 + margin)

    verts = [p0, ctrl, p1, p2]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor='#64748b', lw=0.8, zorder=2)
    ax.add_patch(patch)

    # arrow head at the very bottom
    ax.annotate("", xy=p2, xytext=(x1, turn_y),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#64748b"), zorder=2)

def layers_y(n, i, y_top, y_bot):
    """y position of layer i (0-indexed) out of n layers, evenly spaced y_top → y_bot."""
    if n <= 1:
        return (y_top + y_bot) / 2
    return y_top - (y_top - y_bot) * i / (n - 1)

def main(lang):
    global LANG
    LANG = lang

    if LANG == "zh":
        plt.rcParams.update({"font.serif": ["Noto Serif CJK JP", "Nimbus Roman", "DejaVu Serif"]})
    else:
        plt.rcParams.update({"font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "figure.dpi": 300, "savefig.dpi": 300})

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # content area: from sub_box_top-BOFF_T down to sub_box_bot+BOFF_B
    sub_top = PAD_B + SUB_H
    c_top = sub_top - BOFF_T
    c_bot = PAD_B + BOFF_B

    # ========== LEFT: OFFLINE ==========
    draw_sub_box(ax, LEFT_X, PAD_B, SUB_W_L, SUB_H, t("sub_left_label"))
    cl = LEFT_X + SUB_W_L / 2

    ys_l = [layers_y(L_LAYERS, i, c_top, c_bot) for i in range(L_LAYERS)]

    draw_box(ax, cl, ys_l[0], BOX_W, BOX_H, t("doc"), *colors["input"])
    draw_box(ax, cl, ys_l[1], BOX_W, BOX_H, t("chunk"), *colors["process"])
    arrow(ax, cl, ys_l[0], cl, ys_l[1])

    # L2: GenQ, GenK — LLM processing from Chunk (left side)
    l2_keys = [("gen_meta", colors["llm"]), ("gen_kw", colors["llm"])]
    l2_offsets = [-0.85, -0.15]
    for xo, (key, sty) in zip(l2_offsets, l2_keys):
        draw_box(ax, cl + xo, ys_l[2], 0.60, BOX_H + 0.03, t(key), *sty, fs=6)
        # Straight diagonal from Chunk bottom center to L2 box top center
        arrow(ax, cl, ys_l[1], cl + xo, ys_l[2], x_src=cl, x_dst=cl + xo)

    # L3: LLM indexes on left, direct indexes on right
    l3_keys = [
        ("idx_meta", colors["index"]), ("idx_kw", colors["index"]),
        ("idx_vec", colors["index"]), ("idx_bm25", colors["index"]),
    ]
    l3_offsets = [-0.85, -0.15, +0.48, +1.08]
    l3_w = 0.48
    for i, (xo, (key, sty)) in enumerate(zip(l3_offsets, l3_keys)):
        draw_box(ax, cl + xo, ys_l[3], l3_w, BOX_H + 0.03, t(key), *sty, fs=5.5)
        if i == 0:      # IdxQ ← GenQ
            arrow(ax, cl - 0.85, ys_l[2], cl + xo, ys_l[3])
        elif i == 1:    # IdxK ← GenK
            arrow(ax, cl - 0.15, ys_l[2], cl + xo, ys_l[3])
        elif i == 2:    # IdxVec ← Chunk: smooth curve to L2 height then vertical down
            arrow_curved_elbow(ax, cl, ys_l[1], cl + xo, ys_l[2], ys_l[3])
        else:           # IdxBM ← Chunk: smooth curve to L2 height then vertical down
            arrow_curved_elbow(ax, cl, ys_l[1], cl + xo, ys_l[2], ys_l[3])

    # ========== RIGHT: ONLINE ==========
    draw_sub_box(ax, RIGHT_X, PAD_B, SUB_W_R, SUB_H, t("sub_right_label"))
    cr = RIGHT_X + SUB_W_R / 2

    ys_r = [layers_y(R_LAYERS, i, c_top, c_bot) for i in range(R_LAYERS)]

    draw_box(ax, cr, ys_r[0], BOX_W, BOX_H, t("query"), *colors["input"])
    draw_box(ax, cr, ys_r[1], BOX_W, BOX_H + 0.04, t("rewrite"), *colors["process"], fs=6)
    arrow(ax, cr, ys_r[0], cr, ys_r[1])

    # L2: four paths — single-line
    po = [-0.90, -0.30, 0.30, 0.90]
    pkw = 0.52; pkh = 0.22
    pt_en = ["1:Semantic", "2:Intent", "3A:Lexical", "3B:KW-Semantic"]
    pt_zh = ["1：语义", "2：意图", "3A：词法", "3B：关键词语义"]
    pt = pt_zh if LANG == "zh" else pt_en
    for pk, xo in zip(pt, po):
        draw_box(ax, cr + xo, ys_r[2], pkw, pkh, pk, *colors["process"], fs=5.5)
        arrow(ax, cr, ys_r[1], cr + xo, ys_r[2], x_dst=cr + xo)

    # L3: RRF
    rrf_x = cr + (po[2] + po[3]) / 2
    draw_box(ax, rrf_x, ys_r[3], 0.38, 0.22, t("rrf"), *colors["fusion"], fs=6)
    arrow(ax, cr + po[2], ys_r[2], rrf_x, ys_r[3])
    arrow(ax, cr + po[3], ys_r[2], rrf_x, ys_r[3])

    # L4: pooling
    draw_box(ax, cr, ys_r[4], BOX_W, 0.22, t("pool"), *colors["process"], fs=6.5)
    arrow(ax, cr + po[0], ys_r[2], cr, ys_r[4], x_dst=cr)
    arrow(ax, cr + po[1], ys_r[2], cr, ys_r[4], x_dst=cr)
    arrow(ax, rrf_x, ys_r[3], cr, ys_r[4], x_dst=cr)

    # L5: rerank
    draw_box(ax, cr, ys_r[5], BOX_W, 0.22, t("rerank"), *colors["llm"], fs=6.5)
    arrow(ax, cr, ys_r[4], cr, ys_r[5])

    # L6: topk
    draw_box(ax, cr, ys_r[6], BOX_W, 0.22, t("topk"), *colors["process"], fs=6.5)
    arrow(ax, cr, ys_r[5], cr, ys_r[6])

    # L7: gen_ans
    draw_box(ax, cr, ys_r[7], BOX_W, 0.22, t("gen_ans"), *colors["output"], fs=6.5)
    arrow(ax, cr, ys_r[6], cr, ys_r[7])

    out = f"/home/pushihao/RAG/Reports/docs/pics/rag_architecture_{LANG}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(pad=0.10)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["en", "zh", "both"], default="en")
    a = p.parse_args()
    if a.lang == "both":
        main("en"); main("zh")
    else:
        main(a.lang)
