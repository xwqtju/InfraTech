from __future__ import annotations

import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path("models/deepseek_v4/CSA计算流图.pptx")
PREVIEW = Path("models/deepseek_v4/CSA计算流图_preview")
PREVIEW.mkdir(parents=True, exist_ok=True)

SLIDE_W = 13_333_333
SLIDE_H = 7_500_000


def emu(v: float) -> int:
    return int(round(v))


def px(x: float, base_w: float = 1920.0) -> int:
    return emu(x / base_w * SLIDE_W)


def py(y: float, base_h: float = 1080.0) -> int:
    return emu(y / base_h * SLIDE_H)


COLORS = {
    "bg": "FFFFFF",
    "ink": "151515",
    "muted": "60646C",
    "blue": "7B93F4",
    "blue_dark": "586DCE",
    "blue_soft": "F1F5FF",
    "orange": "FF9F1A",
    "orange_dark": "E88900",
    "orange_soft": "FFF6E8",
    "green": "91D64B",
    "green_dark": "6DAE2B",
    "green_soft": "F5FBEF",
    "cyan": "58D0C4",
    "cyan_dark": "32A99D",
    "red": "FA1B1B",
    "red_dark": "C91414",
    "purple": "7533AD",
    "purple_dark": "5C238B",
    "purple_soft": "F8F2FF",
    "black": "222222",
    "line": "333333",
    "pink": "FFC1C1",
}


def solid_fill(color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def line_xml(color: str = "333333", width: int = 15_000, dash: str | None = None) -> str:
    dash_xml = f'<a:prstDash val="{dash}"/>' if dash else ""
    return f'<a:ln w="{width}">{solid_fill(color)}{dash_xml}</a:ln>'


def text_body(text: str, size: int, color: str, bold: bool = False, align: str = "ctr") -> str:
    paragraphs = text.split("\n")
    out = []
    for line in paragraphs:
        b = ' b="1"' if bold else ""
        out.append(
            f'<a:p><a:pPr algn="{align}"/>'
            f'<a:r><a:rPr lang="zh-CN" sz="{size}"{b}>{solid_fill(color)}</a:rPr>'
            f"<a:t>{escape(line)}</a:t></a:r><a:endParaRPr lang=\"zh-CN\" sz=\"{size}\"/></a:p>"
        )
    return (
        '<p:txBody><a:bodyPr wrap="square" anchor="mid" anchorCtr="1"/>'
        '<a:lstStyle/>'
        + "".join(out)
        + "</p:txBody>"
    )


def shape(
    sid: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    fill: str = "FFFFFF",
    stroke: str = "333333",
    text_color: str = "FFFFFF",
    size: int = 1800,
    bold: bool = False,
    radius: str = "roundRect",
    dash: str | None = None,
    align: str = "ctr",
) -> str:
    return (
        '<p:sp>'
        f'<p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{px(x)}" y="{py(y)}"/><a:ext cx="{px(w)}" cy="{py(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>'
        f'{solid_fill(fill)}{line_xml(stroke, 12_000, dash)}</p:spPr>'
        f'{text_body(text, size, text_color, bold, align) if text else "<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>"}'
        '</p:sp>'
    )


def text_box(
    sid: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    size: int,
    color: str = "151515",
    bold: bool = False,
    align: str = "l",
) -> str:
    return (
        '<p:sp>'
        f'<p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{px(x)}" y="{py(y)}"/><a:ext cx="{px(w)}" cy="{py(h)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
        f'{text_body(text, size, color, bold, align)}'
        '</p:sp>'
    )


def line(
    sid: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "333333",
    width: int = 18_000,
    dash: str | None = None,
    arrow: bool = True,
) -> str:
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1) or 1
    h = abs(y2 - y1) or 1
    flip_h = ' flipH="1"' if x2 < x1 else ""
    flip_v = ' flipV="1"' if y2 < y1 else ""
    head = '<a:headEnd type="triangle"/>' if arrow else ""
    dash_xml = f'<a:prstDash val="{dash}"/>' if dash else ""
    return (
        '<p:cxnSp>'
        f'<p:nvCxnSpPr><p:cNvPr id="{sid}" name="Arrow {sid}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
        f'<p:spPr><a:xfrm{flip_h}{flip_v}><a:off x="{px(x)}" y="{py(y)}"/><a:ext cx="{px(w)}" cy="{py(h)}"/></a:xfrm>'
        '<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
        f'<a:ln w="{width}">{solid_fill(color)}{dash_xml}{head}</a:ln>'
        '</p:spPr></p:cxnSp>'
    )


def chevron(
    sid: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fill: str,
    stroke: str,
    size: int = 1700,
) -> str:
    return shape(sid, name, x, y, w, h, text, fill, stroke, "FFFFFF", size, False, "trapezoid")


def circle(sid: int, name: str, x: float, y: float, d: float, text: str, fill: str, stroke: str) -> str:
    return shape(sid, name, x, y, d, d, text, fill, stroke, "FFFFFF", 1700, True, "ellipse")


def group_box(sid: int, x: float, y: float, w: float, h: float, title: str, color: str, soft: str) -> list[str]:
    return [
        shape(sid, title + " box", x, y, w, h, "", soft, color, radius="roundRect", dash="dash"),
        text_box(sid + 1, title, x + 20, y + 16, w - 40, 34, title, 2400, color, True),
    ]


def slide_xml(parts: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        + "".join(parts)
        + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
    )


def title_bar(title: str, subtitle: str = "") -> list[str]:
    parts = [
        text_box(10, "slide title", 60, 28, 1200, 48, title, 3000, COLORS["ink"], True),
    ]
    if subtitle:
        parts.append(text_box(11, "slide subtitle", 60, 78, 1400, 34, subtitle, 1500, COLORS["muted"]))
    return parts


def add_footer(parts: list[str], idx: int) -> None:
    parts.append(text_box(900, "footer", 60, 1036, 1100, 24, "CSA Computation Graph | DeepSeek V4 attention notes", 900, "7A7A7A"))
    parts.append(text_box(901, "page", 1810, 1036, 60, 24, str(idx), 900, "7A7A7A", align="r"))


def overview_slide() -> str:
    p = title_bar(
        "CSA 计算流图总览",
        "Compressed Sparse Attention = Token 压缩 + Indexer Top-k + Window/Compress 稀疏 MQA",
    )
    sid = 20
    p += group_box(sid, 55, 145, 1810, 850, "CSA 主计算路径", COLORS["orange_dark"], COLORS["orange_soft"])
    sid += 10
    p += group_box(sid, 95, 235, 420, 330, "Q 路径", COLORS["orange_dark"], "FFFFFF")
    sid += 10
    q_nodes = [
        ("Linear(Q down)", 180, 292),
        ("RMSNorm", 192, 360),
        ("Linear(Q up)", 180, 428),
        ("Q head Norm", 180, 496),
    ]
    for txt, x, y in q_nodes:
        p.append(shape(sid, txt, x, y, 170, 40, txt, COLORS["orange"], COLORS["orange_dark"], size=1150))
        sid += 1
    for y in [332, 400, 468]:
        p.append(line(sid, 265, y, 265, y + 28)); sid += 1
    p.append(shape(sid, "RoPE", 410, 496, 80, 40, "RoPE", COLORS["orange"], COLORS["orange_dark"], size=1100)); sid += 1
    p.append(line(sid, 350, 516, 410, 516)); sid += 1
    p.append(shape(sid, "Q", 565, 496, 76, 40, "Q", COLORS["orange"], COLORS["orange_dark"], size=1200)); sid += 1
    p.append(line(sid, 490, 516, 565, 516)); sid += 1

    p += group_box(sid, 95, 610, 600, 300, "Token Level Compressor (C4A)", COLORS["blue_dark"], COLORS["blue_soft"]); sid += 10
    comp = [
        ("Linear(KV proj)", 150, 684, COLORS["blue"], COLORS["blue_dark"]),
        ("kv_state", 150, 780, COLORS["purple"], COLORS["purple_dark"]),
        ("Linear(Gate proj)", 415, 684, COLORS["blue"], COLORS["blue_dark"]),
        ("score_state", 415, 780, COLORS["purple"], COLORS["purple_dark"]),
    ]
    for txt, x, y, f, s in comp:
        p.append(shape(sid, txt, x, y, 190, 45, txt, f, s, size=1100)); sid += 1
    p.append(line(sid, 245, 729, 245, 780)); sid += 1
    p.append(line(sid, 510, 729, 510, 780)); sid += 1
    p.append(shape(sid, "compress cache", 305, 860, 220, 42, "KV cache (compress)", COLORS["red"], COLORS["red_dark"], size=1050)); sid += 1
    p.append(line(sid, 245, 825, 305, 881)); sid += 1
    p.append(line(sid, 510, 825, 525, 881)); sid += 1

    p += group_box(sid, 760, 235, 430, 330, "Indexer", COLORS["green_dark"], COLORS["green_soft"]); sid += 10
    for txt, y in [("Q up + RoPE", 296), ("Quant / Rotate", 365), ("Matmul KV", 434), ("ReLU + Sum", 503)]:
        p.append(shape(sid, txt, 875, y, 190, 42, txt, COLORS["green"], COLORS["green_dark"], size=1050)); sid += 1
    for y in [338, 407, 476]:
        p.append(line(sid, 970, y, 970, y + 27)); sid += 1
    p.append(shape(sid, "topk", 1100, 503, 70, 42, "Top-k", COLORS["green"], COLORS["green_dark"], size=950)); sid += 1
    p.append(line(sid, 1065, 524, 1100, 524, COLORS["green_dark"])); sid += 1
    p.append(text_box(sid, "topk shape", 1080, 552, 150, 24, "[1, top_k]", 850, COLORS["muted"])); sid += 1

    p += group_box(sid, 760, 610, 610, 300, "Sparse MQA Attention", COLORS["orange_dark"], COLORS["orange_soft"]); sid += 10
    p.append(shape(sid, "concat", 840, 680, 440, 50, "Concat KV: window + compress Top-k + sink", COLORS["orange"], COLORS["orange_dark"], size=1150)); sid += 1
    p.append(shape(sid, "attention", 840, 780, 440, 55, "Attention(MQA): softmax(QK^T/sqrt(d))V", COLORS["orange"], COLORS["orange_dark"], size=1050)); sid += 1
    p.append(line(sid, 1060, 730, 1060, 780)); sid += 1
    p.append(shape(sid, "O", 930, 858, 260, 42, "O: [1, heads, head_dim]", COLORS["orange"], COLORS["orange_dark"], size=1050)); sid += 1
    p.append(line(sid, 1060, 835, 1060, 858)); sid += 1
    p.append(line(sid, 641, 516, 840, 806)); sid += 1
    p.append(line(sid, 415, 860, 840, 706, COLORS["green_dark"], dash="dash")); sid += 1
    p.append(line(sid, 1170, 524, 1060, 680, COLORS["green_dark"], dash="dash")); sid += 1

    p += group_box(sid, 1440, 235, 365, 675, "输出投影与残差", COLORS["orange_dark"], "FFFFFF"); sid += 10
    for txt, y in [("RoPE inverse", 340), ("Linear(O down)", 450), ("Linear(O up)", 560)]:
        p.append(shape(sid, txt, 1525, y, 195, 46, txt, COLORS["orange"], COLORS["orange_dark"], size=1100)); sid += 1
    p.append(circle(sid, "residual add", 1600, 660, 54, "+", COLORS["blue"], COLORS["blue_dark"])); sid += 1
    p.append(shape(sid, "hc_post", 1515, 760, 220, 48, "hc_post", COLORS["blue"], COLORS["blue_dark"], size=1200)); sid += 1
    p.append(line(sid, 1622, 386, 1622, 450)); sid += 1
    p.append(line(sid, 1622, 496, 1622, 560)); sid += 1
    p.append(line(sid, 1622, 606, 1627, 660)); sid += 1
    p.append(line(sid, 1627, 714, 1627, 760)); sid += 1
    p.append(line(sid, 1190, 879, 1525, 363)); sid += 1
    p.append(text_box(sid, "params", 1450, 828, 330, 60, "heads=128  head_dim=512\nindex_topk=1024  window_size=128", 900, COLORS["muted"])); sid += 1
    add_footer(p, 1)
    return slide_xml(p)


def compressor_slide() -> str:
    p = title_bar("CSA: Token Level Compressor（C4A）", "将当前 token 写入压缩缓存，并滚动更新 kv_state / score_state")
    sid = 20
    p += group_box(sid, 80, 150, 1760, 830, "Compressor 数据流", COLORS["blue_dark"], COLORS["blue_soft"]); sid += 10
    p.append(text_box(sid, "input", 110, 205, 260, 32, "输入 h: [1, hidden_size]", 1200, COLORS["muted"])); sid += 1
    p.append(shape(sid, "kv proj", 190, 290, 300, 58, "Linear(KV proj)", COLORS["blue"], COLORS["blue_dark"], size=1500)); sid += 1
    p.append(shape(sid, "gate proj", 1120, 290, 320, 58, "Linear(Gate proj)", COLORS["blue"], COLORS["blue_dark"], size=1500)); sid += 1
    p.append(line(sid, 260, 237, 340, 290)); sid += 1
    p.append(line(sid, 260, 237, 1280, 290)); sid += 1
    p.append(shape(sid, "kv_state", 230, 430, 220, 58, "kv_state", COLORS["purple"], COLORS["purple_dark"], size=1500)); sid += 1
    p.append(shape(sid, "ape", 1010, 430, 130, 58, "APE", COLORS["blue"], COLORS["blue_dark"], size=1500)); sid += 1
    p.append(circle(sid, "plus", 1210, 430, 58, "+", COLORS["blue"], COLORS["blue_dark"])); sid += 1
    p.append(shape(sid, "score", 1340, 430, 230, 58, "score_state", COLORS["purple"], COLORS["purple_dark"], size=1500)); sid += 1
    p.append(line(sid, 340, 348, 340, 430)); sid += 1
    p.append(line(sid, 1280, 348, 1239, 430)); sid += 1
    p.append(line(sid, 1140, 459, 1210, 459)); sid += 1
    p.append(line(sid, 1268, 459, 1340, 459)); sid += 1
    p.append(shape(sid, "softmax", 1180, 590, 220, 58, "softmax", COLORS["purple"], COLORS["purple_dark"], size=1500)); sid += 1
    p.append(circle(sid, "mul", 760, 590, 58, "×", COLORS["green"], COLORS["green_dark"])); sid += 1
    p.append(shape(sid, "sum", 650, 730, 220, 58, "sum", COLORS["blue"], COLORS["blue_dark"], size=1500)); sid += 1
    p.append(shape(sid, "norm", 950, 730, 220, 58, "RMSNorm", COLORS["blue"], COLORS["blue_dark"], size=1500)); sid += 1
    p.append(shape(sid, "rope", 1250, 730, 180, 58, "RoPE", COLORS["blue"], COLORS["blue_dark"], size=1500)); sid += 1
    p.append(shape(sid, "compress cache", 1435, 840, 280, 68, "KV cache\n(compress)", COLORS["red"], COLORS["red_dark"], size=1400)); sid += 1
    p.append(line(sid, 1455, 488, 1290, 590)); sid += 1
    p.append(line(sid, 1290, 648, 790, 590, COLORS["purple_dark"])); sid += 1
    p.append(line(sid, 340, 488, 760, 590)); sid += 1
    p.append(line(sid, 789, 648, 760, 730)); sid += 1
    p.append(line(sid, 870, 759, 950, 759)); sid += 1
    p.append(line(sid, 1170, 759, 1250, 759)); sid += 1
    p.append(line(sid, 1430, 759, 1575, 840)); sid += 1
    p.append(text_box(sid, "note", 110, 860, 1050, 72, "要点：C4A 不把所有 token 都完整保留，而是用 gate/score 机制把历史信息压入有限槽位；后续 CSA 从 compress cache 中按 Indexer 结果取 Top-k。", 1250, COLORS["ink"])); sid += 1
    add_footer(p, 2)
    return slide_xml(p)


def indexer_slide() -> str:
    p = title_bar("CSA: Indexer", "用轻量相关性打分决定哪些压缩 KV 进入真实 Attention")
    sid = 20
    p += group_box(sid, 70, 150, 1780, 830, "Indexer 打分与选择", COLORS["green_dark"], COLORS["green_soft"]); sid += 10
    left = [("Linear(Q up_proj)", 210), ("RoPE", 320), ("Rotate", 430), ("Quant", 540)]
    for txt, y in left:
        p.append(shape(sid, txt, 300, y, 310, 62, txt, COLORS["green"], COLORS["green_dark"], size=1500)); sid += 1
    for y in [272, 382, 492]:
        p.append(line(sid, 455, y, 455, y + 48)); sid += 1
    p.append(shape(sid, "kv value", 1000, 520, 250, 64, "KV value", COLORS["red"], COLORS["red_dark"], size=1450)); sid += 1
    p.append(shape(sid, "matmul", 750, 540, 250, 62, "Matmul", COLORS["green"], COLORS["green_dark"], size=1500)); sid += 1
    p.append(line(sid, 610, 571, 750, 571)); sid += 1
    p.append(line(sid, 1000, 552, 1000, 552, arrow=False)); sid += 1
    p.append(line(sid, 1000, 552, 1000, 571)); sid += 1
    p.append(shape(sid, "relu", 750, 420, 250, 62, "ReLU", COLORS["green"], COLORS["green_dark"], size=1500)); sid += 1
    p.append(shape(sid, "sum", 750, 300, 250, 62, "Sum", COLORS["green"], COLORS["green_dark"], size=1500)); sid += 1
    p.append(shape(sid, "topk", 750, 180, 250, 62, "Top-k", COLORS["green"], COLORS["green_dark"], size=1500)); sid += 1
    p.append(line(sid, 875, 540, 875, 482)); sid += 1
    p.append(line(sid, 875, 420, 875, 362)); sid += 1
    p.append(line(sid, 875, 300, 875, 242)); sid += 1
    p.append(shape(sid, "weights", 300, 745, 310, 62, "Linear(weights proj)", COLORS["green"], COLORS["green_dark"], size=1450)); sid += 1
    p.append(shape(sid, "scale", 750, 745, 250, 62, "Scale", COLORS["green"], COLORS["green_dark"], size=1500)); sid += 1
    p.append(line(sid, 610, 776, 750, 776)); sid += 1
    p.append(line(sid, 875, 745, 875, 602)); sid += 1
    p.append(text_box(sid, "shape1", 1060, 185, 450, 44, "index_topk: [1, top_k]", 1400, COLORS["ink"], True)); sid += 1
    p.append(text_box(sid, "shape2", 1060, 250, 560, 170, "Indexer 的输出是索引控制流：\n1. 对压缩 KV 做相关性估计\n2. Top-k 只返回位置/索引\n3. 真正的 V 加权求和仍在 Attention 中完成", 1300, COLORS["ink"])); sid += 1
    p.append(shape(sid, "select", 1220, 650, 310, 62, "Select Top-k", "FFFFFF", "78B8FF", text_color="4D83D9", size=1500)); sid += 1
    p.append(line(sid, 1000, 211, 1375, 650, COLORS["green_dark"], dash="dash")); sid += 1
    p.append(line(sid, 1375, 712, 1375, 870, COLORS["green_dark"], dash="dash")); sid += 1
    p.append(shape(sid, "selected cache", 1180, 870, 390, 58, "selected compress KV → Attention", COLORS["orange"], COLORS["orange_dark"], size=1350)); sid += 1
    add_footer(p, 3)
    return slide_xml(p)


def attention_slide() -> str:
    p = title_bar("CSA: Sparse MQA Attention", "Window KV 保局部细节，Compress Top-k 保长程信息，sink 提供稳定锚点")
    sid = 20
    p += group_box(sid, 80, 150, 1760, 830, "Attention 输入拼接与输出", COLORS["orange_dark"], COLORS["orange_soft"]); sid += 10
    p.append(shape(sid, "Q", 180, 315, 210, 62, "Q\n[1, heads, head_dim]", COLORS["orange"], COLORS["orange_dark"], size=1200)); sid += 1
    p.append(shape(sid, "window", 560, 230, 280, 78, "KV cache\n(window)", COLORS["red"], COLORS["red_dark"], size=1400)); sid += 1
    p.append(shape(sid, "compress", 560, 420, 280, 78, "KV cache\n(compress Top-k)", COLORS["red"], COLORS["red_dark"], size=1250)); sid += 1
    p.append(shape(sid, "sink", 560, 610, 280, 62, "sink token", COLORS["orange"], COLORS["orange_dark"], size=1450)); sid += 1
    p.append(shape(sid, "concat", 1030, 365, 430, 82, "Concat KV\nwindow + selected compress + sink", COLORS["orange"], COLORS["orange_dark"], size=1350)); sid += 1
    p.append(shape(sid, "attn", 1030, 565, 430, 92, "Attention(MQA)\nsoftmax(QK^T / sqrt(d)) V", COLORS["orange"], COLORS["orange_dark"], size=1350)); sid += 1
    p.append(shape(sid, "output", 1580, 565, 220, 92, "O\n[1, heads, head_dim]", COLORS["orange"], COLORS["orange_dark"], size=1300)); sid += 1
    p.append(line(sid, 390, 346, 1030, 611)); sid += 1
    p.append(line(sid, 840, 269, 1030, 390)); sid += 1
    p.append(line(sid, 840, 459, 1030, 406, COLORS["green_dark"], dash="dash")); sid += 1
    p.append(line(sid, 840, 641, 1030, 423)); sid += 1
    p.append(line(sid, 1245, 447, 1245, 565)); sid += 1
    p.append(line(sid, 1460, 611, 1580, 611)); sid += 1
    p.append(text_box(sid, "window note", 500, 325, 400, 60, "window_size=128\n保证最近上下文完整参与", 1150, COLORS["muted"])); sid += 1
    p.append(text_box(sid, "compress note", 500, 520, 440, 70, "index_topk=1024\n由 Indexer 选出最相关压缩槽位", 1150, COLORS["muted"])); sid += 1
    p.append(text_box(sid, "formula note", 1010, 715, 620, 90, "稀疏来自 K/V 集合的选择；Attention 公式本身仍是标准 QK^T → softmax → V。", 1300, COLORS["ink"])); sid += 1
    add_footer(p, 4)
    return slide_xml(p)


def output_slide() -> str:
    p = title_bar("CSA: 输出投影与 Block 回路", "Attention 输出回到 hidden_size 后，与 hc_pre 残差合并并继续进入 MoE")
    sid = 20
    p += group_box(sid, 80, 150, 1760, 830, "Block 内部回路", COLORS["orange_dark"], COLORS["orange_soft"]); sid += 10
    stages = [
        ("hc_pre", 170, 400, COLORS["blue"], COLORS["blue_dark"]),
        ("RMSNorm", 400, 400, COLORS["blue"], COLORS["blue_dark"]),
        ("CSA", 640, 375, COLORS["orange"], COLORS["orange_dark"]),
        ("RoPE inverse", 920, 330, COLORS["orange"], COLORS["orange_dark"]),
        ("Linear(O down)", 920, 440, COLORS["orange"], COLORS["orange_dark"]),
        ("Linear(O up)", 920, 550, COLORS["orange"], COLORS["orange_dark"]),
        ("hc_post", 1290, 400, COLORS["blue"], COLORS["blue_dark"]),
        ("MoE", 1530, 400, COLORS["blue"], COLORS["blue_dark"]),
    ]
    for txt, x, y, f, s in stages:
        w, h = (220, 64)
        if txt == "CSA":
            w, h = (210, 120)
        p.append(shape(sid, txt, x, y, w, h, txt, f, s, size=1500)); sid += 1
    p.append(line(sid, 390, 432, 400, 432)); sid += 1
    p.append(line(sid, 620, 432, 640, 432)); sid += 1
    p.append(line(sid, 850, 432, 920, 362)); sid += 1
    p.append(line(sid, 1030, 394, 1030, 440)); sid += 1
    p.append(line(sid, 1030, 504, 1030, 550)); sid += 1
    p.append(circle(sid, "add", 1190, 408, 64, "+", COLORS["blue"], COLORS["blue_dark"])); sid += 1
    p.append(line(sid, 1140, 582, 1190, 440)); sid += 1
    p.append(line(sid, 1254, 440, 1290, 432)); sid += 1
    p.append(line(sid, 1510, 432, 1530, 432)); sid += 1
    p.append(line(sid, 280, 464, 280, 705, COLORS["line"], dash="dash")); sid += 1
    p.append(line(sid, 280, 705, 1190, 705, COLORS["line"], dash="dash")); sid += 1
    p.append(line(sid, 1190, 705, 1190, 472, COLORS["line"], dash="dash")); sid += 1
    p.append(text_box(sid, "residual", 325, 725, 620, 34, "残差旁路：hc_pre 与 CSA 输出投影后的 hidden 向量相加", 1200, COLORS["muted"])); sid += 1
    p.append(text_box(sid, "params", 640, 555, 240, 104, "CSA 参数\nheads=128\nhead_dim=512\no_groups=16", 1100, COLORS["muted"])); sid += 1
    p.append(text_box(sid, "note", 920, 725, 720, 90, "O down/up 是低秩输出投影：先降到 o_groups * o_lora_rank，再升回 hidden_size，降低输出投影计算和存储压力。", 1300, COLORS["ink"])); sid += 1
    add_footer(p, 5)
    return slide_xml(p)


def content_types(n: int) -> str:
    overrides = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for i in range(1, n + 1):
        overrides.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + "".join(overrides) + "</Types>"


def rels_root() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '</Relationships>'
    )


def presentation_xml(n: int) -> str:
    slds = "".join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, n + 1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        f'<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{n+1}"/></p:sldMasterIdLst>'
        f'<p:sldIdLst>{slds}</p:sldIdLst>'
        f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        '<p:defaultTextStyle><a:defPPr><a:defRPr lang="zh-CN"/></a:defPPr></p:defaultTextStyle>'
        '</p:presentation>'
    )


def presentation_rels(n: int) -> str:
    rels = []
    for i in range(1, n + 1):
        rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
    rels.append(f'<Relationship Id="rId{n+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
    rels.append(f'<Relationship Id="rId{n+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>"


def slide_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '</Relationships>'
    )


def master_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>'
        '</p:sldMaster>'
    )


def master_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
        '</Relationships>'
    )


def layout_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">'
        '<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    )


def layout_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
        '</Relationships>'
    )


def theme_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="CSA Theme">'
        '<a:themeElements><a:clrScheme name="CSA">'
        '<a:dk1><a:srgbClr val="151515"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="333333"/></a:dk2><a:lt2><a:srgbClr val="F7F7F7"/></a:lt2>'
        '<a:accent1><a:srgbClr val="FF9F1A"/></a:accent1><a:accent2><a:srgbClr val="7B93F4"/></a:accent2>'
        '<a:accent3><a:srgbClr val="91D64B"/></a:accent3><a:accent4><a:srgbClr val="7533AD"/></a:accent4>'
        '<a:accent5><a:srgbClr val="FA1B1B"/></a:accent5><a:accent6><a:srgbClr val="58D0C4"/></a:accent6>'
        '<a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink>'
        '</a:clrScheme><a:fontScheme name="CSA Fonts"><a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface="Microsoft YaHei"/></a:majorFont>'
        '<a:minorFont><a:latin typeface="Aptos"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>'
        '<a:fmtScheme name="CSA Format"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
        '<a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
        '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
        '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>'
        '</a:themeElements></a:theme>'
    )


def core_xml() -> str:
    now = datetime.now(timezone.utc).isoformat()
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>CSA 计算流图</dc:title><dc:creator>Codex</dc:creator>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        '</cp:coreProperties>'
    )


def app_xml(n: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat>'
        f'<Slides>{n}</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides>'
        '</Properties>'
    )


def write_pptx() -> None:
    slides = [overview_slide(), compressor_slide(), indexer_slide(), attention_slide(), output_slide()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", rels_root())
        z.writestr("docProps/core.xml", core_xml())
        z.writestr("docProps/app.xml", app_xml(len(slides)))
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", master_xml())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels())
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout_xml())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels())
        z.writestr("ppt/theme/theme1.xml", theme_xml())
        for i, s in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", s)
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels())


if __name__ == "__main__":
    write_pptx()
    print(OUT)
