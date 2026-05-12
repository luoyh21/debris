"""一次性补丁脚本：把所有 docs_static HTML 侧边栏中的『算法正确性验证』下方
追加一个『STK 跨算法验证』链接。仅用于一次性维护，不参与运行时。"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "api" / "docs_static"
NEEDLE = '<a href="/docs/modules/validation">算法正确性验证</a>'
INSERT = NEEDLE + '\n    <a href="/docs/modules/stk_validation">STK 跨算法验证</a>'

n = 0
for fp in DOCS.rglob("*.html"):
    txt = fp.read_text(encoding="utf-8")
    if NEEDLE in txt and "stk_validation" not in txt:
        fp.write_text(txt.replace(NEEDLE, INSERT, 1), encoding="utf-8")
        print("patched", fp)
        n += 1
print("done", n)
