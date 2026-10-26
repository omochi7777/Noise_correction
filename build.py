#!/usr/bin/env python3
"""src/ の部品を1枚の index.html にまとめます。  使い方: python3 build.py"""
import base64, pathlib

root = pathlib.Path(__file__).parent
src = root / "src"

page = (src / "template.html").read_text(encoding="utf-8")
core = (src / "core.js").read_text(encoding="utf-8")
sample = base64.b64encode((src / "sample.webp").read_bytes()).decode()

page = page.replace("__CORE__", core).replace("__SAMPLE__", "data:image/webp;base64," + sample)

doc = (
    "<!doctype html>\n<html lang=\"ja\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">\n"
    "<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
    "body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
    "</head>\n<body>\n" + page + "\n</body>\n</html>\n"
)
(root / "index.html").write_text(doc, encoding="utf-8")
print(f"index.html を書き出しました（{len(doc.encode()) // 1024} KB）")
