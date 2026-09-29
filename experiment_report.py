"""Write a local, dependency-free report of actual runner outputs."""
import html
import json
from pathlib import Path
from urllib.parse import quote


def save_result(output_folder, results):
    folder = Path(output_folder)
    (folder / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    rows = []
    for result in results:
        figures = []
        for key, label in [('input', 'Processed input'), ('reconstruction', 'Inversion reconstruction'),
                           ('output', 'Optimization output')]:
            src = html.escape(quote(result[key], safe='/'), quote=True)
            figures.append(f'<figure><a href="{src}"><img src="{src}" alt="{label}"></a>'
                           f'<figcaption>{label}</figcaption></figure>')
        title = html.escape(result['source'])
        rows.append(f'<section><h2>{title} · Seed {int(result["seed"])}</h2>' + ''.join(figures) + '</section>')
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Neural Plagiarism — experiment results</title>"
            "<style>body{max-width:1000px;margin:40px auto;padding:0 20px;font:16px system-ui;line-height:1.5}"
            "figure{display:inline-block;vertical-align:top;width:30%;margin:1%}"
            "img{width:100%;height:auto}figcaption{font-size:14px}section{margin-top:32px}"
            "@media(max-width:600px){figure{width:100%;margin:12px 0}}</style>"
            "<h1>Neural Plagiarism</h1><p>Input, inversion reconstruction, and optimization output.</p>"
            "<p>This runner does not compute watermark detection or attack success metrics. "
            "Visual comparisons alone do not establish protection bypass.</p>"
            "<p><a href='config.json'>Configuration</a> · <a href='results.json'>Result manifest</a></p>"
            + ''.join(rows) + '</html>')
    (folder / 'index.html').write_text(page, encoding='utf-8')
