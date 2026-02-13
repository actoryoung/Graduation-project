import markdown2
from pathlib import Path

md_file = Path('docs/thesis/文献综述_本科版.md')
with open(md_file, 'r', encoding='utf-8') as f:
    content = f.read()

html = markdown2.markdown(content, extras=['tables', 'fenced-code-blocks', 'header-ids', 'toc', 'footnotes'])

html_with_style = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>多模态情感分析系统设计与实现文献综述</title>
<style>
body {{ font-family: "Times New Roman", "SimSun", serif; font-size: 12pt; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
h1 {{ text-align: center; font-size: 18pt; }}
h2 {{ font-size: 14pt; color: #333; }}
h3 {{ font-size: 12pt; color: #444; }}
p {{ text-align: justify; margin-bottom: 0.8em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; }}
</style>
</head>
<body>{html}
</body>
</html>'''

html_file = md_file.with_suffix('.html')
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_with_style)
print(f'HTML created: {html_file}')
print('You can open this file in a browser and print to PDF')
