import sys
from playwright.sync_api import sync_playwright
url, out = sys.argv[1], sys.argv[2]
click = sys.argv[3] if len(sys.argv) > 3 else None
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 1300, "height": 800})
    logs = []
    pg.on("console", lambda m: logs.append(m.type + ": " + m.text))
    pg.on("pageerror", lambda e: logs.append("PAGEERROR " + str(e)))
    pg.goto(url); pg.wait_for_timeout(4000)
    if click:
        pg.mouse.click(*map(int, click.split(",")))
        pg.wait_for_timeout(800)
    pg.screenshot(path=out)
    print("\n".join(logs[:15]))
    b.close()
