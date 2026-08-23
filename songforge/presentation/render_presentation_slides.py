"""Render every scene of the presentation video to a 1920x1080 PNG."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from presentation_narration import SCENES

W, H = 1920, 1080
BG      = (253, 247, 242)
TEXT    = (59, 46, 42)
ACCENT  = (248, 112, 96)
MUTED   = (138, 122, 114)
CARD    = (255, 255, 255)
LINE    = (233, 220, 212)
OURS    = (46, 138, 106)

FONT_DIRS = ["/System/Library/Fonts/Supplemental/", "/System/Library/Fonts/"]

def font(name, size):
    for d in FONT_DIRS:
        p = Path(d, name)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                pass
    return ImageFont.load_default(size)

def bold(size):    return font("Arial Bold.ttf", size)
def regular(size): return font("Arial.ttf", size)

def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def canvas():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # a quiet accent rule along the top, so slides read as one deck
    d.rectangle([0, 0, W, 8], fill=ACCENT)
    return img, d

def brand(d):
    d.rounded_rectangle([84, H - 96, 128, H - 52], radius=12, fill=ACCENT)
    d.text((138, H - 74), "SongForge", font=bold(26), fill=TEXT, anchor="lm")

def render_title(s):
    img, d = canvas()
    d.rounded_rectangle([W//2 - 46, 300, W//2 + 46, 392], radius=26, fill=ACCENT)
    d.text((W//2, 346), "♪", font=bold(52), fill=(255, 255, 255), anchor="mm")
    d.text((W//2, 500), s["title"], font=bold(132), fill=TEXT, anchor="mm")
    for i, line in enumerate(wrap(d, s["subtitle"], regular(38), 1400)):
        d.text((W//2, 610 + i*56), line, font=regular(38), fill=MUTED, anchor="mm")
    return img

def render_bullets(s):
    img, d = canvas()
    d.text((160, 210), s["title"], font=bold(76), fill=TEXT, anchor="lm")
    y = 360
    for b in s["bullets"]:
        d.ellipse([160, y + 16, 182, y + 38], fill=ACCENT)
        for i, line in enumerate(wrap(d, b, regular(42), 1500)):
            d.text((220, y + 27 + i*58), line, font=regular(42), fill=TEXT, anchor="lm")
        y += 58 * max(1, len(wrap(d, b, regular(42), 1500))) + 54
    if s.get("kicker"):
        d.line([160, y + 20, W - 160, y + 20], fill=LINE, width=2)
        for i, line in enumerate(wrap(d, s["kicker"], regular(40), 1560)):
            d.text((160, y + 90 + i*54), line, font=regular(40), fill=ACCENT, anchor="lm")
    brand(d)
    return img

def render_table(s):
    img, d = canvas()
    d.text((160, 190), s["title"], font=bold(72), fill=TEXT, anchor="lm")
    y = 310
    row_h = 104
    for label, mid, right in s["rows"]:
        d.rounded_rectangle([160, y, W - 160, y + row_h - 16], radius=16, fill=CARD)
        d.text((200, y + (row_h - 16)//2), label, font=regular(36), fill=TEXT, anchor="lm")
        colour = OURS if mid == "OURS" else (ACCENT if mid == "NOT OURS" else TEXT)
        fnt = bold(34) if mid in ("OURS", "NOT OURS") else regular(34)
        d.text((1080, y + (row_h - 16)//2), mid, font=fnt, fill=colour, anchor="mm")
        d.text((W - 200, y + (row_h - 16)//2), right, font=regular(30), fill=MUTED, anchor="rm")
        y += row_h
    brand(d)
    return img

def render_stat(s):
    img, d = canvas()
    d.text((W//2, 400), s["big"], font=bold(220), fill=ACCENT, anchor="mm")
    d.text((W//2, 570), s["big_label"], font=regular(46), fill=TEXT, anchor="mm")
    d.line([W//2 - 400, 650, W//2 + 400, 650], fill=LINE, width=2)
    for i, line in enumerate(wrap(d, s["detail"], regular(36), 1400)):
        d.text((W//2, 720 + i*50), line, font=regular(36), fill=MUTED, anchor="mm")
    brand(d)
    return img

def render_shot(s, shots_dir):
    img, d = canvas()
    shot = Image.open(Path(shots_dir, s["shot"])).convert("RGB")
    # Fit the capture inside a browser-like card, leaving room for a caption.
    target_w = 1560
    scale = target_w / shot.width
    shot = shot.resize((target_w, int(shot.height * scale)), Image.LANCZOS)
    max_h = 760
    if shot.height > max_h:
        shot = shot.crop((0, 0, shot.width, max_h))
    x, y = (W - shot.width)//2, 236
    d.rounded_rectangle([x - 14, y - 58, x + shot.width + 14, y + shot.height + 14],
                        radius=18, fill=CARD)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([x + 12 + i*30, y - 40, x + 30 + i*30, y - 22], fill=c)
    img.paste(shot, (x, y))
    d.text((W//2, 96), s["caption"], font=bold(52), fill=TEXT, anchor="mm")
    return img

def main():
    base = Path(__file__).parent
    out = base / "slides"; out.mkdir(exist_ok=True)
    shots = base / "shots"
    renderers = {"title": render_title, "bullets": render_bullets,
                 "table": render_table, "stat": render_stat}
    for s in SCENES:
        if s["kind"] == "shot":
            img = render_shot(s, shots)
        else:
            img = renderers[s["kind"]](s)
        path = out / f"{s['key']}.png"
        img.save(path)
        print("  ", path.name)
    print(f"{len(SCENES)} slides rendered")

if __name__ == "__main__":
    main()
