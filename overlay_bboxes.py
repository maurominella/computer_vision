
import json, os
from PIL import Image, ImageDraw

# Usage: python overlay_bboxes.py input.jpg artifacts/llm_triage.json output.png

def draw_overlay(image_path, json_path, output_path):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    with open(json_path, 'r') as f:
        data = json.load(f)
    suspects = data.get('suspects', [])
    for s in suspects:
        x,y,w,h = s['bbox']
        draw.rectangle([(x,y),(x+w,y+h)], outline=(255,0,0), width=3)
        note = f"{s['defect_type']} | {s.get('severity','')} | conf={s.get('confidence',0):.2f}"
        draw.text((x, max(0,y-18)), note, fill=(255,0,0))
    img.save(output_path)

if __name__ == '__main__':
    import sys
    if len(sys.argv)<4:
        print('Usage: python overlay_bboxes.py <image> <json> <output>')
    else:
        draw_overlay(sys.argv[1], sys.argv[2], sys.argv[3])
