# Presentation video

The 3:17 submission video is built from source, not edited by hand, so any
number that changes in the repository can be corrected by re-running the build
rather than by re-recording a take.

- `presentation_narration.py` — the scene list: every slide's content and its
  spoken narration, side by side. Each factual claim is checkable against the
  repository, and no scene states or implies that a song was generated for the
  video. None was: the foundation model requires a GPU with compute capability
  >= 8.0, and none was available. A demo video that fakes its own result is
  worth less than no video at all.
- `render_presentation_slides.py` — renders each scene to a 1920x1080 PNG in
  the palette the deployed app uses.

## Rebuilding

Screenshots of the live site are captured with headless Chrome:

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SITE="https://frontend-ashen-mu-51.vercel.app"
"$CHROME" --headless --disable-gpu --hide-scrollbars \
    --virtual-time-budget=9000 --window-size=1600,1000 \
    --screenshot=shots/00_home.png "$SITE"
```

Slides, then narration (macOS `say`), then one clip per scene, then concat:

```bash
python presentation/render_presentation_slides.py
say -v Samantha -r 168 -o scene.aiff "<narration>"
ffmpeg -loop 1 -framerate 30 -i slide.png -i scene.wav \
       -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest scene.mp4
ffmpeg -f concat -safe 0 -i concat.txt -c copy joined.mp4
```

Output: 1920x1080 H.264 / AAC, 197.5 s.
