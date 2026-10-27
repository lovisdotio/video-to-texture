# 02 · Styled video with the same camera move

Two fal calls (`generate.py`, needs `FAL_KEY` in the environment):

1. **Reference image**: `bytedance/seedream/v5/pro/edit` turns clay frame 1 into a
   photoreal still with the target look (ukiyo-e indigo & gold livery).
   Prompt: `seedream_prompt.txt`.
2. **Video**: `bytedance/seedance-2.5/reference-to-video` takes the clay orbit as
   `@Video1` and the still as `@Image1`, and produces a 30 s, 720p, 24 fps orbit with
   the same camera path. Prompt: `seedance_prompt.txt`. `prompts.md` keeps both prompts
   with their settings.

```bash
python generate.py reference --frame clay_frames/frame_0001.png
python generate.py video --clay clay_orbit_720p24.mp4 --reference out/seedream_0.png
```

Lessons:

- **Regenerate the reference whenever the camera or frame 1 changes.** A stale reference
  wasted a full batch of videos.
- **Seedance's content classifier is not deterministic.** Wording such as
  "recreate this video" or "apply X to ..." can get a request rejected as *video
  editing* (the job completes with an `invalid_request` detail and no video). Phrase
  it as "using @Video1 strictly as reference for the camera and layout, generate ...".
  The prompt here is kept as written at the time; if it is rejected, reword those
  sentences that way.
- Generate a few takes and pick the one whose car stays most consistent; the fit in
  step 03 uses one take only.
