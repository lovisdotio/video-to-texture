#!/bin/bash
# Render every frame, encode H.264 1080p30 (BT.709), mux music, delete frames.
# Usage: ./encode.sh [music.mp3] [out.mp4]
#   FROM=0 TO=90 ./encode.sh   -> test a range (frames FROM..TO-1); music is offset to match.
set -euo pipefail
cd "$(dirname "$0")"
MUSIC="${1:-../music/music_120bpm.mp3}"
OUT="${2:-explainer_1080p30.mp4}"
FPS=30
FROM="${FROM:-0}"
FRAMES="frames"

[ -f "$MUSIC" ] || { echo "music not found: $MUSIC" >&2; exit 1; }
rm -rf "$FRAMES"
if [ -n "${TO:-}" ]; then node render.mjs --from "$FROM" --to "$TO" --out "$FRAMES"; else node render.mjs --from "$FROM" --out "$FRAMES"; fi

N=$(ls "$FRAMES"/*.jpg | wc -l | tr -d ' ')
DUR=$(echo "scale=4; $N / $FPS" | bc)
SS=$(echo "scale=4; $FROM / $FPS" | bc)
FADE=$(echo "scale=4; d=$DUR - 1.5; if (d < 0) d=0; d" | bc)
echo "frames=$N duration=${DUR}s music_offset=${SS}s fade_at=${FADE}s"

PRE="atrim=0:$DUR,asetpts=PTS-STARTPTS,afade=t=out:st=$FADE:d=1.5"
# loudness pass 1: measure
M=$(ffmpeg -hide_banner -nostats -ss "$SS" -i "$MUSIC" -af "$PRE,loudnorm=I=-14:TP=-1:LRA=11:print_format=json" -f null - 2>&1 | sed -n '/^{/,/^}/p')
g() { echo "$M" | grep "\"$1\"" | sed -E 's/.*: "([^"]+)".*/\1/'; }
LN="loudnorm=I=-14:TP=-1:LRA=11:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true"

ffmpeg -hide_banner -loglevel warning -y \
  -framerate $FPS -start_number "$FROM" -i "$FRAMES/%05d.jpg" \
  -ss "$SS" -i "$MUSIC" \
  -map 0:v -map 1:a \
  -vf "scale=in_range=full:out_range=tv:out_color_matrix=bt709:flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p,setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709" \
  -c:v libx264 -preset slow -crf 16 -profile:v high -pix_fmt yuv420p \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
  -af "$PRE,$LN,aresample=48000" -c:a aac -b:a 320k -ar 48000 \
  -t "$DUR" -movflags +faststart "$OUT"

# verify before deleting frames
VD=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$OUT")
if [ "$VD" = "$N" ]; then rm -rf "$FRAMES"; echo "OK $OUT ($N frames), frames deleted"; else echo "frame count mismatch ($VD vs $N), frames kept" >&2; exit 1; fi
