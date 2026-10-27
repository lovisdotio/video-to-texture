#!/bin/zsh
# Run PATINA on every fully padded 2K atlas page (FAL_KEY must be set in the environment).
# Inputs: v12/<page>_2k.png  ->  outputs: v12/<page>_<i>_<map>.png
cd "$(dirname "$0")/v12"
for p in body_hero_8k exterior_opaque running_metal lights_translucent cockpit glass tires; do
  python3 -I ../patina.py ${p}_2k.png ${p} || echo "FAIL $p"
done
