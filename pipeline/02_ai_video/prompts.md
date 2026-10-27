# AI prompts — Car City pipeline

## Step 1 — Seedream 5 Pro edit (`bytedance/seedream/v5/pro/edit`)
Input: clay render frame 1 (`renders/clay_frames/frame_0001.png`)
image_size: landscape_16_9 (2K), output_format: png

PROMPT:
"Transform this untextured 3D clay render into a hyperrealistic photograph, an extraordinary
cinematic shot with rich detail everywhere, while keeping EXACTLY the same composition,
camera angle, perspective and all object shapes and positions from the input image.
The scene: a Ferrari 458 Spider convertible art car wrapped in an intricate ukiyo-e
Japanese woodblock livery — Hokusai great-wave crests, koi fish and stylized clouds in
deep indigo blue and off-white, with fine gold filigree linework flowing along the body
panels — parked in the middle of a city intersection on dark asphalt, photographed from
a low three-quarter front angle.
Zebra crosswalk stripes on the road, painted lane markings, concrete sidewalks with curbs,
a modern glass-and-concrete office building facade with deep window reveals behind the car,
green leafy street trees, black metal streetlight poles, a park bench, a dark trash can.
Golden late-afternoon sunlight from the left, long soft shadows, crisp reflections on the
car paint, photorealistic PBR materials, 35mm photography, f/8 deep focus, everything
tack sharp, 8k detail. Keep every object exactly where it is in the source image —
same silhouettes, same proportions, same layout."

## Step 2 — Seedance 2.5 reference-to-video (`bytedance/seedance-2.5/reference-to-video`)
Inputs: video_urls=[clay orbit mp4], image_urls=[Seedream result]
duration: "30", resolution: "720p", aspect_ratio: "16:9", generate_audio: false

PROMPT (v2 camera — 2 revolutions, wide 24mm, progressive look-up):
"@Video1 is a 30-second untextured 3D clay render: a wide-angle camera orbits a parked
Ferrari 458 Spider in the middle of a city intersection, starting close on a three-quarter
front view at street level, circling the car over two full revolutions while spiraling
upward and outward, the view gradually rising from the car to the building facades and
then the skyline, ending on a high wide view of the entire crossroads with the car still
visible at the bottom of the frame. @Image1 is the photorealistic reference for this
scene: it defines the exact car with its livery, and the materials, colors and lighting
of everything.
Recreate this exact video photorealistically: one single continuous 30-second shot, no
cuts, no scene transitions, no fades. Follow the camera trajectory, timing and geometry
of @Video1 frame by frame — every building, streetlight, tree, crosswalk stripe, bench
and trash can stays exactly where it is in the clay render — while applying exactly the
visual style, materials, colors and lighting of @Image1 to the entire scene throughout
the whole orbit.
THE CAR: it is the exact same car as in @Image1 in every single frame, with zero identity
drift — a Ferrari 458 Spider convertible, roof open, black two-seat cockpit interior,
silver five-spoke wheels, wrapped in an intricate ukiyo-e Japanese woodblock art livery:
deep indigo blue base, off-white Hokusai great-wave crests breaking across the hood and
doors, detailed koi fish on the front fenders and hood, stylized clouds, and fine gold
filigree linework flowing along the body lines. This identical livery pattern wraps
continuously and coherently around the entire body — hood, doors, fenders, rear deck —
and stays perfectly consistent from every camera angle for the full 30 seconds: same
colors, same wave positions, same koi placement, same gold lines in every frame. The car
never changes, never morphs, is never replaced, and stays perfectly still with its wheels
motionless.
THE SCENE: glossy clear-coat car paint, dark asphalt, white road markings, concrete-and-
glass buildings with deep window reveals, leafy green trees, golden late-afternoon
sunlight with long soft shadows, consistent sun direction and consistent materials from
every camera angle, photoreal 35mm cinematography, no people, no moving vehicles, smooth
continuous camera motion."
