"""High-quality v2 differentiable video-to-UV fitting on a fal H100.

The input is the lossless archive produced by ``prepare_dataset_hq.py`` (preferably
the ``*_masked.zip`` variant).  The mesh and cameras stay fixed: only a static RGB
atlas is optimized.  Unlike v1, this entrypoint initializes fine detail directly
from 1024 px renders into an 8K atlas instead of merely enlarging a 2K solution.

No work is performed when this module is imported.  The ``__main__`` block reads
configuration from environment variables and invokes the remote function.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import fal
from fal.container import ContainerImage
from fal.toolkit import File


# The nvdiffrast wheel is prebuilt for this exact Torch/CUDA combination.  fal's
# worker sandbox cannot compile CUDA extensions, so do not replace this with a
# git install or a source distribution.
DOCKERFILE = r"""
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    numpy==1.26.4 pillow==10.4.0 requests==2.32.3 pydantic==2.13.4
RUN pip install --no-cache-dir \
    --extra-index-url https://miropsota.github.io/torch_packages_builder \
    "nvdiffrast==0.4.0+253ac4fpt2.4.1cu121" \
    && python -c "import torch, nvdiffrast.torch as dr; assert torch.version.cuda == '12.1'; print('NVDIFFRAST_PREBUILT_OK')"
"""


IMAGE = ContainerImage.from_dockerfile_str(DOCKERFILE)
SCHEMA_PREFIX = "car-city-video-fit-dataset-v2-hq"

# ``fal run`` imports the module and invokes the referenced function with no
# command-line arguments.  Capture local environment overrides as function
# defaults at import time; explicit programmatic arguments still take priority.
DEFAULT_DATASET_URL = os.environ.get("DATASET_URL", "")
if not DEFAULT_DATASET_URL:
    try:
        DEFAULT_DATASET_URL = Path(__file__).with_name("dataset_url_hq.txt").read_text().strip()
    except OSError:
        pass
DEFAULT_COARSE_ITERATIONS = int(os.environ.get("FIT_COARSE_ITERS", "500"))
DEFAULT_FINE_ITERATIONS = int(os.environ.get("FIT_FINE_ITERS", "1200"))
DEFAULT_COARSE_TEXTURE = int(os.environ.get("FIT_COARSE_TEXTURE", "4096"))
DEFAULT_FINE_TEXTURE = int(os.environ.get("FIT_FINE_TEXTURE", "8192"))
DEFAULT_COARSE_RENDER = int(os.environ.get("FIT_COARSE_RENDER", "512"))
DEFAULT_FINE_RENDER = int(os.environ.get("FIT_FINE_RENDER", "1024"))
DEFAULT_BATCH_SIZE = int(os.environ.get("FIT_BATCH_SIZE", "3"))
DEFAULT_SEED = int(os.environ.get("FIT_SEED", "63"))
DEFAULT_MIN_MEDIAN_IOU = float(os.environ.get("FIT_MIN_MEDIAN_IOU", "0.55"))


def _log(message: str) -> None:
    print(f"[video-fit-hq] {message}", flush=True)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a bounded ordinary ZIP without path traversal or symlinks."""
    import stat
    import zipfile

    max_total_bytes = 6 * 1024**3
    with zipfile.ZipFile(archive, "r") as handle:
        infos = handle.infolist()
        total = sum(item.file_size for item in infos)
        if total > max_total_bytes:
            raise ValueError(f"Dataset expands to {total} bytes; limit is {max_total_bytes}")
        root = destination.resolve()
        for item in infos:
            candidate = (destination / item.filename).resolve()
            if candidate != root and root not in candidate.parents:
                raise ValueError(f"Unsafe archive member: {item.filename!r}")
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive symlinks are not accepted: {item.filename!r}")
        handle.extractall(destination)


def _download_dataset(url: str, destination: Path) -> None:
    import shutil
    import time

    if url.startswith("file://"):
        source = Path(url[7:])
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, destination)
        return
    local = Path(url).expanduser()
    if "://" not in url and local.is_file():
        shutil.copyfile(local, destination)
        return
    if not url.startswith(("http://", "https://")):
        raise ValueError("dataset_url must be an HTTP(S) URL or an existing local path")

    import requests

    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with destination.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if destination.stat().st_size < 1024:
                raise ValueError("Downloaded dataset is unexpectedly small")
            return
        except BaseException as exc:  # preserve the useful final HTTP/network error
            last_error = exc
            if destination.exists():
                destination.unlink()
            _log(f"dataset download attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download dataset after 3 attempts: {last_error}")


def _c2w_to_w2c(torch, c2w):
    """Rigid Blender/OpenGL camera inverse (same convention as UniTEX)."""
    w2c = torch.zeros_like(c2w)
    rotation_t = c2w[..., :3, :3].transpose(-1, -2)
    w2c[..., :3, :3] = rotation_t
    w2c[..., :3, 3:] = -rotation_t @ c2w[..., :3, 3:]
    w2c[..., 3, 3] = 1.0
    return w2c


def _intr_to_proj_opengl_topdown(torch, intrinsics, width: int, height: int):
    """OpenGL/-Z projection whose raster tensor rows follow a top-down image.

    This is deliberately not UniTEX's ``intr_to_proj`` verbatim.  Its X principal
    point sign is wrong for off-centre crops (the old six-view cameras hid that
    because cx was exactly 0.5).  Here clip_w=-camera_z and both principal point
    terms are derived directly from K.
    """
    near, far = 0.01, 1000.0
    projection = torch.zeros(
        (*intrinsics.shape[:-2], 4, 4),
        dtype=intrinsics.dtype,
        device=intrinsics.device,
    )
    projection[..., 0, 0] = 2.0 * intrinsics[..., 0, 0] / float(width)
    # K uses the ordinary image convention where integer coordinates denote
    # pixel centres, while raster NDC maps the centre of pixel zero to
    # 2*(0.5/W)-1.  The +0.5 terms preserve that half-pixel convention.
    projection[..., 0, 2] = 1.0 - 2.0 * (intrinsics[..., 0, 2] + 0.5) / float(width)
    # nvdiffrast row zero corresponds to NDC y=-1.  Negating fy therefore makes
    # its output memory layout line up with ordinary top-down JPEG/PNG tensors.
    projection[..., 1, 1] = -2.0 * intrinsics[..., 1, 1] / float(height)
    projection[..., 1, 2] = 1.0 - 2.0 * (intrinsics[..., 1, 2] + 0.5) / float(height)
    projection[..., 2, 2] = -(far + near) / (far - near)
    projection[..., 2, 3] = -2.0 * far * near / (far - near)
    projection[..., 3, 2] = -1.0
    return projection


def _build_mvp(torch, c2w, intrinsics, calibration_width: int, calibration_height: int):
    w2c = _c2w_to_w2c(torch, c2w)
    projection = _intr_to_proj_opengl_topdown(
        torch, intrinsics, calibration_width, calibration_height
    )
    return projection @ w2c, w2c


def _validate_projection(torch, vertices, mvp, projection_crop, width: int, height: int) -> float:
    """Cross-check OpenGL clip pixels against K_crop @ w2c_opencv."""
    # A deterministic spread of vertices is enough to catch every sign/axis bug,
    # without materializing N*V reference projections.
    vertex_ids = torch.linspace(
        0, vertices.shape[0] - 1, steps=min(4096, vertices.shape[0]), device=vertices.device
    ).round().long()
    view_ids = torch.linspace(
        0, mvp.shape[0] - 1, steps=min(12, mvp.shape[0]), device=vertices.device
    ).round().long()
    selected = vertices[vertex_ids]
    homo = torch.cat([selected, torch.ones_like(selected[:, :1])], dim=-1)
    clip = homo.unsqueeze(0) @ mvp[view_ids].transpose(1, 2)
    ndc = clip[..., :2] / clip[..., 3:4]
    pixels = (ndc + 1.0) * torch.tensor(
        [width * 0.5, height * 0.5], dtype=vertices.dtype, device=vertices.device
    ) - 0.5

    ref_h = homo.unsqueeze(0) @ projection_crop[view_ids].transpose(1, 2)
    reference = ref_h[..., :2] / ref_h[..., 2:3]
    finite = torch.isfinite(pixels).all(dim=-1) & torch.isfinite(reference).all(dim=-1)
    if not bool(finite.any()):
        raise ValueError("Projection validation produced no finite points")
    return float((pixels[finite] - reference[finite]).abs().max().item())


def _build_render_cache(
    torch,
    dr,
    ctx,
    vertices,
    faces,
    uv_coords,
    uv_faces,
    mvp,
    resolution,
    batch_size,
    topology_hash,
):
    """Rasterize fixed geometry once; texture iterations only resample the atlas."""
    homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
    clips, rasts, uvs, uv_das, alphas = [], [], [], [], []
    for start in range(0, mvp.shape[0], batch_size):
        stop = min(start + batch_size, mvp.shape[0])
        clip = homo.unsqueeze(0) @ mvp[start:stop].transpose(1, 2)
        rast, rast_db = dr.rasterize(
            ctx, clip.contiguous(), faces, [resolution, resolution], grad_db=True
        )
        # Geometry uses the shared original vertex topology.  UV interpolation
        # may use a different index buffer because triangle ordering is identical.
        uv, uv_da = dr.interpolate(
            uv_coords.unsqueeze(0),
            rast,
            uv_faces,
            rast_db=rast_db,
            diff_attrs="all",
        )
        hard = (rast[..., 3:] > 0).float()
        alpha = dr.antialias(
            hard, rast, clip.contiguous(), faces, topology_hash=topology_hash
        )
        clips.append(clip.detach())
        rasts.append(rast.detach())
        uvs.append(uv.detach())
        uv_das.append(uv_da.detach())
        alphas.append(alpha.detach())
    cache = {
        "resolution": int(resolution),
        "clip": torch.cat(clips, dim=0),
        "rast": torch.cat(rasts, dim=0),
        "uv": torch.cat(uvs, dim=0),
        "uv_da": torch.cat(uv_das, dim=0),
        "alpha": torch.cat(alphas, dim=0),
        "topology_hash": topology_hash,
    }
    _log(
        f"cached {mvp.shape[0]} fixed renders at {resolution}px "
        f"({sum(value.numel() * value.element_size() for key, value in cache.items() if hasattr(value, 'numel')) / 1024**3:.2f} GiB)"
    )
    return cache


def _render_texture(torch, dr, cache, frame_ids, texture, faces, max_mip_level: int = 3):
    ids = torch.as_tensor(frame_ids, dtype=torch.long, device=texture.device)
    rast = cache["rast"].index_select(0, ids).contiguous()
    clip = cache["clip"].index_select(0, ids).contiguous()
    uv = cache["uv"].index_select(0, ids).contiguous()
    uv_da = cache["uv_da"].index_select(0, ids).contiguous()
    # The official API broadcasts a batch-one texture across the UV minibatch;
    # keeping it shared avoids copying 2.25 GiB per 8K/B=3 fine iteration.
    rgb = dr.texture(
        texture.contiguous(),
        uv,
        uv_da=uv_da,
        filter_mode="linear-mipmap-linear",
        boundary_mode="clamp",
        max_mip_level=max_mip_level,
    )
    hard = (rast[..., 3:] > 0.0).to(rgb.dtype)
    foreground = dr.antialias(
        (rgb * hard).contiguous(),
        rast,
        clip,
        faces,
        topology_hash=cache["topology_hash"],
    )
    alpha = cache["alpha"].index_select(0, ids)
    return foreground, alpha, hard


def _resize_targets(torch, F, frames_cpu, masks_cpu, frame_ids, resolution, device):
    frame_ids_cpu = torch.as_tensor(frame_ids, dtype=torch.long)
    frames = frames_cpu.index_select(0, frame_ids_cpu).to(device=device, dtype=torch.float32)
    frames = frames.permute(0, 3, 1, 2).div_(255.0)
    if frames.shape[-1] != resolution:
        frames = F.interpolate(frames, (resolution, resolution), mode="bilinear", align_corners=False)
    frames = frames.permute(0, 2, 3, 1).contiguous()
    if masks_cpu is None:
        return frames, None
    masks = masks_cpu.index_select(0, frame_ids_cpu).to(device=device, dtype=torch.float32)
    masks = masks.unsqueeze(1).div_(255.0)
    if masks.shape[-1] != resolution:
        masks = F.interpolate(masks, (resolution, resolution), mode="bilinear", align_corners=False)
    return frames, masks.permute(0, 2, 3, 1).contiguous()


def _erode_mask(torch, F, mask_nhwc, radius: int):
    if radius <= 0:
        return mask_nhwc
    kernel = radius * 2 + 1
    mask = mask_nhwc.permute(0, 3, 1, 2)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel, stride=1, padding=radius)
    return eroded.permute(0, 2, 3, 1)


def _alignment_check(torch, F, dr, cache, frames_cpu, masks_cpu, faces, batch_size):
    resolution = cache["resolution"]
    n = cache["rast"].shape[0]
    ious = []
    rendered_masks = []
    for start in range(0, n, batch_size):
        ids = list(range(start, min(start + batch_size, n)))
        rast = cache["rast"][start : start + len(ids)].contiguous()
        hard = (rast[..., 3:] > 0).float()
        alpha = cache["alpha"][start : start + len(ids)]
        _, target = _resize_targets(
            torch, F, frames_cpu, masks_cpu, ids, resolution, rast.device
        )
        if target is None:
            target_binary = hard > 0.5
        else:
            target_binary = target > 0.5
        rendered_binary = alpha > 0.5
        intersection = (rendered_binary & target_binary).flatten(1).sum(dim=1).float()
        union = (rendered_binary | target_binary).flatten(1).sum(dim=1).float()
        batch_iou = (intersection + 1.0) / (union + 1.0)
        ious.extend(float(value) for value in batch_iou.cpu())
        rendered_masks.append((rendered_binary[..., 0].cpu().to(torch.uint8) * 255))
    return ious, torch.cat(rendered_masks, dim=0)


def _uv_occupancy(torch, dr, ctx, uv_coords, uv_faces, resolution):
    clip = torch.cat(
        [uv_coords * 2.0 - 1.0, torch.zeros_like(uv_coords[:, :1]), torch.ones_like(uv_coords[:, :1])],
        dim=-1,
    )
    rast, _ = dr.rasterize(ctx, clip.unsqueeze(0).contiguous(), uv_faces, [resolution, resolution])
    occupancy = (rast[..., 3:] > 0).detach()
    del rast
    return occupancy


def _pull_push_fill(torch, F, color_nhwc, known_nhwc, occupancy_nhwc):
    """Fill sparse reprojection holes with a multiscale normalized pull-push."""
    color = color_nhwc.permute(0, 3, 1, 2)
    mask = known_nhwc.permute(0, 3, 1, 2).float()
    weighted_global = (color * mask).sum(dim=(0, 2, 3), keepdim=True) / mask.sum().clamp_min(1.0)
    colors = [color]
    masks = [mask]
    while min(colors[-1].shape[-2:]) > 2:
        previous_color, previous_mask = colors[-1], masks[-1]
        pooled_mask = F.avg_pool2d(previous_mask, 2, 2)
        pooled_color = F.avg_pool2d(previous_color * previous_mask, 2, 2) / pooled_mask.clamp_min(1e-8)
        pooled_known = (pooled_mask > 0).float()
        colors.append(torch.where(pooled_known.bool(), pooled_color, torch.zeros_like(pooled_color)))
        masks.append(pooled_known)
    filled = torch.where(masks[-1].bool(), colors[-1], weighted_global.expand_as(colors[-1]))
    for level in range(len(colors) - 2, -1, -1):
        up = F.interpolate(filled, size=colors[level].shape[-2:], mode="bilinear", align_corners=False)
        filled = torch.where(masks[level].bool(), colors[level], up)
    filled = filled.permute(0, 2, 3, 1)
    neutral = torch.full_like(filled, 0.5)
    return torch.where(occupancy_nhwc, filled, neutral)


def _pad_atlas(torch, F, texture_nhwc, occupancy_nhwc, padding: int):
    """Outpaint chart gutters without changing chart-interior texels."""
    color = texture_nhwc.permute(0, 3, 1, 2)
    mask = occupancy_nhwc.permute(0, 3, 1, 2).float()
    for _ in range(max(0, padding)):
        weight = F.avg_pool2d(mask, 3, 1, 1)
        candidate = F.avg_pool2d(color * mask, 3, 1, 1) / weight.clamp_min(1e-8)
        new = (mask < 0.5) & (weight > 0)
        color = torch.where(new.expand_as(color), candidate, color)
        mask = torch.where(new, torch.ones_like(mask), mask)
    return color.permute(0, 2, 3, 1).contiguous(), mask.permute(0, 2, 3, 1) > 0.5


def _reprojection_init(
    torch,
    F,
    dr,
    cache,
    frames_cpu,
    masks_cpu,
    frame_ids,
    vertices,
    vertex_normals,
    faces,
    c2w,
    sharpness_weights,
    occupancy,
    texture_size,
    batch_size,
    padding,
):
    """Robust two-pass image-space splat into UV texels.

    The first pass is a weighted mean.  The second is an IRLS clipped mean around
    that estimate, a memory-bounded approximation to a per-texel trimmed median.
    """
    device = vertices.device
    total_texels = texture_size * texture_size

    def splat(reference=None):
        accum = torch.zeros((total_texels, 3), dtype=torch.float32, device=device)
        weights = torch.zeros((total_texels, 1), dtype=torch.float32, device=device)
        counts = torch.zeros((total_texels, 1), dtype=torch.float32, device=device)
        for offset in range(0, len(frame_ids), batch_size):
            ids = frame_ids[offset : offset + batch_size]
            ids_gpu = torch.as_tensor(ids, dtype=torch.long, device=device)
            target, target_mask = _resize_targets(
                torch, F, frames_cpu, masks_cpu, ids, cache["resolution"], device
            )
            rast = cache["rast"].index_select(0, ids_gpu)
            uv = cache["uv"].index_select(0, ids_gpu)
            hard = rast[..., 3:] > 0
            if target_mask is None:
                target_mask = hard.float()
            else:
                target_mask = (target_mask > 0.5).float()

            camera_position = c2w.index_select(0, ids_gpu)[:, :3, 3]
            view_direction = F.normalize(camera_position[:, None, :] - vertices[None, :, :], dim=-1)
            facing_vertex = (view_direction * vertex_normals[None, :, :]).sum(dim=-1, keepdim=True)
            facing, _ = dr.interpolate(facing_vertex.contiguous(), rast, faces)
            facing = facing.clamp(0.0, 1.0).square().mul(0.85).add(0.15)
            quality = sharpness_weights.index_select(0, ids_gpu).view(-1, 1, 1, 1)
            sample_weight = target_mask * facing * quality

            u = uv[..., 0]
            v = uv[..., 1]
            valid = hard[..., 0] & (target_mask[..., 0] > 0.5)
            valid &= torch.isfinite(u) & torch.isfinite(v)
            valid &= (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
            x = (u * texture_size).long().clamp(0, texture_size - 1)
            y = (v * texture_size).long().clamp(0, texture_size - 1)
            atlas_index = (y * texture_size + x)[valid]
            color = target[valid]
            # Boolean indexing already flattens the three image dimensions.
            # Add the channel axis afterwards; using ``[valid, None]`` makes
            # PyTorch interpret ``None`` as a second advanced index.
            weight = sample_weight[..., 0][valid].unsqueeze(-1)
            if reference is not None:
                delta = (color - reference.index_select(0, atlas_index)).abs().mean(dim=-1, keepdim=True)
                # Redescending-enough for moving highlights/background leaks while
                # retaining a gradient contribution from moderately inconsistent views.
                robust = (0.12 / delta.clamp_min(0.12)).clamp(max=1.0)
                robust = torch.where(delta > 0.35, robust * 0.1, robust)
                weight = weight * robust
            accum.scatter_add_(0, atlas_index[:, None].expand(-1, 3), color * weight)
            weights.scatter_add_(0, atlas_index[:, None], weight)
            counts.scatter_add_(0, atlas_index[:, None], torch.ones_like(weight))
        return accum, weights, counts

    with torch.no_grad():
        first_color, first_weight, first_count = splat()
        first_mean = first_color / first_weight.clamp_min(1e-8)
        # Only the first-pass mean and unweighted support are needed below.
        # Releasing these two accumulators before the robust pass removes
        # 1.00 GiB of live 8K float32 storage at production resolution.
        del first_color, first_weight
        color_sum, weight_sum, _ = splat(first_mean)
        del first_mean
        direct = color_sum / weight_sum.clamp_min(1e-8)
        del color_sum
        positive = weight_sum[:, 0] > 1e-7
        known = positive.view(1, texture_size, texture_size, 1) & occupancy
        direct = direct.view(1, texture_size, texture_size, 3)
        filled = _pull_push_fill(torch, F, direct, known, occupancy)

        support = first_count.view(1, texture_size, texture_size, 1)
        support_values = support[support > 0]
        support_scale = torch.quantile(support_values, 0.95).clamp_min(1.0) if support_values.numel() else torch.tensor(1.0, device=device)
        support_image = torch.log1p(support) / torch.log1p(support_scale)
        support_image = support_image.clamp(0.0, 1.0) * occupancy.float()

        confidence_raw = weight_sum.view(1, texture_size, texture_size, 1)
        confidence_values = confidence_raw[confidence_raw > 0]
        confidence_scale = (
            torch.quantile(confidence_values, 0.75).clamp_min(1e-6)
            if confidence_values.numel()
            else torch.tensor(1.0, device=device)
        )
        confidence = (1.0 - torch.exp(-confidence_raw / confidence_scale)) * occupancy.float()
        padded, _ = _pad_atlas(torch, F, filled, occupancy, padding)
        coverage = float((known & occupancy).sum().item() / occupancy.sum().clamp_min(1).item())
    _log(f"robust reprojection initialized {coverage:.1%} of occupied {texture_size}px texels directly")
    return padded, support_image, confidence, coverage


def _gaussian_window(torch, device, dtype, size=11, sigma=1.5):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) * 0.5
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, size, size).repeat(3, 1, 1, 1)


def _masked_dssim(torch, F, prediction, target, mask, window, frame_weights):
    x = prediction.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    y = target.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=3)
    mu_y = F.conv2d(y, window, padding=pad, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ).clamp_min(1e-8)
    dssim = ((1.0 - ssim) * 0.5).mean(dim=1, keepdim=True)
    weights = mask.permute(0, 3, 1, 2)
    mask_sum = weights.flatten(1).sum(dim=1)
    per_frame = (dssim * weights).flatten(1).sum(dim=1) / mask_sum.clamp_min(1.0)
    valid_weights = frame_weights * (mask_sum > 0).to(frame_weights.dtype)
    if not bool((valid_weights > 0).any()):
        return dssim.sum() * 0.0
    return (per_frame * valid_weights).sum() / valid_weights.sum().clamp_min(1e-6)


def _frame_batches(np, frame_ids, batch_size, seed):
    """Yield temporally stratified batches instead of clustered random views.

    Seedance's camera distance and visible side evolve continuously.  Pure random
    batches occasionally contain near-duplicate views; one draw from each temporal
    stratum gives every gradient step broad surface coverage while remaining
    deterministic.  Sharpness still controls each frame's loss weight.
    """
    rng = np.random.default_rng(seed)
    ordered = np.sort(np.asarray(frame_ids, dtype=np.int64))
    strata = [chunk.copy() for chunk in np.array_split(ordered, min(batch_size, len(ordered)))]
    cursors = [len(chunk) for chunk in strata]
    while True:
        batch = []
        for stratum_index, chunk in enumerate(strata):
            if cursors[stratum_index] >= len(chunk):
                rng.shuffle(chunk)
                cursors[stratum_index] = 0
            batch.append(int(chunk[cursors[stratum_index]]))
            cursors[stratum_index] += 1
        if len(batch) < batch_size:
            batch += rng.choice(ordered, size=batch_size - len(batch), replace=True).tolist()
        rng.shuffle(batch)
        yield batch


def _optimize_stage(
    np,
    torch,
    F,
    dr,
    cache,
    frames_cpu,
    masks_cpu,
    faces,
    initial_texture,
    confidence,
    sharpness_weights,
    train_ids,
    iterations,
    learning_rate,
    batch_size,
    seed,
    stage_name,
    max_mip_level,
):
    if iterations <= 0:
        return initial_texture.detach(), []
    logits = torch.nn.Parameter(
        torch.logit(initial_texture.clamp(1.0 / 255.0, 254.0 / 255.0))
    )
    optimizer = torch.optim.Adam([logits], lr=learning_rate, betas=(0.9, 0.99))
    sampler = _frame_batches(np, train_ids, batch_size, seed)
    window = _gaussian_window(torch, logits.device, logits.dtype)
    initial_flat = initial_texture.detach().reshape(-1, 3)
    confidence_flat = confidence.detach().reshape(-1)
    history = []
    start_time = __import__("time").perf_counter()
    for iteration in range(iterations):
        ids = next(sampler)
        texture = torch.sigmoid(logits)
        prediction, alpha, hard = _render_texture(
            torch, dr, cache, ids, texture, faces, max_mip_level=max_mip_level
        )
        target, target_mask = _resize_targets(
            torch, F, frames_cpu, masks_cpu, ids, cache["resolution"], logits.device
        )
        if target_mask is None:
            target_mask = hard.detach()
        safe_target = _erode_mask(torch, F, (target_mask > 0.5).float(), radius=2)
        common = safe_target * (alpha.detach() > 0.95).float()
        error = torch.sqrt((prediction - target).square().mean(dim=-1, keepdim=True) + 1e-6)
        # Detached IRLS weights make Charbonnier resistant to Seedance flicker,
        # moving highlights and small mask leaks without an expensive pixel sort.
        robust_weight = (0.15 / error.detach().clamp_min(0.15)).clamp(max=1.0)
        weighted_mask = common * robust_weight
        rgb_mask_sum = weighted_mask.flatten(1).sum(dim=1)
        per_frame_rgb = (error * weighted_mask).flatten(1).sum(dim=1) / rgb_mask_sum.clamp_min(1.0)
        quality = sharpness_weights[torch.as_tensor(ids, device=logits.device)]
        valid_quality = quality * (rgb_mask_sum > 0).to(quality.dtype)
        rgb_loss = (per_frame_rgb * valid_quality).sum() / valid_quality.sum().clamp_min(1e-6)

        ssim_mask = _erode_mask(torch, F, common, radius=5) * robust_weight
        dssim_loss = _masked_dssim(
            torch, F, prediction, target, ssim_mask, window, quality
        )

        sample_count = min(65_536, initial_flat.shape[0])
        prior_ids = torch.randint(0, initial_flat.shape[0], (sample_count,), device=logits.device)
        prior_confidence = confidence_flat.index_select(0, prior_ids)
        current_sample = texture.reshape(-1, 3).index_select(0, prior_ids)
        prior_error = torch.sqrt(
            (current_sample - initial_flat.index_select(0, prior_ids)).square().mean(dim=-1) + 1e-6
        )
        prior_loss = (prior_error * prior_confidence).sum() / prior_confidence.sum().clamp_min(1.0)
        loss = rgb_loss + 0.10 * dssim_loss + 0.01 * prior_loss

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{stage_name} produced non-finite loss at iteration {iteration}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], max_norm=10.0)
        optimizer.step()

        if iteration == 0 or (iteration + 1) % 25 == 0 or iteration + 1 == iterations:
            record = {
                "iteration": iteration + 1,
                "loss": float(loss.detach()),
                "rgb": float(rgb_loss.detach()),
                "dssim": float(dssim_loss.detach()),
                "prior": float(prior_loss.detach()),
                "elapsed_seconds": __import__("time").perf_counter() - start_time,
            }
            history.append(record)
            _log(
                f"{stage_name} {iteration + 1:04d}/{iterations}: "
                f"loss={record['loss']:.5f} rgb={record['rgb']:.5f} dssim={record['dssim']:.5f}"
            )
    return torch.sigmoid(logits.detach()), history


def _evaluate_residuals(
    torch,
    F,
    dr,
    cache,
    frames_cpu,
    masks_cpu,
    faces,
    texture,
    batch_size,
    max_mip_level,
):
    residuals = []
    n = cache["rast"].shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            ids = list(range(start, min(start + batch_size, n)))
            prediction, alpha, hard = _render_texture(
                torch, dr, cache, ids, texture, faces, max_mip_level=max_mip_level
            )
            target, target_mask = _resize_targets(
                torch, F, frames_cpu, masks_cpu, ids, cache["resolution"], texture.device
            )
            if target_mask is None:
                target_mask = hard
            common = _erode_mask(torch, F, (target_mask > 0.5).float(), 2)
            common = common * (alpha > 0.95).float()
            error = (prediction - target).abs().mean(dim=-1, keepdim=True)
            values = (error * common).flatten(1).sum(dim=1) / common.flatten(1).sum(dim=1).clamp_min(1.0)
            residuals.extend(float(value) for value in values.cpu())
    return residuals


def _save_rgb(PIL_Image, torch, tensor, path: Path, flip_vertical: bool = True):
    array = tensor.detach().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if flip_vertical:
        array = array[::-1].copy()
    PIL_Image.fromarray(array, mode="RGB").save(path, optimize=True)


def _save_gray(PIL_Image, torch, tensor, path: Path, flip_vertical: bool = True):
    array = tensor.detach().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    array = array.squeeze()
    if flip_vertical:
        array = array[::-1].copy()
    PIL_Image.fromarray(array, mode="L").save(path, optimize=True)


def _alignment_sheet(PIL_Image, ImageDraw, ImageFilter, frames_cpu, masks_cpu, rendered_masks, ious, path):
    count = len(ious)
    chosen = [round(value) for value in __import__("numpy").linspace(0, count - 1, 8)]
    tile_size, label_height, columns = 256, 22, 4
    sheet = PIL_Image.new("RGB", (columns * tile_size, 2 * (tile_size + label_height)), "black")
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(chosen):
        frame = PIL_Image.fromarray(frames_cpu[index].numpy(), mode="RGB").resize((tile_size, tile_size))
        rendered = PIL_Image.fromarray(rendered_masks[index].numpy(), mode="L").resize(
            (tile_size, tile_size), PIL_Image.Resampling.NEAREST
        )
        if masks_cpu is None:
            target = rendered
        else:
            target = PIL_Image.fromarray(masks_cpu[index].numpy(), mode="L").resize(
                (tile_size, tile_size), PIL_Image.Resampling.NEAREST
            )
        target_edge = target.filter(ImageFilter.MaxFilter(5))
        target_inner = target.filter(ImageFilter.MinFilter(5))
        rendered_edge = rendered.filter(ImageFilter.MaxFilter(5))
        rendered_inner = rendered.filter(ImageFilter.MinFilter(5))
        import numpy as np

        rgb = np.asarray(frame).copy()
        red = np.asarray(target_edge, dtype=np.int16) - np.asarray(target_inner, dtype=np.int16)
        green = np.asarray(rendered_edge, dtype=np.int16) - np.asarray(rendered_inner, dtype=np.int16)
        rgb[red > 0] = [255, 40, 40]
        rgb[green > 0] = [30, 255, 70]
        x = (slot % columns) * tile_size
        y = (slot // columns) * (tile_size + label_height) + label_height
        sheet.paste(PIL_Image.fromarray(rgb, mode="RGB"), (x, y))
        draw.text((x + 4, y - label_height + 4), f"frame {index:03d}  IoU {ious[index]:.3f}", fill="white")
    sheet.save(path, optimize=True)


def _preview_sheet(PIL_Image, ImageDraw, frames_cpu, predictions, frame_ids, path):
    thumb, label_height, columns = 256, 22, 2
    pair_width = thumb * 2
    rows = (len(frame_ids) + columns - 1) // columns
    sheet = PIL_Image.new("RGB", (columns * pair_width, rows * (thumb + label_height)), "black")
    draw = ImageDraw.Draw(sheet)
    for slot, (index, prediction) in enumerate(zip(frame_ids, predictions)):
        target = PIL_Image.fromarray(frames_cpu[index].numpy(), mode="RGB").resize((thumb, thumb))
        rendered = PIL_Image.fromarray(prediction, mode="RGB").resize((thumb, thumb))
        x = (slot % columns) * pair_width
        y = (slot // columns) * (thumb + label_height) + label_height
        sheet.paste(target, (x, y))
        sheet.paste(rendered, (x + thumb, y))
        draw.text((x + 4, y - label_height + 4), f"frame {index:03d}: target | fitted", fill="white")
    sheet.save(path, optimize=True)


def _export_obj(np, output_dir, vertices, normals, faces, uv_coords, uv_faces, texture_name):
    obj_path = output_dir / "ferrari_video_fit_hq.obj"
    mtl_path = output_dir / "ferrari_video_fit_hq.mtl"
    with obj_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
        output.write(
            "# car-city high-quality differentiable video texture fit\n"
            "mtllib ferrari_video_fit_hq.mtl\no ferrari_video_fit_hq\n"
        )
        for start in range(0, len(vertices), 20_000):
            output.write("".join(f"v {x:.8g} {y:.8g} {z:.8g}\n" for x, y, z in vertices[start : start + 20_000]))
        for start in range(0, len(uv_coords), 20_000):
            output.write("".join(f"vt {u:.8g} {v:.8g}\n" for u, v in uv_coords[start : start + 20_000]))
        for start in range(0, len(normals), 20_000):
            output.write("".join(f"vn {x:.8g} {y:.8g} {z:.8g}\n" for x, y, z in normals[start : start + 20_000]))
        output.write("usemtl video_fit_basecolor\ns 1\n")
        for start in range(0, len(faces), 10_000):
            lines = []
            for geometry_face, uv_face in zip(faces[start : start + 10_000], uv_faces[start : start + 10_000]):
                corners = [
                    f"{int(geometry_face[k]) + 1}/{int(uv_face[k]) + 1}/{int(geometry_face[k]) + 1}"
                    for k in range(3)
                ]
                lines.append("f " + " ".join(corners) + "\n")
            output.write("".join(lines))
    mtl_path.write_text(
        "newmtl video_fit_basecolor\n"
        "Ka 0 0 0\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n"
        f"map_Kd {texture_name}\n",
        encoding="utf-8",
    )
    return obj_path


@fal.function(
    image=IMAGE,
    machine_type="GPU-H100",
    request_timeout=10_800,
)
def fit_texture_hq(
    dataset_url: str = DEFAULT_DATASET_URL,
    coarse_iterations: int = DEFAULT_COARSE_ITERATIONS,
    fine_iterations: int = DEFAULT_FINE_ITERATIONS,
    coarse_texture_size: int = DEFAULT_COARSE_TEXTURE,
    fine_texture_size: int = DEFAULT_FINE_TEXTURE,
    coarse_render_size: int = DEFAULT_COARSE_RENDER,
    fine_render_size: int = DEFAULT_FINE_RENDER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    min_median_iou: float = DEFAULT_MIN_MEDIAN_IOU,
):
    """Fit and return a ZIP containing a static atlas, diagnostics and OBJ."""
    import gc
    import hashlib
    import shutil
    import tempfile
    import time

    import numpy as np
    import nvdiffrast.torch as dr
    import torch
    import torch.nn.functional as F
    from PIL import Image as PIL_Image
    from PIL import ImageDraw, ImageFilter

    started = time.perf_counter()
    dataset_url = dataset_url or os.environ.get("DATASET_URL", "")
    if not dataset_url:
        raise ValueError("Pass dataset_url or set DATASET_URL before `fal run`")
    for name, value in {
        "coarse_iterations": coarse_iterations,
        "fine_iterations": fine_iterations,
        "coarse_texture_size": coarse_texture_size,
        "fine_texture_size": fine_texture_size,
        "coarse_render_size": coarse_render_size,
        "fine_render_size": fine_render_size,
        "batch_size": batch_size,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if not 0.0 <= min_median_iou <= 1.0:
        raise ValueError("min_median_iou must be in [0,1]")
    if fine_texture_size < coarse_texture_size:
        raise ValueError("fine_texture_size must be >= coarse_texture_size")
    if fine_render_size < coarse_render_size:
        raise ValueError("fine_render_size must be >= coarse_render_size")
    if any(size & (size - 1) for size in (coarse_texture_size, fine_texture_size)):
        raise ValueError("Texture sizes must be powers of two")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The fitter requires a CUDA GPU")
    device = torch.device("cuda")
    _log(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} CUDA={torch.version.cuda}")
    if not str(torch.__version__).startswith("2.4.1") or torch.version.cuda != "12.1":
        raise RuntimeError("Expected torch 2.4.1 with CUDA 12.1")
    ctx = dr.RasterizeCudaContext(device=device)

    workspace = Path(tempfile.mkdtemp(prefix="car-city-video-fit-hq-"))
    archive_path = workspace / "dataset.zip"
    dataset_dir = workspace / "dataset"
    output_dir = workspace / "results"
    dataset_dir.mkdir()
    output_dir.mkdir()
    _download_dataset(dataset_url, archive_path)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    _log(f"downloaded dataset ({archive_path.stat().st_size / 1024**2:.1f} MiB, sha256={archive_sha256[:12]}…)")
    _safe_extract_zip(archive_path, dataset_dir)

    manifest_path = dataset_dir / "manifest.json"
    npz_path = dataset_dir / "dataset.npz"
    if not manifest_path.is_file() or not npz_path.is_file():
        raise ValueError("Dataset ZIP must contain root manifest.json and dataset.npz")
    manifest = json.loads(manifest_path.read_text())
    schema = str(manifest.get("schema_version", ""))
    if not schema.startswith(SCHEMA_PREFIX):
        raise ValueError(f"Unsupported dataset schema: {schema!r}")
    expected_npz_hash = manifest.get("files", {}).get("dataset_npz_sha256")
    actual_npz_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if expected_npz_hash and actual_npz_hash != expected_npz_hash:
        raise ValueError(
            f"dataset.npz sha256 mismatch: {actual_npz_hash} != {expected_npz_hash}"
        )

    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "video_frame_indices",
            "blender_frame_numbers",
            "frame_files",
            "c2w",
            "w2c",
            "K_crop",
            "projection_crop",
            "vertices_world",
            "vertex_normals_world",
            "faces",
            "uv_vertex_remap",
            "uv_faces",
            "uv_coords",
            "sharpness_laplacian_var",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"dataset.npz missing arrays: {missing}")
        arrays = {name: np.asarray(data[name]) for name in required}

    n = len(arrays["video_frame_indices"])
    indices = arrays["video_frame_indices"]
    if n < 24 or not np.all(np.diff(indices) == np.diff(indices)[0]):
        raise ValueError("Expected at least 24 evenly sampled HQ video frames")
    if batch_size > n:
        raise ValueError(f"batch_size {batch_size} exceeds frame count {n}")
    if not np.array_equal(arrays["blender_frame_numbers"], indices + 1):
        raise ValueError("Pose rule must be Blender frame = video frame + 1")
    vertices_np = arrays["vertices_world"].astype(np.float32, copy=False)
    normals_np = arrays["vertex_normals_world"].astype(np.float32, copy=False)
    faces_np = arrays["faces"].astype(np.int32, copy=False)
    uv_remap_np = arrays["uv_vertex_remap"].astype(np.int32, copy=False)
    uv_faces_np = arrays["uv_faces"].astype(np.int32, copy=False)
    uv_coords_np = arrays["uv_coords"].astype(np.float32, copy=False)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3 or faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError("Invalid mesh shapes")
    if uv_coords_np.ndim != 2 or uv_coords_np.shape[1] != 2 or uv_faces_np.shape != faces_np.shape:
        raise ValueError("Invalid UV topology shapes")
    if not np.array_equal(uv_remap_np[uv_faces_np], faces_np):
        raise ValueError("UV topology no longer maps one-to-one to geometry triangles")
    if not all(np.isfinite(value).all() for value in (vertices_np, normals_np, uv_coords_np, arrays["c2w"], arrays["K_crop"])):
        raise ValueError("Dataset contains non-finite values")
    if uv_coords_np.min() < -1e-5 or uv_coords_np.max() > 1.00001:
        raise ValueError("UVs leave the normalized atlas")

    stored_size = tuple(manifest.get("images", {}).get("stored_size", [1024, 1024]))
    if len(stored_size) != 2 or stored_size[0] != stored_size[1] or stored_size[0] not in (1024, 1536):
        raise ValueError(f"HQ runner expects square 1024/1536 PNG crops, got {stored_size}")
    stored_width, stored_height = int(stored_size[0]), int(stored_size[1])
    atlas_size = tuple(manifest.get("uv", {}).get("atlas_size", [8192, 8192]))
    if atlas_size != (8192, 8192):
        raise ValueError(f"HQ v2 dataset must carry one 8192x8192 atlas, got {atlas_size}")
    if fine_texture_size > max(atlas_size):
        raise ValueError(f"Requested {fine_texture_size}px texture exceeds dataset atlas {atlas_size}")
    if fine_render_size > stored_width:
        raise ValueError(
            f"fine render {fine_render_size}px exceeds stored crop {stored_width}px; "
            "that would be an optimization-time upscale"
        )
    if coarse_render_size > stored_width:
        raise ValueError(
            f"coarse render {coarse_render_size}px exceeds stored crop {stored_width}px; "
            "that would be an optimization-time upscale"
        )
    source_atlas_size = int(max(atlas_size))
    source_padding = int(manifest.get("uv", {}).get("padding_pixels", 8))
    fine_padding = max(
        1, min(16, int(round(source_padding * fine_texture_size / source_atlas_size)))
    )
    coarse_padding = max(
        1, min(16, int(round(source_padding * coarse_texture_size / source_atlas_size)))
    )
    # A 4 px gutter is safe through mip level 2; the production 4K atlas keeps
    # all 8 source pixels and therefore uses the requested maximum level 3.
    coarse_max_mip = min(4, int(np.floor(np.log2(coarse_padding))))
    fine_max_mip = min(4, int(np.floor(np.log2(fine_padding))))
    coarse_tag = f"{coarse_texture_size // 1024}k"
    fine_tag = f"{fine_texture_size // 1024}k"

    frame_paths = [dataset_dir / str(value) for value in arrays["frame_files"].tolist()]
    if not all(path.is_file() for path in frame_paths):
        raise ValueError("One or more frame files listed in dataset.npz are missing")
    expected_frame_hashes = manifest.get("files", {}).get("frame_sha256", [])
    if expected_frame_hashes:
        if len(expected_frame_hashes) != len(frame_paths):
            raise ValueError("Manifest frame hash count does not match the dataset")
        for path, expected_hash in zip(frame_paths, expected_frame_hashes):
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Frame sha256 mismatch: {path.name}")
    frames_np = np.stack([np.asarray(PIL_Image.open(path).convert("RGB"), dtype=np.uint8) for path in frame_paths])
    frames_cpu = torch.from_numpy(frames_np.copy())
    del frames_np
    mask_paths = [dataset_dir / "masks" / f"{index:04d}.png" for index in range(n)]
    masks_present = [path.is_file() for path in mask_paths]
    if any(masks_present) and not all(masks_present):
        raise ValueError("Dataset contains only a partial masks/0000.png..0119.png sequence")
    if all(masks_present):
        expected_mask_hashes = manifest.get("files", {}).get("mask_sha256", [])
        if expected_mask_hashes:
            if len(expected_mask_hashes) != len(mask_paths):
                raise ValueError("Manifest mask hash count does not match the dataset")
            for path, expected_hash in zip(mask_paths, expected_mask_hashes):
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"Mask sha256 mismatch: {path.name}")
        masks_np = np.stack([np.asarray(PIL_Image.open(path).convert("L"), dtype=np.uint8) for path in mask_paths])
        if masks_np.shape != (n, stored_height, stored_width):
            raise ValueError(f"Invalid mask array shape: {masks_np.shape}")
        masks_cpu = torch.from_numpy(masks_np.copy())
        del masks_np
        mask_mode = "BiRefNet target masks"
    else:
        masks_cpu = None
        mask_mode = "rendered-silhouette fallback (no target masks supplied)"
        _log("WARNING: masks are absent; alignment IoU is not independently measurable")

    vertices = torch.as_tensor(vertices_np, dtype=torch.float32, device=device).contiguous()
    vertex_normals = F.normalize(
        torch.as_tensor(normals_np, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    faces = torch.as_tensor(faces_np, dtype=torch.int32, device=device).contiguous()
    uv_faces = torch.as_tensor(uv_faces_np, dtype=torch.int32, device=device).contiguous()
    uv_coords = torch.as_tensor(uv_coords_np, dtype=torch.float32, device=device).contiguous()
    topology_hash = dr.antialias_construct_topology_hash(faces)
    c2w = torch.as_tensor(arrays["c2w"], dtype=torch.float32, device=device)
    K_crop = torch.as_tensor(arrays["K_crop"], dtype=torch.float32, device=device)
    projection_crop = torch.as_tensor(arrays["projection_crop"], dtype=torch.float32, device=device)
    mvp, computed_w2c = _build_mvp(torch, c2w, K_crop, stored_width, stored_height)
    stored_w2c = torch.as_tensor(arrays["w2c"], dtype=torch.float32, device=device)
    inverse_error = float((computed_w2c - stored_w2c).abs().max().item())
    if inverse_error > 2e-5:
        raise ValueError(f"c2w rigid inverse differs from stored w2c by {inverse_error:.6g}")
    projection_error = _validate_projection(
        torch, vertices, mvp, projection_crop, stored_width, stored_height
    )
    _log(f"camera convention validation max error={projection_error:.6f}px")
    if projection_error > 0.5:
        raise ValueError(
            f"OpenGL/K_crop projection mismatch ({projection_error:.3f}px > 0.5px); refusing to fit"
        )

    sharpness = torch.as_tensor(
        arrays["sharpness_laplacian_var"].astype(np.float32), device=device
    )
    median_sharpness = sharpness.median().clamp_min(1e-6)
    sharpness_weights = (sharpness / median_sharpness).sqrt().clamp(0.15, 1.5)

    _log("building coarse fixed-render cache and checking alignment")
    coarse_cache = _build_render_cache(
        torch,
        dr,
        ctx,
        vertices,
        faces,
        uv_coords,
        uv_faces,
        mvp,
        coarse_render_size,
        batch_size,
        topology_hash,
    )
    alignment_ious, rendered_masks = _alignment_check(
        torch, F, dr, coarse_cache, frames_cpu, masks_cpu, faces, batch_size
    )
    median_iou = float(np.median(alignment_ious))
    _alignment_sheet(
        PIL_Image,
        ImageDraw,
        ImageFilter,
        frames_cpu,
        masks_cpu,
        rendered_masks,
        alignment_ious,
        output_dir / "alignment_check.png",
    )
    alignment_summary = {
        "median": median_iou,
        "mean": float(np.mean(alignment_ious)),
        "minimum": float(np.min(alignment_ious)),
        "maximum": float(np.max(alignment_ious)),
        "p10": float(np.quantile(alignment_ious, 0.10)),
        "per_frame": alignment_ious,
        "threshold": min_median_iou,
        "mask_mode": mask_mode,
    }
    _log(
        f"alignment IoU median={median_iou:.3f} mean={alignment_summary['mean']:.3f} "
        f"p10={alignment_summary['p10']:.3f}"
    )
    if masks_cpu is not None and median_iou < min_median_iou:
        failure_report = {
            "status": "alignment_failed",
            "dataset_schema": schema,
            "dataset_sha256": archive_sha256,
            "projection_max_error_pixels": projection_error,
            "alignment": alignment_summary,
        }
        (output_dir / "report.json").write_text(json.dumps(failure_report, indent=2))
        diagnostic_zip = shutil.make_archive(str(workspace / "alignment_failure"), "zip", output_dir)
        diagnostic = File.from_path(diagnostic_zip, content_type="application/zip")
        _log(f"ALIGNMENT_FAILURE_DIAGNOSTIC_URL={diagnostic.url}")
        raise RuntimeError(
            f"Median silhouette IoU {median_iou:.3f} is below {min_median_iou:.3f}; "
            f"camera/mask alignment must be fixed before optimization. Diagnostic: {diagnostic.url}"
        )

    holdout_ids = [index for index in range(n) if index % 5 == 0]
    train_ids = [index for index in range(n) if index % 5 != 0]
    # Extremely weak individual alignment should not poison the atlas even when
    # the median passes.  Keep the threshold conservative to preserve orbit coverage.
    if masks_cpu is not None:
        train_ids = [index for index in train_ids if alignment_ious[index] >= max(0.35, min_median_iou * 0.70)]
    if len(train_ids) < 48:
        raise RuntimeError(f"Only {len(train_ids)} usable training frames remain")

    coarse_occupancy = _uv_occupancy(
        torch, dr, ctx, uv_coords, uv_faces, coarse_texture_size
    )
    initial_texture, coarse_support, coarse_confidence, direct_coverage = _reprojection_init(
        torch,
        F,
        dr,
        coarse_cache,
        frames_cpu,
        masks_cpu,
        train_ids,
        vertices,
        vertex_normals,
        faces,
        c2w,
        sharpness_weights,
        coarse_occupancy,
        coarse_texture_size,
        batch_size,
        coarse_padding,
    )
    _save_rgb(PIL_Image, torch, initial_texture, output_dir / f"texture_reprojection_init_{coarse_tag}.png")
    _save_gray(PIL_Image, torch, coarse_support, output_dir / f"support_{coarse_tag}.png")
    _save_gray(PIL_Image, torch, coarse_confidence, output_dir / f"confidence_{coarse_tag}.png")
    # Support is diagnostic-only; do not retain a full atlas through optimization.
    del coarse_support

    coarse_texture, coarse_history = _optimize_stage(
        np,
        torch,
        F,
        dr,
        coarse_cache,
        frames_cpu,
        masks_cpu,
        faces,
        initial_texture,
        coarse_confidence,
        sharpness_weights,
        train_ids,
        coarse_iterations,
        0.03,
        batch_size,
        seed,
        "coarse",
        coarse_max_mip,
    )
    with torch.no_grad():
        coarse_texture_padded, _ = _pad_atlas(
            torch, F, coarse_texture, coarse_occupancy, coarse_padding
        )
    _save_rgb(PIL_Image, torch, coarse_texture_padded, output_dir / f"texture_{coarse_tag}.png")

    coarse_residuals = _evaluate_residuals(
        torch,
        F,
        dr,
        coarse_cache,
        frames_cpu,
        masks_cpu,
        faces,
        coarse_texture_padded,
        batch_size,
        coarse_max_mip,
    )
    # Two-pass robustness: the fine fit drops the worst 20% of training frames by
    # actual masked reprojection residual, never by held-out scores.
    ranked_train = sorted(train_ids, key=lambda index: coarse_residuals[index])
    retained_count = max(48, int(round(len(ranked_train) * 0.80)))
    fine_train_ids = ranked_train[:retained_count]
    dropped_ids = ranked_train[retained_count:]
    _log(f"fine pass retains {len(fine_train_ids)}/{len(train_ids)} training frames; dropped {dropped_ids}")

    del coarse_cache
    gc.collect()
    torch.cuda.empty_cache()
    _log("building fine fixed-render cache")
    fine_cache = _build_render_cache(
        torch,
        dr,
        ctx,
        vertices,
        faces,
        uv_coords,
        uv_faces,
        mvp,
        fine_render_size,
        batch_size,
        topology_hash,
    )
    fine_occupancy = _uv_occupancy(torch, dr, ctx, uv_coords, uv_faces, fine_texture_size)
    with torch.no_grad():
        coarse_upsampled = F.interpolate(
            coarse_texture_padded.permute(0, 3, 1, 2),
            (fine_texture_size, fine_texture_size),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()
    del coarse_texture_padded, coarse_texture, initial_texture
    del coarse_confidence, coarse_occupancy
    gc.collect()
    torch.cuda.empty_cache()

    # Crucial HQ difference: reproject the original 1024 px targets directly into
    # the 8K UV grid.  The coarse atlas contributes only to low-confidence/unseen
    # texels; it is not treated as newly created fine detail.
    fine_direct, fine_support, fine_confidence, fine_direct_coverage = _reprojection_init(
        torch,
        F,
        dr,
        fine_cache,
        frames_cpu,
        masks_cpu,
        fine_train_ids,
        vertices,
        vertex_normals,
        faces,
        c2w,
        sharpness_weights,
        fine_occupancy,
        fine_texture_size,
        batch_size,
        fine_padding,
    )
    with torch.no_grad():
        blend = fine_confidence.clamp(0.0, 1.0)
        fine_initial = fine_direct * blend + coarse_upsampled * (1.0 - blend)
        fine_initial, _ = _pad_atlas(torch, F, fine_initial, fine_occupancy, fine_padding)
    del fine_direct, coarse_upsampled, blend
    gc.collect()
    torch.cuda.empty_cache()
    _save_rgb(
        PIL_Image,
        torch,
        fine_initial,
        output_dir / f"texture_reprojection_init_{fine_tag}.png",
    )
    _save_gray(PIL_Image, torch, fine_support, output_dir / f"support_{fine_tag}.png")
    # The optimizer uses confidence and the initialization prior, but not the
    # support visualization.  At 8K this early release saves 256 MiB for all
    # 1200 fine iterations.
    del fine_support

    fine_texture, fine_history = _optimize_stage(
        np,
        torch,
        F,
        dr,
        fine_cache,
        frames_cpu,
        masks_cpu,
        faces,
        fine_initial,
        fine_confidence,
        sharpness_weights,
        fine_train_ids,
        fine_iterations,
        0.01,
        batch_size,
        seed + 1,
        "fine",
        fine_max_mip,
    )
    _save_gray(PIL_Image, torch, fine_confidence, output_dir / f"confidence_{fine_tag}.png")
    del fine_initial, fine_confidence
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        fine_padded, padded_mask = _pad_atlas(
            torch, F, fine_texture, fine_occupancy, fine_padding
        )
        fine_padded = torch.where(padded_mask, fine_padded, torch.zeros_like(fine_padded))
        base_atlas = torch.where(fine_occupancy, fine_texture, torch.zeros_like(fine_texture))
    fine_texture_name = f"texture_{fine_tag}_padded.png"
    _save_rgb(PIL_Image, torch, base_atlas, output_dir / f"texture_{fine_tag}.png")
    _save_rgb(PIL_Image, torch, fine_padded, output_dir / fine_texture_name)
    _save_gray(PIL_Image, torch, fine_occupancy.float(), output_dir / f"atlas_occupancy_{fine_tag}.png")

    final_residuals = _evaluate_residuals(
        torch,
        F,
        dr,
        fine_cache,
        frames_cpu,
        masks_cpu,
        faces,
        fine_padded,
        batch_size,
        fine_max_mip,
    )
    preview_ids = holdout_ids[:: max(1, len(holdout_ids) // 8)][:8]
    predictions = []
    with torch.no_grad():
        for index in preview_ids:
            prediction, _, _ = _render_texture(
                torch,
                dr,
                fine_cache,
                [index],
                fine_padded,
                faces,
                max_mip_level=fine_max_mip,
            )
            rendered = prediction[0].clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
            predictions.append(rendered)
    _preview_sheet(
        PIL_Image,
        ImageDraw,
        frames_cpu,
        predictions,
        preview_ids,
        output_dir / "heldout_preview.png",
    )
    # Keep the full fine-render predictions as separate lossless images.  The
    # compact preview sheet is intentionally only a quick visual summary; these
    # files let the local v1-vs-HQ evaluator compare the exact same held-out views
    # without throwing away the 1024 px result first.
    preview_predictions_dir = output_dir / "heldout_predictions"
    preview_predictions_dir.mkdir(parents=True, exist_ok=True)
    for index, prediction in zip(preview_ids, predictions):
        PIL_Image.fromarray(prediction, mode="RGB").save(
            preview_predictions_dir / f"frame_{index:03d}.png",
            format="PNG",
            optimize=True,
        )

    _export_obj(
        np,
        output_dir,
        vertices_np,
        normals_np,
        faces_np,
        uv_coords_np,
        uv_faces_np,
        fine_texture_name,
    )
    peak_vram_allocated_gib = float(torch.cuda.max_memory_allocated(device) / 1024**3)
    peak_vram_reserved_gib = float(torch.cuda.max_memory_reserved(device) / 1024**3)
    total_seconds = time.perf_counter() - started
    report = {
        "status": "complete",
        "method": (
            "HQ v2 fixed-pose static RGB sigmoid atlas; lossless crops; "
            "direct 8K robust reprojection; temporally stratified nvdiffrast fitting"
        ),
        "dataset_schema": schema,
        "dataset_sha256": archive_sha256,
        "mask_mode": mask_mode,
        "versions": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvdiffrast": getattr(__import__("nvdiffrast"), "__version__", "unknown"),
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "resources": {
            "peak_vram_allocated_gib": peak_vram_allocated_gib,
            "peak_vram_reserved_gib": peak_vram_reserved_gib,
        },
        "configuration": {
            "seed": seed,
            "coarse_iterations": coarse_iterations,
            "fine_iterations": fine_iterations,
            "coarse_texture_size": coarse_texture_size,
            "fine_texture_size": fine_texture_size,
            "coarse_render_size": coarse_render_size,
            "fine_render_size": fine_render_size,
            "batch_size": batch_size,
            "coarse_max_mip_level": coarse_max_mip,
            "fine_max_mip_level": fine_max_mip,
            "coarse_atlas_padding": coarse_padding,
            "fine_atlas_padding": fine_padding,
            "coarse_learning_rate": 0.03,
            "fine_learning_rate": 0.01,
            "holdout_rule": "dataset index modulo 5 equals zero",
        },
        "mesh": {
            "vertices": int(len(vertices_np)),
            "triangles": int(len(faces_np)),
            "uv_vertices": int(len(uv_coords_np)),
        },
        "camera_validation": {
            "rigid_inverse_max_abs_error": inverse_error,
            "projection_max_error_pixels": projection_error,
        },
        "alignment": alignment_summary,
        "splits": {
            "holdout": holdout_ids,
            "coarse_train": train_ids,
            "fine_train": fine_train_ids,
            "residual_dropped_before_fine": dropped_ids,
        },
        "reprojection_direct_coverage": {
            "coarse": direct_coverage,
            "fine": fine_direct_coverage,
        },
        "coarse_history": coarse_history,
        "fine_history": fine_history,
        "coarse_masked_l1_per_frame": coarse_residuals,
        "final_masked_l1_per_frame": final_residuals,
        "metrics": {
            "coarse_train_l1_mean": float(np.mean([coarse_residuals[i] for i in train_ids])),
            "final_fine_train_l1_mean": float(np.mean([final_residuals[i] for i in fine_train_ids])),
            "final_holdout_l1_mean": float(np.mean([final_residuals[i] for i in holdout_ids])),
        },
        "heldout_preview": {
            "dataset_indices": preview_ids,
            "prediction_directory": "heldout_predictions",
            "prediction_resolution": fine_render_size,
        },
        "elapsed_seconds": total_seconds,
        "uv_orientation": "OBJ vt uses raw u-right/v-up; PNGs are flipped once from internal v=0 row zero",
        "notes": [
            "No pose or geometry parameter was optimized.",
            "No cross-island TV regularizer was used.",
            f"{fine_texture_name} is the texture referenced by the OBJ material.",
            "Fine detail was initialized from source pixels at fine render resolution, not a coarse upscale.",
            "Training batches are temporally stratified for broad view coverage.",
            "heldout_predictions contains lossless full-resolution fitted views for local v1-vs-HQ evaluation.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "README.txt").write_text(
        "Car City video-fit output\n\n"
        f"Open ferrari_video_fit_hq.obj; its MTL points to {fine_texture_name}.\n"
        f"texture_{fine_tag}.png has only chart interiors; the padded version contains "
        f"{fine_padding} px gutters.\n"
        "alignment_check.png uses red=target mask and green=rendered mesh silhouette.\n"
        "heldout_preview.png compares reference frames (left) with fitted unlit renders (right).\n"
        "See report.json for camera checks, losses, splits and per-frame residuals.\n"
    )

    results_zip = shutil.make_archive(str(workspace / "video_fit_hq_results"), "zip", output_dir)
    artifact = File.from_path(results_zip, content_type="application/zip")
    _log(f"RESULT_URL={artifact.url}")
    _log(
        f"peak VRAM allocated={peak_vram_allocated_gib:.1f} GiB "
        f"reserved={peak_vram_reserved_gib:.1f} GiB"
    )
    _log(f"completed in {total_seconds / 60:.1f} minutes")
    return {
        "status": "complete",
        "results_zip": artifact.url,
        "median_alignment_iou": median_iou,
        "holdout_masked_l1": report["metrics"]["final_holdout_l1_mean"],
        "elapsed_seconds": total_seconds,
    }


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


if __name__ == "__main__":
    # ``fal run`` has no arbitrary function-argument flags.  Reading the local
    # environment here passes explicit serializable arguments to the remote call.
    result = fit_texture_hq(
        dataset_url=os.environ.get("DATASET_URL", DEFAULT_DATASET_URL),
        coarse_iterations=_env_int("FIT_COARSE_ITERS", 500),
        fine_iterations=_env_int("FIT_FINE_ITERS", 1200),
        coarse_texture_size=_env_int("FIT_COARSE_TEXTURE", 4096),
        fine_texture_size=_env_int("FIT_FINE_TEXTURE", 8192),
        coarse_render_size=_env_int("FIT_COARSE_RENDER", 512),
        fine_render_size=_env_int("FIT_FINE_RENDER", 1024),
        batch_size=_env_int("FIT_BATCH_SIZE", 3),
        seed=_env_int("FIT_SEED", 63),
        min_median_iou=_env_float("FIT_MIN_MEDIAN_IOU", 0.55),
    )
    print(json.dumps(result, indent=2, default=str))
