from .utils import *  

def merge_tiles_crop_only_rescaled(tiles, tile_meta: str) -> torch.Tensor:
    # -------------------------------------------
    # 모든 tile 다 똑같은 사이즈라 tiles[0]이미지 기준으로 함
    # meta 안에 기록된 tile사이즈와 현재 upscale된 이미지를 이용해 scale를 계산함
    # scale이 이용해서 upscaled 된 core의 위치를 다시 계산하고 최종canvas에(out) core만 붙이기
    # -------------------------------------------
    meta = json.loads(tile_meta)

    if not meta.get("do_split", False):
        img = ensure_bhwc(tiles[0])
        return img.contiguous()

    tile_records = meta["tiles"]
    if len(tile_records) != len(tiles):
        raise RuntimeError(f"tiles count mismatch: got {len(tiles)}, meta has {len(tile_records)}")

    crop_w = int(meta["tile_w"])
    crop_h = int(meta["tile_h"])

    first = ensure_bhwc(tiles[0])
    B, new_h, new_w, C = first.shape
    device = first.device
    dtype = first.dtype

    sx = new_w / float(crop_w)
    sy = new_h / float(crop_h)

    # 비율이 똑같해야 하는데 만약에 미세차이 생기면 평균내서 하나의 s로 통일
    s_canvas = float(0.5 * (sx + sy))

    orig_w = int(math.ceil(int(meta["orig_w"]) * s_canvas))
    orig_h = int(math.ceil(int(meta["orig_h"]) * s_canvas))

    # 최종 canvas를 계산 사이즈로 초기화하기 
    out = torch.zeros((B, orig_h, orig_w, C), device=device, dtype=dtype)

    def endpoints(x0: int, y0: int, w: int, h: int, s: float):
        x1 = x0 + w
        y1 = y0 + h
        dx0 = int(math.floor(x0 * s))
        dy0 = int(math.floor(y0 * s))
        dx1 = int(math.ceil(x1 * s))
        dy1 = int(math.ceil(y1 * s))
        return dx0, dy0, dx1, dy1

    for t, rec in zip(tiles, tile_records):
        t = ensure_bhwc(t)
        _, th, tw, _ = t.shape

        sx_i = tw / float(crop_w)
        sy_i = th / float(crop_h)
        s = float(0.5 * (sx_i + sy_i))

        core_x0 = int(rec["core_x0"])
        core_y0 = int(rec["core_y0"])
        core_valid_w = int(rec.get("core_valid_w", meta["core_w"]))
        core_valid_h = int(rec.get("core_valid_h", meta["core_h"]))

        dst_x0, dst_y0, dst_x1, dst_y1 = endpoints(core_x0, core_y0, core_valid_w, core_valid_h, s)

        dst_x0 = max(0, min(dst_x0, orig_w))
        dst_y0 = max(0, min(dst_y0, orig_h))
        dst_x1 = max(0, min(dst_x1, orig_w))
        dst_y1 = max(0, min(dst_y1, orig_h))
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            continue

        cit_x0 = int(rec["core_in_tile_x0"])
        cit_y0 = int(rec["core_in_tile_y0"])
        src_x0, src_y0, src_x1, src_y1 = endpoints(cit_x0, cit_y0, core_valid_w, core_valid_h, s)

        src_x0 = max(0, min(src_x0, tw))
        src_y0 = max(0, min(src_y0, th))
        src_x1 = max(0, min(src_x1, tw))
        src_y1 = max(0, min(src_y1, th))

        pw = min(dst_x1 - dst_x0, src_x1 - src_x0)
        ph = min(dst_y1 - dst_y0, src_y1 - src_y0)
        if pw <= 0 or ph <= 0:
            continue

        dst_x1 = dst_x0 + pw
        dst_y1 = dst_y0 + ph
        src_x1 = src_x0 + pw
        src_y1 = src_y0 + ph

        out[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = t[:, src_y0:src_y1, src_x0:src_x1, :]

    return out.contiguous()


def merge_tiles_crop_only_rescaled_overlap_blend_hf_select(
    tiles,
    tile_meta: str,
    overlap_ratio: float = 1.0,
    overlap_min_px: int = 16,
    overlap_max_px: int = 256,
    hf_sigma: float = 1.2,    
):
    # -------------------------------------------
    # overlap_min_px ~ overlap_max_px 사용자가 지정할 수 있으며 이 범위 내에서, meta에 기록된 padding과 사용자가 지정한 overlap_ratio를 이용해 tile간의 실제 overlap 영역 크기 계산하기
    # low freq는 overlap 영역에서 가중치 mask 만들어서 여러 tile이 겹치는 부분은 가중 평균으로 부드럽게 합성하기
    # high freq는 겹치는 부분에서 pixel마다 가중치 mask가 더 큰 tile의 high freq를 선택해서 합성하기 (winner-take-all) 이래야지 겹치는 부분이 여러 tile의 평균이 되면서 뭉개지는 현상 방지할 수 있을가같음
    # -------------------------------------------
    meta = json.loads(tile_meta)

    if not meta.get("do_split", False):
        img = ensure_bhwc(tiles[0])
        return img.contiguous()

    tile_records = meta["tiles"]
    if len(tile_records) != len(tiles):
        raise RuntimeError(f"tiles count mismatch: got {len(tiles)}, meta has {len(tile_records)}")

    crop_w = int(meta["tile_w"])
    crop_h = int(meta["tile_h"])

    first = ensure_bhwc(tiles[0])
    B, new_h, new_w, C = first.shape
    device = first.device
    dtype = first.dtype

    sx = new_w / float(crop_w)
    sy = new_h / float(crop_h)
    s_canvas = float(0.5 * (sx + sy))

    orig_w = int(round(int(meta["orig_w"]) * s_canvas))
    orig_h = int(round(int(meta["orig_h"]) * s_canvas))

    # lowfreq blend accum 
    accum_low  = torch.zeros((B, orig_h, orig_w, C), device=device, dtype=dtype)
    weight_low = torch.zeros((B, orig_h, orig_w, 1), device=device, dtype=dtype)

    # highfreq select (winner-take-all) 
    out_high   = torch.zeros((B, orig_h, orig_w, C), device=device, dtype=dtype)
    best_score = torch.zeros((B, orig_h, orig_w, 1), device=device, dtype=dtype)  # per-pixel best m

    rows = int(meta.get("rows", 1))
    cols = int(meta.get("cols", 1))

    for t, rec in zip(tiles, tile_records):
        t = ensure_bhwc(t)
        _, th, tw, _ = t.shape

        sx_i = tw / float(crop_w)
        sy_i = th / float(crop_h)
        s = float(0.5 * (sx_i + sy_i))

        r = int(rec["row"]); c = int(rec["col"])

        # core pos in output
        core_x0 = int(round(int(rec["core_x0"]) * s))
        core_y0 = int(round(int(rec["core_y0"]) * s))

        core_valid_w = int(rec.get("core_valid_w", meta["core_w"]))
        core_valid_h = int(rec.get("core_valid_h", meta["core_h"]))
        core_w_out = int(round(core_valid_w * s))
        core_h_out = int(round(core_valid_h * s))

        # core in tile
        core_in_tile_x0 = int(round(int(rec["core_in_tile_x0"]) * s))
        core_in_tile_y0 = int(round(int(rec["core_in_tile_y0"]) * s))

        # pad -> overlap
        pad_x = int(rec.get("pad_x", meta.get("pad_x", 0)))
        pad_y = int(rec.get("pad_y", meta.get("pad_y", 0)))
        pad_out_x = int(round(pad_x * s))
        pad_out_y = int(round(pad_y * s))

        ov_x = int(round(pad_out_x * float(overlap_ratio)))
        ov_y = int(round(pad_out_y * float(overlap_ratio)))
        ov_x = max(overlap_min_px, min(overlap_max_px, ov_x))
        ov_y = max(overlap_min_px, min(overlap_max_px, ov_y))

        has_left   = (c > 0)
        has_right  = (c < cols - 1)
        has_top    = (r > 0)
        has_bottom = (r < rows - 1)

        ext_l = ov_x if has_left else 0
        ext_r = ov_x if has_right else 0
        ext_t = ov_y if has_top else 0
        ext_b = ov_y if has_bottom else 0

        # limit by actual tile margins
        ext_l = min(ext_l, core_in_tile_x0)
        ext_t = min(ext_t, core_in_tile_y0)
        ext_r = min(ext_r, max(0, tw - (core_in_tile_x0 + core_w_out)))
        ext_b = min(ext_b, max(0, th - (core_in_tile_y0 + core_h_out)))

        dst_x0 = core_x0 - ext_l
        dst_y0 = core_y0 - ext_t
        dst_x1 = core_x0 + core_w_out + ext_r
        dst_y1 = core_y0 + core_h_out + ext_b

        src_x0 = core_in_tile_x0 - ext_l
        src_y0 = core_in_tile_y0 - ext_t
        src_x1 = core_in_tile_x0 + core_w_out + ext_r
        src_y1 = core_in_tile_y0 + core_h_out + ext_b

        # clamp dst and sync src
        if dst_x0 < 0:
            sh = -dst_x0; dst_x0 = 0; src_x0 += sh
        if dst_y0 < 0:
            sh = -dst_y0; dst_y0 = 0; src_y0 += sh
        if dst_x1 > orig_w:
            sh = dst_x1 - orig_w; dst_x1 = orig_w; src_x1 -= sh
        if dst_y1 > orig_h:
            sh = dst_y1 - orig_h; dst_y1 = orig_h; src_y1 -= sh

        src_x0 = max(0, min(src_x0, tw)); src_x1 = max(0, min(src_x1, tw))
        src_y0 = max(0, min(src_y0, th)); src_y1 = max(0, min(src_y1, th))

        ph = src_y1 - src_y0
        pw = src_x1 - src_x0
        if ph <= 0 or pw <= 0:
            continue

        # ensure dst matches src size
        dst_x1 = dst_x0 + pw
        dst_y1 = dst_y0 + ph
        if dst_x1 > orig_w or dst_y1 > orig_h:
            pw = min(pw, orig_w - dst_x0)
            ph = min(ph, orig_h - dst_y0)
            if pw <= 0 or ph <= 0:
                continue
            src_x1 = src_x0 + pw; src_y1 = src_y0 + ph
            dst_x1 = dst_x0 + pw; dst_y1 = dst_y0 + ph

        patch = t[:, src_y0:src_y1, src_x0:src_x1, :]  # [B,ph,pw,C]

        # mask m
        m = make_overlap_weight_mask(
            h=ph, w=pw,
            left=ext_l, right=ext_r, top=ext_t, bottom=ext_b,
            device=device, dtype=dtype
        ).expand(B, -1, -1, -1)  # [B,ph,pw,1]

        # frequency split (Gaussian Low-pass filter)
        low = gaussian_blur_bhwc(patch, sigma=hf_sigma)
        high = patch - low

        # lowfreq blend
        accum_low[:, dst_y0:dst_y1, dst_x0:dst_x1, :] += low * m
        weight_low[:, dst_y0:dst_y1, dst_x0:dst_x1, :] += m

        # highfreq select 추출과 비교
        # m가 큰 tile이 해당 픽셀의 high freq 결정권을 가짐 (winner-take-all)
        cur_best = best_score[:, dst_y0:dst_y1, dst_x0:dst_x1, :]
        take = (m > cur_best).to(dtype)  # [B,ph,pw,1] 0/1
        best_score[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = torch.maximum(cur_best, m)
        # take 계산한 결과를 이용해 이 pixel high이면 체대쓰기, 아니면 그대로 남김
        out_high[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = (
            out_high[:, dst_y0:dst_y1, dst_x0:dst_x1, :] * (1.0 - take) + high * take
        )

    out_low = accum_low / torch.clamp(weight_low, min=EPS)
    out = out_low + out_high
    out = torch.clamp(out, 0.0, 1.0)
    return out.contiguous()

