from .utils import *  


def compute_plan(W: int, H: int, r: float) -> dict:
    # ------------------------------------------
    # 전략: 
    # 1536/r > image long side -> no split
    # core_long = floor_to_multiple(1536/r, 8)
    # pad가 auto_pad_for_r 이함수로 지정함
    # tile = core + 2*pad
    # tile_long 고정, tile_short을 ratio 만족하면서 8의 배수로 맞추면서 조정 (수식 위와 같음) -> 실제 조정한 내용, 계산할 때 pad 길이 고정, core 길이가 따라서 변경
    # tile_w <= W and tile_h <= H 이어야 crop-only 가능, 아니면 no split
    # ------------------------------------------
    r = float(r)
    r = max(1e-6, r)

    long_img = max(W, H)
    core_long_raw = 1536.0 / r

    if core_long_raw > long_img:
        return {
            "do_split": False,
            "reason": "1536/r > image long side",
            "r": r,
            "orig_w": W, "orig_h": H,
            "core_w": W, "core_h": H,
            "pad_x": 0, "pad_y": 0,
            "tile_w": W, "tile_h": H,
            "rows": 1, "cols": 1,
            "tiles": [{"row": 0, "col": 0, "crop_x0": 0, "crop_y0": 0, "crop_w": W, "crop_h": H}]
        }

    # 8의 배수로 
    core_long = max(8, floor_to_multiple(core_long_raw, 8))
    img_ratio = W / H

    # 2) 원본 이미지의 H, W 비율에 따라 core_w, core_h 결정 (긴 쪽이 core_long이 되도록)
    if img_ratio >= 1.0:
        core_w = core_long
        core_h_target = core_w / img_ratio
        core_h = max(8, floor_to_multiple(core_h_target, 8))
    else:
        core_h = core_long
        core_w_target = core_h * img_ratio
        core_w = max(8, floor_to_multiple(core_w_target, 8))

    pad_x, pad_y = auto_pad_for_r(r)

    tile_w0 = int(core_w + 2 * pad_x)
    tile_h0 = int(core_h + 2 * pad_y)

    if tile_w0 >= tile_h0:
        tile_w = tile_w0
        tile_h = find_min_valid_short_side_given_long_one_side(tile_w, tile_h0)

        # pad 8의 배수고 -> core도 8의 배수이어야 됨)
        core_h = int(tile_h - 2 * pad_y)
        if core_h < 8 or (core_h % 8) != 0:
            raise RuntimeError(f"Invalid core_h after ratio solve: core_h={core_h}, tile_h={tile_h}, pad_y={pad_y}")
    else:
        tile_h = tile_h0
        tile_w = find_min_valid_short_side_given_long_one_side(tile_h, tile_w0)

        core_w = int(tile_w - 2 * pad_x)
        if core_w < 8 or (core_w % 8) != 0:
            raise RuntimeError(f"Invalid core_w after ratio solve: core_w={core_w}, tile_w={tile_w}, pad_x={pad_x}")

    # 최종적으로 계산된 tile_w, tile_h이 원본 이미지보다 크면 split 불가 (crop-only 불가), 아니면 split 진행 (예를 들어서 주목 애기 이미지...)
    if tile_w > W or tile_h > H:
        return {
            "do_split": False,
            "reason": "crop-only requires tile fit inside image; computed tile exceeds image",
            "r": r,
            "orig_w": W, "orig_h": H,
            "core_w": W, "core_h": H,
            "pad_x": 0, "pad_y": 0,
            "tile_w": W, "tile_h": H,
            "rows": 1, "cols": 1,
            "tiles": [{"row": 0, "col": 0, "crop_x0": 0, "crop_y0": 0, "crop_w": W, "crop_h": H}]
        }

    cols = ceil_div(W, int(core_w))
    rows = ceil_div(H, int(core_h))

    return {
        "do_split": True,
        "reason": "ok",
        "r": r,
        "orig_w": int(W), "orig_h": int(H),
        "core_w": int(core_w), "core_h": int(core_h),
        "pad_x": int(pad_x), "pad_y": int(pad_y),
        "tile_w": int(tile_w), "tile_h": int(tile_h),
        "rows": int(rows), "cols": int(cols),
        "tiles": []
    }


# 모든 tile의 정보들 meta에 기록하면서 split (crop-only)
def split_tiles_crop_only(image_bhwc: torch.Tensor, r: float):
    assert image_bhwc.dim() == 4, "Expected IMAGE tensor [B,H,W,C]"
    B, H, W, C = image_bhwc.shape

    # merge할 때 meta 기준으로 계산하기 때문에, split할 때도 meta에 core_w, core_h, pad_x, pad_y 기준으로 tile_w, tile_h 계산해서 그걸로 crop 해야 나중에 merge할 때 사이즈 mismatch 안생김
    meta = compute_plan(W, H, r)
    if not meta["do_split"]:
        return [image_bhwc], json.dumps(meta)

    core_w = int(meta["core_w"])
    core_h = int(meta["core_h"])
    pad_x  = int(meta["pad_x"])
    pad_y  = int(meta["pad_y"])
    tile_w = int(meta["tile_w"])
    tile_h = int(meta["tile_h"])
    rows   = int(meta["rows"])
    cols   = int(meta["cols"])

    tiles = []
    for ry in range(rows):
        for cx in range(cols):
            core_x0 = cx * core_w
            core_y0 = ry * core_h

            crop_x0_ideal = core_x0 - pad_x
            crop_y0_ideal = core_y0 - pad_y

            crop_x0 = clamp(crop_x0_ideal, 0, W - tile_w)
            crop_y0 = clamp(crop_y0_ideal, 0, H - tile_h)

            crop_x1 = crop_x0 + tile_w
            crop_y1 = crop_y0 + tile_h

            tile = image_bhwc[:, crop_y0:crop_y1, crop_x0:crop_x1, :]

            core_in_tile_x0 = core_x0 - crop_x0
            core_in_tile_y0 = core_y0 - crop_y0

            core_valid_w = max(0, min(core_w, W - core_x0))
            core_valid_h = max(0, min(core_h, H - core_y0))

            tiles.append(tile.contiguous())

            meta["tiles"].append({
                "row": int(ry), "col": int(cx),
                "crop_x0": int(crop_x0), "crop_y0": int(crop_y0),
                "crop_w": int(tile_w), "crop_h": int(tile_h),
                "core_x0": int(core_x0), "core_y0": int(core_y0),
                "core_w": int(core_w), "core_h": int(core_h),
                "core_valid_w": int(core_valid_w),
                "core_valid_h": int(core_valid_h),
                "core_in_tile_x0": int(core_in_tile_x0),
                "core_in_tile_y0": int(core_in_tile_y0),
                "pad_x": int(pad_x), "pad_y": int(pad_y),
                "tile_w": int(tile_w), "tile_h": int(tile_h),
            })

    # 마지막으로 검증 short side가 계산된거 수식과 맞는지 한번 더 체크
    if not ratio_ok_final(tile_w, tile_h):
        raise RuntimeError(
            f"Final tile size does NOT satisfy one-side ratio constraint: tile_w={tile_w}, tile_h={tile_h}"
        )

    return tiles, json.dumps(meta)
