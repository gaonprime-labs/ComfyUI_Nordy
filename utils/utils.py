import math
import json
import torch
import torch.nn.functional as F

EPS = 1e-6

def as_list(x):
    return x if isinstance(x, list) else [x]

def floor_to_multiple(v: float, m: int) -> int:
    return int(math.floor(v / m) * m)

def ceil_to_multiple(v: float, m: int) -> int:
    return int(math.ceil(v / m) * m)

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def ratio_ok_only_one_side(tile_long: int, tile_short: int) -> bool:
    tile_long = int(tile_long)
    tile_short = int(tile_short)
    if tile_long <= 0 or tile_short <= 0 or tile_short > tile_long:
        return False
    # ----------------------------------------
    # size 처리하는 로직은 
    # tile * 4 -> contrain to 1536, 1536 -> * 4 -> 4096 * 4096
    # 1536 직후 사이즈 다 8의 배수로 맞춰주고, 이후는 size가 8의 배수로 맞추지 않아도 됨
    # 수식:
    # short / long = short' / long'(1536)가 8의 배수다 -> short' = short * (1536 / long) 이고, short'이 8의 배수면 됨 -> (192*short) % long == 0
    # ----------------------------------------
    return (192 * tile_short) % tile_long == 0

# 최종적으로 다시한번 검증 -> short-side가 8의 배수 만족하는지
def ratio_ok_final(tile_w: int, tile_h: int) -> bool:
    tl = max(int(tile_w), int(tile_h))
    ts = min(int(tile_w), int(tile_h))
    return ratio_ok_only_one_side(tl, ts)

# 현재 긴 쪽은 1536 이고 짧은 쪽이 8의 배수 되는 가장 작은 사이즈 찾기
def find_min_valid_short_side_given_long_one_side(long_side: int, short_target: float, max_iter: int = 50000) -> int:
    s = max(8, ceil_to_multiple(float(short_target), 8))
    it = 0
    while it < max_iter:
        if ratio_ok_only_one_side(long_side, s):
            return s
        s += 8
        it += 1
    raise RuntimeError("Failed to find valid short side for one-side ratio constraint.")

# 사용자 지정한 factor에 따른 pad 자동 매핑 (8의 배수로)
def auto_pad_for_r(r: float) -> tuple[int,int]:
    #rr = int(round(float(r)))
    if r <= 1:
        p = 128
    elif 1 < r <= 2:
        p = 64
    elif 2 < r <= 3:
        p = 32
    elif 3 < r <= 4:
        p = 16
    else:
        p = int(round(128.0 / max(1.0, float(r))))
        p = max(16, min(128, p))
    p = ceil_to_multiple(p, 8)
    return p, p

def ensure_bhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("Expected IMAGE tensor [B,H,W,C]")
    return x

def linear_ramp_1d(n: int, device, dtype) -> torch.Tensor:
    if n <= 0:
        return torch.ones((0,), device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, steps=n, device=device, dtype=dtype)

def make_overlap_weight_mask(
    h: int, w: int,
    left: int, right: int, top: int, bottom: int,
    device, dtype
) -> torch.Tensor:
    # --------------------------------------------
    # 중간부분 = 1
    # overlap 영역은 0->1 linear로 가중치 마스크 만들기 
    # --------------------------------------------
    wx = torch.ones((w,), device=device, dtype=dtype)
    wy = torch.ones((h,), device=device, dtype=dtype)

    if left > 0:
        r = linear_ramp_1d(left, device, dtype)
        wx[:left] = torch.minimum(wx[:left], r)

    if right > 0:
        r = linear_ramp_1d(right, device, dtype)
        wx[-right:] = torch.minimum(wx[-right:], torch.flip(r, dims=[0]))

    if top > 0:
        r = linear_ramp_1d(top, device, dtype)
        wy[:top] = torch.minimum(wy[:top], r)

    if bottom > 0:
        r = linear_ramp_1d(bottom, device, dtype)
        wy[-bottom:] = torch.minimum(wy[-bottom:], torch.flip(r, dims=[0]))

    return (wy[:, None] * wx[None, :]).view(1, h, w, 1)

def bhwc_to_bchw(x): return x.permute(0, 3, 1, 2).contiguous()

def bchw_to_bhwc(x): return x.permute(0, 2, 3, 1).contiguous()

def gaussian_kernel1d(sigma: float, dtype, device):
    sigma = float(max(0.01, sigma))
    radius = int(math.ceil(2.0 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    k = k / torch.clamp(k.sum(), min=EPS)
    return k, radius

def gaussian_blur_bhwc(img_bhwc, sigma):
    # img: [B,H,W,C] -> blur per channel
    x = bhwc_to_bchw(img_bhwc)
    B, C, H, W = x.shape
    dtype = x.dtype
    device = x.device

    k, r = gaussian_kernel1d(sigma, dtype, device)  # [K]
    kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)      # [C,1,1,K]
    ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)      # [C,1,K,1]

    # reflect padding
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, ky, groups=C)

    return bchw_to_bhwc(x)

def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))

def resize_like_bchw(src: torch.Tensor, ref: torch.Tensor, mode: str) -> torch.Tensor:
    H, W = int(ref.shape[2]), int(ref.shape[3])
    if int(src.shape[2]) == H and int(src.shape[3]) == W:
        return src
    align = False if mode in ("bilinear", "bicubic") else None
    return F.interpolate(src, size=(H, W), mode=mode, align_corners=align)

def to_gray_bchw(img_bhwc: torch.Tensor) -> torch.Tensor:
    if img_bhwc.dim() != 4:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(img_bhwc.shape)}")

    # ensure float
    if not torch.is_floating_point(img_bhwc):
        img_bhwc = img_bhwc.float()

    # safety: if someone passed 0..255
    if img_bhwc.max() > 1.5:
        img_bhwc = img_bhwc / 255.0

    B, H, W, C = img_bhwc.shape

    # drop alpha if RGBA
    if C == 4:
        img_bhwc = img_bhwc[..., :3]
        C = 3

    if C == 3:
        r = img_bhwc[..., 0]
        g = img_bhwc[..., 1]
        b = img_bhwc[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif C == 1:
        gray = img_bhwc[..., 0]
    else:
        raise ValueError(f"Unsupported channel count: {C}")

    return gray.unsqueeze(1).contiguous()  # [B,1,H,W]
