from .utils.split_by_factor import *  
from .utils.merge_by_meta import *  
from .utils.inject_detail import *

# 사용자가 지정한 factor에 따른 tile 사이즈과 padding 계산해서 meta에 기록
class TileSplitByFactor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # 소숫 허용 (소숫점 둘쨰자리까지)
                "r": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_META")
    RETURN_NAMES = ("tiles", "tile_meta")
    OUTPUT_IS_LIST = (True, False)

    FUNCTION = "run"
    CATEGORY = "image/tiling"

    def run(self, image, r):
        tiles, tile_meta = split_tiles_crop_only(image, r=float(r))
        return (tiles, tile_meta)

# core 기준으로 merge하기
class TileMergeByMetaWOMaskBlending:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_meta": ("TILE_META", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/tiling"
    INPUT_IS_LIST = True

    def run(self, tiles, tile_meta):
        if isinstance(tile_meta, list):
            tile_meta = tile_meta[0]
        merged = merge_tiles_crop_only_rescaled(tiles, tile_meta)
        return (merged,)

class TileMergeByMeta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_meta": ("TILE_META", ),
                "overlap_ratio": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05}),
                "overlap_min_px": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1}),
                "overlap_max_px": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/tiling"
    INPUT_IS_LIST = True

    def run(self, tiles, tile_meta, overlap_ratio, overlap_min_px, overlap_max_px):
        tile_meta = tile_meta[0]
        overlap_ratio = overlap_ratio[0]
        overlap_min_px = overlap_min_px[0]
        overlap_max_px = overlap_max_px[0]
        if isinstance(tile_meta, list):
            tile_meta = tile_meta[0]

        # 
        merged = merge_tiles_crop_only_rescaled_overlap_blend_hf_select(
            tiles, tile_meta,
            overlap_ratio=float(overlap_ratio),
            overlap_min_px=int(overlap_min_px),
            overlap_max_px=int(overlap_max_px),
        )
        return (merged,)


class DetailInjectAfterUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pre_image": ("IMAGE",),
                "up_image": ("IMAGE",),
                "resize_mode": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "inject"
    CATEGORY = "image/postprocess"

    def inject(self, pre_image, up_image, resize_mode):
        pre_list = as_list(pre_image)
        up_list = as_list(up_image)

        outs = []
        for pre_i, up_i in zip(pre_list, up_list):
            outs.append(inject_one(pre_i, up_i, resize_mode))

        return (outs,)