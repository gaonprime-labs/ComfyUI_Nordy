from .nodes import SaveImageS3PresignedUrlNordy, MemoryTest
from .tiled_flux_upscale_nodes import TileSplitByFactor, TileMergeByMeta, TileMergeByMetaWOMaskBlending, DetailInjectAfterUpscale

NODE_CLASS_MAPPINGS = {
    "SaveImageS3PresignedUrlNordy": SaveImageS3PresignedUrlNordy,
    "MemoryTest": MemoryTest,
    # =========tiled_flux_upscale_nodes===========
    "TileSplitByFactor": TileSplitByFactor,
    "TileMergeByMeta": TileMergeByMeta,
    "TileMergeByMetaWOMaskBlending": TileMergeByMetaWOMaskBlending,
    "DetailInjectAfterUpscale": DetailInjectAfterUpscale,
    # ============================================
}   

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageS3PresignedUrlNordy": "Save Image S3 PresignedUrl (Nordy)",
    "MemoryTest": "Memory Test",
    # =========tiled_flux_upscale_nodes===========
    "TileSplitByFactor": "Tile Split By Factor R ",
    "TileMergeByMeta": "Tile Merge By Meta",
    "TileMergeByMetaWOMaskBlending" : "Tile Merge By Meta Without Mask Blending",
    "DetailInjectAfterUpscale": "Detail Inject After Upscale",
    # ============================================
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]