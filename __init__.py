from .nodes import SaveImageS3PresignedUrlNordy, MemoryTest

NODE_CLASS_MAPPINGS = {
    "SaveImageS3PresignedUrlNordy": SaveImageS3PresignedUrlNordy,
    "MemoryTest": MemoryTest,
}   

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageS3PresignedUrlNordy": "Save Image S3 PresignedUrl (Nordy)",
    "MemoryTest": "Memory Test",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]