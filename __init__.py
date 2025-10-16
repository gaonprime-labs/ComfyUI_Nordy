from .nodes import SaveImageS3PresignedUrlNordy

NODE_CLASS_MAPPINGS = {
    "SaveImageS3PresignedUrlNordy": SaveImageS3PresignedUrlNordy,
}   

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageS3PresignedUrlNordy": "Save Image S3 PresignedUrl (Nordy)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]