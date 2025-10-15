from .nodes import SaveImageS3PresignedUrlNordy, CreditTesterNordy

NODE_CLASS_MAPPINGS = {
    "SaveImageS3PresignedUrlNordy": SaveImageS3PresignedUrlNordy,
    "CreditTesterNordy": CreditTesterNordy,
}   

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageS3PresignedUrlNordy": "Save Image S3 PresignedUrl (Nordy)",
    "CreditTesterNordy": "Credit Tester (Nordy)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]