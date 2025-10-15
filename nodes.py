import io
import json
import numpy as np
import requests

from util.nordy_deduct_credit import nordy_deduct_credit
from util.logger import lg

from PIL import Image
from PIL.PngImagePlugin import PngInfo

class SaveImageS3PresignedUrlNordy:
    CATEGORY = "Nordy"
    DESCRIPTION = "Save image to S3 presigned url"
    @classmethod    
    def INPUT_TYPES(s):
        return {
                "required":  {
                    "images": ("IMAGE",),
                    "presigned_url": ("STRING",),
                    "set_metadata": ("BOOLEAN", {"default": False}),
                },
                "hidden": {
                    "prompt": "PROMPT",
                    "extra_pnginfo": "EXTRA_PNGINFO",
                    "job_id": "JOB_ID",
                    "user_id": "USER_ID",
                }
            }
    # OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_image_s3_presigned_url"
    
    def save_image_s3_presigned_url(self, images, presigned_url, set_metadata=False, prompt=None, extra_pnginfo=None, job_id=None, user_id=None,):
        with lg.context(f"SaveImageS3PresignedUrl job_id:{job_id}, user_id:{user_id}"):
            
            lg.debug(f"presigned_url:{presigned_url}, set_metadata:{set_metadata}")
            
            # 첫 번째 이미지만 사용
            image = images[0]

            # presigned url이 없으면 처리하지 않음
            if presigned_url == "" or presigned_url is None:
                lg.debug("presigned_url is None")
                return (images, )

            try:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                
                # 메타데이터 추가
                metadata = None
                if set_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                # PNG 바이트 데이터로 변환
                image_buffer = io.BytesIO()
                img.save(image_buffer, format='PNG', pnginfo=metadata)
                image_buffer.seek(0)

                # presigned URL로 업로드
                headers = {
                    'Content-Type': 'image/png'
                }

                response = requests.put(
                    presigned_url,
                    data=image_buffer.getvalue(),
                    headers=headers
                )

                if response.status_code == 200:
                    lg.debug("Image successfully uploaded to S3")
                    return (images, )
                else:
                    lg.debug(f"Failed to upload image. Status code: {response.status_code}")
                    lg.debug(f"Response: {response.text}")
                    raise RuntimeError(f"Failed to upload image: Response: {response.text}, status: {response.status_code}")

            except Exception as e:
                lg.debug(f"Error uploading image: {str(e)}")
                raise RuntimeError(f"Failed to upload image: {str(e)}")
            
@nordy_deduct_credit
class CreditTesterNordy:
    CATEGORY = "Nordy"
    DESCRIPTION = "Credit tester"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "credit": ("FLOAT",),
            },
            "hidden": {
                "job_id": "JOB_ID",
                "user_id": "USER_ID",
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "credit_tester"
    
    def credit_tester(self, images, credit, job_id, user_id):
        return (images, )