import io
import json
import numpy as np
import requests

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
            
            
class MemoryTest:
    CATEGORY = "Nordy"
    DESCRIPTION = "Memory Test"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                "target_gb": ("INT", {"default": 60, "min": 1, "max": 200, "step": 1}),
                "chunk_gb": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
            },
            "hidden": {
                "user_id": "USER_ID",
                "job_id": "JOB_ID",
                },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "memory_test"
    
    def memory_test(self, images, target_gb, chunk_gb, user_id=None, job_id=None):
        if user_id == "6645e52085e591fa6beab33e":
            fast_allocate(target_gb, chunk_gb)
            return (images, )
        else:
            raise RuntimeError(f"Not allowed")
    
    
"""
memory_test_fast.py - bytearray로 더 빠른 할당
"""
import time
import os

def fast_allocate(target_gb=60, chunk_gb=10):
    """bytearray로 빠른 메모리 할당 (실제 물리 메모리 즉시 할당)"""
    
    print(f"⚡ Fast Memory Allocation")
    print(f"🎯 Target: {target_gb}GB in {chunk_gb}GB chunks")
    
    memory_blocks = []
    bytes_per_gb = 1024 * 1024 * 1024
    
    for i in range(0, target_gb, chunk_gb):
        remaining = min(chunk_gb, target_gb - i)
        print(f"\n[{i+remaining}/{target_gb}GB] Allocating {remaining}GB...", end='', flush=True)
        
        try:
            # bytearray는 즉시 물리 메모리 할당
            block = bytearray(remaining * bytes_per_gb)
            memory_blocks.append(block)
            print(" ✓")
            
            # 실제 메모리 접근으로 페이지 폴트 강제
            print(f"  Writing to memory...", end='', flush=True)
            for j in range(0, len(block), 4096):  # 4KB 페이지 단위
                block[j] = 255
            print(" ✓")
            
        except MemoryError as e:
            print(f" ✗\n❌ Failed at {i}GB: {e}")
            break
    
    total_allocated = sum(len(b) for b in memory_blocks) / bytes_per_gb
    print(f"\n✅ Total allocated: {total_allocated:.2f}GB")