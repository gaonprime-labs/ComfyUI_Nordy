import torch
from .utils.utils import to_gray_bchw
from .utils.laplacian_score import laplacian_variance


class LaplacianVarianceScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reduce": (["mean", "min", "max"],),  
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("variance",)
    FUNCTION = "run"
    CATEGORY = "image/quality"

    def run(self, image, reduce="mean"):
        gray = to_gray_bchw(image)
        vars_b = laplacian_variance(gray)  

        if reduce == "mean":
            out = vars_b.mean()
        elif reduce == "min":
            out = vars_b.min()
        elif reduce == "max":
            out = vars_b.max()
        else:
            out = vars_b.mean()

        return (float(out.item()),)