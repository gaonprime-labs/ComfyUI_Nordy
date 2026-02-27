import torch
from .utils.utils import to_gray_bchw
from .utils.image_metrics_score import laplacian_variance
from .utils.image_metrics_score import sobel_metrics

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

class TextureDensityMetrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reduce": (["mean", "min", "max"],),  # B>1 인 경우 reduce 방식 선택
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("grad_mean", "grad_var")
    FUNCTION = "run"
    CATEGORY = "image/quality"

    def run(self, image, reduce="mean"):
        gray = to_gray_bchw(image)

        gmean_b, gvar_b = sobel_metrics(gray)

        if reduce == "mean":
            gmean = gmean_b.mean()
            gvar = gvar_b.mean()
        elif reduce == "min":
            gmean = gmean_b.min()
            gvar = gvar_b.min()
        else: 
            gmean = gmean_b.max()
            gvar = gvar_b.max()

        gmean_f = float(gmean.item())
        gvar_f = float(gvar.item())
        return (gmean_f, gvar_f)