from .utils import *

def inject_one(pre_image, up_image, resize_mode): 
    # 우리가 지정하는 inject할 detail의 강도 조절하는 파라미터들
    # NMKD_siax_200k 이미 high freq 많이 들어가 있어서 gain 1.0, t 0.05 정도로 하면 적당히 디테일 살리면서도 너무 과하지 않게 되는 것 (이미지마다 다르긴 할듯) 
    # =====================================================================================================================================
    # 제일 중요한 이유 -> constrain image 할 때 zoom in 과정중 정보손실의 보상!
    # 이외의 이유 ->
    # high freq 자꾸 추가한 이유 -> flux upscale 생성한 내용은 원본 이미지비해서 high freq 상대적 부족한 경향이 있어서 (red maple 이미지에서 보이는 상황), 
    # 원본 이미지에서 high freq만 추출해서 flux upscale 하기전에 추가해주면 flux upscaler하고나서도 high freq 상대적 더 잘 나올 수 있음에 도움이 될 수 있을 것 같아서 시도해봄
    # =====================================================================================================================================
    
    # ---------------------------------------------
    # 현재까지 테스트중에 상대적으로 제일 적당한 페러미터들
    sigma = 2
    t = 0.08
    gain = 0.7
    # ---------------------------------------------
    resize_mode = resize_mode[0] 
    # conv2d는 BCHW format맞추게 함
    pre_image = ensure_bhwc(pre_image)
    up_image = ensure_bhwc(up_image)

    pre_bchw = bhwc_to_bchw(pre_image)
    up_bchw = bhwc_to_bchw(up_image)

    # low freq 추출
    pre_bhwc = bchw_to_bhwc(pre_bchw)  
    low_bhwc = gaussian_blur_bhwc(pre_bhwc, sigma=float(sigma))
    low_bchw = bhwc_to_bchw(low_bhwc)
    # orgin - 주변 편균 값차이가 낮은 내용은 = detail(high freq)
    detail = pre_bchw - low_bchw
    # inject high freq라서 너무 강한 디테일이 들어가면 오히려 보기 싫을 수 있어서, pixel 단위로 gain 곱하기 전에 클램프 해서 최대값 제한하는 방식으로 조절할 수 있게 했음. 
    # (gain은 전체적으로 디테일 강도 조절, t는 개별 픽셀 단위로 너무 강한 디테일이 들어가는거 방지하는 역할)
    t = float(t)
    detail = torch.clamp(detail, -t, t)
    # 강도 조절
    detail = detail * float(gain)

    # 원래 사이즈로 돌아감
    detail_up = resize_like_bchw(detail, up_bchw, mode=resize_mode)
    out = up_bchw + detail_up
    out = torch.clamp(out, 0.0, 1.0)

    return bchw_to_bhwc(out)