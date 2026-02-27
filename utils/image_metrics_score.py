import torch 
import torch.nn.functional as F

# laplacian variance 계산하는 함수 -> 이미지의 디테일 정도 판단하는 지표로 사용하기 위해서 
# 나오는 결과가 0.01기준으로 0.001보다 낮으면 호릇함 상대적 많음 -> flux upscale로 처리필요, 0.01보다 높으면 호릇함이 상대적으로 많지 않음
def laplacian_variance(gray_bchw):
    device = gray_bchw.device
    dtype = gray_bchw.dtype

    lap_kernel = torch.tensor(
        [[[[0, -1, 0],
           [-1,  4, -1],
           [0, -1, 0]]]],
        device=device,
        dtype=dtype
    )

    resp = F.conv2d(gray_bchw, lap_kernel, padding=1)  
    resp = resp.view(resp.size(0), -1)                 
    return resp.var(dim=1)    

# sobel metrics 계산하는 함수 -> 이미지의 texture 밀도정도 판단하는 지표로 사용하기 위해서#
# 많을 수록 밀도 더 많아 보임 (chicago ...)
# 0.1기준으로 0.1이하 이면 texture가 상대적으로 적음 -> flux upscale로 처리필요, 0.1보다 높으면 texture가 상대적으로 많음 (건물 많은 이미지 등)
# 한국아버지 -> 0.0437 
# gigi_default -> 0.0284
# red_maple -> 0.2461
# Chicago -> 0.4596
def sobel_metrics(gray_bchw):
    device = gray_bchw.device
    dtype = gray_bchw.dtype

    kx = torch.tensor(
        [[[[ -1, 0, 1],
           [ -2, 0, 2],
           [ -1, 0, 1]]]],
        device=device, dtype=dtype
    )
    ky = torch.tensor(
        [[[[ -1, -2, -1],
           [  0,  0,  0],
           [  1,  2,  1]]]],
        device=device, dtype=dtype
    )

    gx = F.conv2d(gray_bchw, kx, padding=1)
    gy = F.conv2d(gray_bchw, ky, padding=1)

    # gradient magnitude
    gm = torch.sqrt(gx * gx + gy * gy + 1e-12)  
    gm = gm.view(gm.size(0), -1)              

    # 평균 값만 먼저 적용 전체 texture에 대해서 많은지를 판단..
    grad_mean = gm.mean(dim=1)
    grad_var = gm.var(dim=1, unbiased=False)
    return grad_mean, grad_var
