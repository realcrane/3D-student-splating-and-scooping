from diff_t_rasterization import compute_relocation, compute_relocation_student_t
import torch
import math

N_max = 51
binoms = torch.zeros((N_max, N_max)).float().cuda()
for n in range(N_max):
    for k in range(n+1):
        binoms[n, k] = math.comb(n, k)

# NOTE: this is for Gaussian
def compute_relocation_cuda(opacity_old, scale_old, N):
    N.clamp_(min=1, max=N_max)
    return compute_relocation(opacity_old, scale_old, N, binoms, N_max)

# NOTE: this is for Student's t
def compute_relocation_student_t_cuda(opacity_old, scale_old, nu_degree, N):
    N.clamp_(min=1, max=N_max)
    return compute_relocation_student_t(opacity_old, scale_old, nu_degree, N, binoms, N_max)