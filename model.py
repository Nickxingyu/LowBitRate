import torch 

from compressai.models import MeanScaleHyperprior

"""
model = MeanScaleHyperprior(128, 192)

model.g_a.load_state_dict(torch.load(f"./model/mbt2018_mean/quality_1/g_a"))

print(model.g_a.state_dict())
"""

class MeanScaleHalfHyperprior(MeanScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super.__init__(N=N, M=M, **kwargs

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 3, stride=1, kernel_size=3),
        )
