import torch

from torch import nn
from compressai.models import MeanScaleHyperprior
from compressai.models.utils import conv


class MeanScaleHalfHyperprior(MeanScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        N = int(N)
        M = int(M)
        if N % 2 != 0 or M % 2 != 0:
            raise ValueError(f"Num of channels of feature map have to be even.")
        super().__init__(N=N, M=M, **kwargs)
        self.N = N
        self.M = M
        self.e_p = nn.Sequential(
            conv(M * 2, M * 2, stride=1, kernel_size=3),
        )

        self.y_predictor = nn.Sequential(
            conv(M * 4, M * 2, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M * 2, M * 1, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M * 1, M * 1, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        hyperprior = self.h_s(z_hat)
        gaussian_params = self.e_p(hyperprior)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_0, _ = self.spatial_split(y)
        scales_hat_0, scales_hat_1 = self.spatial_split(scales_hat)
        means_hat_0, means_hat_1 = self.spatial_split(means_hat)

        y_0_hat, y_0_likelihoods = self.gaussian_conditional(
            y_0,
            scales_hat_0,
            means=means_hat_0,
        )

        means_hat_0 = y_0_hat
        means_hat = self.spatial_merge(means_hat_0, means_hat_1)
        scales_hat_0 = torch.zeros_like(scales_hat_0)
        scales_hat = self.spatial_merge(scales_hat_0, scales_hat_1)
        param_for_prediction = torch.cat((means_hat, scales_hat, hyperprior), dim=1)

        y_predicted = self.y_predictor(param_for_prediction)
        x_hat = self.g_s(y_predicted)

        return {
            x_hat,
            y_0_likelihoods,
            z_likelihoods,
        }

    def spatial_split(self, feature_map):
        dtype = feature_map.dtype
        device = feature_map.device
        _, _, H, W = feature_map.size()
        mask_0, mask_1 = self.get_mask(H, W, dtype, device)

        feature_map_0, feature_map_1 = feature_map.chunk(2, 1)
        feature_map_0_0, feature_map_1_0 = (
            feature_map_0 * mask_0,
            feature_map_1 * mask_0,
        )
        feature_map_0_1, feature_map_1_1 = (
            feature_map_0 * mask_1,
            feature_map_1 * mask_1,
        )

        return feature_map_0_0 + feature_map_1_1, feature_map_0_1 + feature_map_1_0

    def spatial_merge(self, feature_map_0, feature_map_1):
        feature_map = torch.cat((feature_map_0, feature_map_1), dim=1)
        return torch.cat(self.spatial_split(feature_map), dim=1)

    @staticmethod
    def get_mask(height, width, dtype, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
        micro_row = torch.tensor((1, 0), dtype=dtype, device=device)
        mask_row = micro_row.repeat(height // 2)
        mask_col = micro_row.repeat(width // 2)
        mask_0 = micro_mask.repeat(height // 2, width // 2)

        # make the mask for odd num of rows or cols
        if height % 2 == 1 and width % 2 == 1:
            mask_col = torch.cat(
                (mask_col, torch.tensor([1], dtype=dtype, device=device)), dim=0
            )

        if width % 2 == 1:
            mask_0 = torch.cat((mask_0, torch.unsqueeze(mask_row, 1)), dim=1)

        if height % 2 == 1:
            mask_0 = torch.cat((mask_0, torch.unsqueeze(mask_col, 0)), dim=0)

        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_1 = torch.ones_like(mask_0) - mask_0
        return mask_0, mask_1

    """
    Process with mask

    def example(self, x):
        dtype = x.dtype
        device = x.device
        _, _, H, W = x.size()
        mask_0, mask_1 = self.get_mask(H, W, dtype, device)

    
    mask_0.shape : torch.Size([1, 1, H, W])
    mask_1.shape : torch.Size([1, 1, H, W])

    mask_0 with H=8, W=8:

    [[[
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ]]]

    mask_1 with H=8, W=8:

    [[[
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
    ]]]
    
    """

    def load_encoder(self, encoder_path="./models/mbt2018_mean/quality_1/g_a"):
        self.g_a.load_state_dict(torch.load(encoder_path))

    def load_decoder(self, encoder_path="./models/mbt2018_mean/quality_1/g_s"):
        self.g_s.load_state_dict(torch.load(encoder_path))

    def load_pretrain(self, model_path="./models/mbt2018_mean/", quality=1):
        self.load_encoder(model_path + f"quality_{quality}/" + "g_a")
        self.load_decoder(model_path + f"quality_{quality}/" + "g_s")
