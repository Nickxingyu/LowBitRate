import torch 

from compressai.models import MeanScaleHyperprior

"""
model = MeanScaleHyperprior(128, 192)

model.g_a.load_state_dict(torch.load(f"./model/mbt2018_mean/quality_1/g_a"))

print(model.g_a.state_dict())
"""

class MeanScaleHalfHyperprior(MeanScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        N = int(N)
        M = int(M)
        if M % 2 != 0:
            raise ValueError(f"Num of channels of feature map have to be even.")
        super().__init__(N=N, M=M, **kwargs)
        self.N = N
        self.M = M

    def forward(self, x):
        # print(x.shape)
        
        # y = self.g_a(x)
        # z = self.h_a(y)
        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # gaussian_params = self.h_s(z_hat)
        # scales_hat, means_hat = gaussian_params.chunk(2, 1)

        ## AE mean parameters
        #  split means_hat and scales_hat for 
        #  encoding and decoding feature map


        ## Directly predictive parameters
        #  combine the decoded feature map and the predictive 
        #  feature map, and push it into the context model for
        #  constructing whole feature map.

    @staticmethod
    def get_mask(height, width, dtype, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
        micro_row = torch.tensor((1, 0), dtype=dtype, device=device)
        mask_row = micro_row.repeat(height // 2)
        mask_col = micro_row.repeat(width // 2)
        mask_0 = micro_mask.repeat(height // 2, width // 2)

        
        # make the mask for odd num of rows or cols
        if height % 2 == 1 and width % 2 == 1:
            mask_col = torch.cat((mask_col, torch.tensor([1], dtype=dtype, device=device)), dim=0)

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

