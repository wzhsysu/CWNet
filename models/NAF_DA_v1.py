
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class RMSNorm2d(nn.Module):
    """
    Simple RMSNorm over the channel dimension of an (N, C, H, W) tensor.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels), requires_grad=True)

    def forward(self, x):
        # x: (N, C, H, W)
        rms = x.pow(2).mean(dim=1, keepdim=True)        # (N, 1, H, W)
        x_norm = x / torch.sqrt(rms + self.eps)
        return self.weight.view(1, -1, 1, 1) * x_norm   # broadcast to (N, C, H, W)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # LayerNorm2d replaced by RMSNorm2d
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # self.norm1 = RMSNorm2D(c)
        # self.norm2 = RMSNorm2D(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[1,1,1,1], dec_blk_nums=[1,1,1,1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        middle = x

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        # x = self.ending(x)
        # x = x + inp
        # x = x.permute(0, 2, 3, 1)  # [B, H, W ,D]
        #
        # # flatten the output to [B, H*W, D]
        # x = torch.flatten(x, 1, 2)

        return x, middle

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAF_DA(nn.Module):
    """White-balance correction model - lightweight convolutions instead of an MLP."""

    def __init__(self, cfg, decompose_num=5, width=16):
        super().__init__()
        self.decompose_num = decompose_num
        self.width = width

        # backbone feature extraction
        self.backbone = NAFNet(width=16,enc_blk_nums=[1,1,1,8],dec_blk_nums=[1,1,1,1],middle_blk_num=12)

        # chroma prediction
        self.chroma_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, decompose_num * 2)  # output B*dn*2
        )

        self.spatial_modulator = nn.Sequential(
            nn.Conv2d(self.width, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.decompose_num, 1)  # spatially varying basis weights
        )



    def forward(self, x, get_every_ittr=False):
        """
        Args:
            x: [B, 3, H, W] input image
            get_every_ittr: whether to return intermediate results (kept for interface compatibility)
        Returns:
            dict: outputs such as mixed_illum
        """
        chorma = []
        mixmap = []

        B, C, H, W = x.shape
        original_image = x

        features, middle = self.backbone(x)  # [B, width, H, W], [B, width * 16, H/16, W/16]

        chroma_info = self.chroma_predictor(middle)  # [B, dn*2]
        chroma_info = chroma_info.view(B, self.decompose_num, 2)  # [B, dn, 2]



        spatial_weights = self.spatial_modulator(features)  # (B, decompose_num, H, W)
        spatial_weights = F.softmax(spatial_weights, dim=1)


        spatial_weights_flat = spatial_weights.view(B, self.decompose_num, -1)  # (B, decompose_num, H*W)

        weighted_chroma = torch.einsum('bki,bkn->bin', chroma_info, spatial_weights_flat)

        mixed_illum = weighted_chroma.view(B, 2, H, W)

        output_list = [mixed_illum]
        chorma.append(chroma_info)
        mixmap.append(spatial_weights)

        ret_dict = {
            "mixed_illum": output_list[-1],
            "chroma" : chorma[-1],
            "mixmap" : mixmap[-1]
        }

        if get_every_ittr:
            ret_dict["mixed_illum"] = torch.stack(output_list, dim=1)
            ret_dict["chroma"] = torch.stack(chorma, dim=1)
            ret_dict["mixmap"] = torch.stack(mixmap, dim=1)


        return ret_dict
