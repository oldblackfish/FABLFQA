import torch
import torch.nn as nn
from einops import rearrange
import math


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.div = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(5, 1, 1), stride=(1, 1, 1), dilation=(5, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )

        self.FQL = FQL()

        '''regression'''
        self.task_main = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        self.task_aux = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 5)
        )

        self.gap3D = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()

    def forward(self, x):
        x_out = self.div(x)

        x_out = self.FQL(x_out)
        x_out = self.flat(self.gap3D(x_out))

        q = self.task_main(x_out)
        c = self.task_aux(x_out)

        return q, c


class FQL(nn.Module):

    def __init__(self):
        super(FQL, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU()
        )
        self.angRes = 5
        self.channels_1 = 128
        self.channels_2 = 256
        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.
        self.trans_1 = AngTrans(self.channels_1, self.angRes, self.MHSA_params)
        self.trans_2 = AngTrans(self.channels_2, self.angRes, self.MHSA_params)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        for m in self.modules():
            m.h = x.size(-2)
            m.w = x.size(-1)
        ang_position = self.pos_encoding(x, dim=[2], token_dim=self.channels_1)
        for m in self.modules():
            m.ang_position = ang_position
        x = self.trans_1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        for m in self.modules():
            m.h = x.size(-2)
            m.w = x.size(-1)
        ang_position = self.pos_encoding(x, dim=[2], token_dim=self.channels_2)
        for m in self.modules():
            m.ang_position = ang_position
        x = self.trans_2(x)
        return x


class AngTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(AngTrans, self).__init__()
        self.angRes = angRes
        self.ang_dim = channels
        self.norm = nn.LayerNorm(self.ang_dim)
        self.attention = nn.MultiheadAttention(self.ang_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    @staticmethod
    def SAI2Token(buffer):
        buffer_token = rearrange(buffer, 'b c a h w -> a (b h w) c')
        return buffer_token

    def Token2SAI(self, buffer_token):
        buffer = rearrange(buffer_token, '(a) (b h w) (c) -> b c a h w', a=self.angRes, h=self.h, w=self.w)
        return buffer

    def forward(self, buffer):
        ang_token = self.SAI2Token(buffer)
        ang_PE = self.SAI2Token(self.ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = self.Token2SAI(ang_token)

        return buffer


class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim: list, token_dim):
        self.token_dim = token_dim
        assert len(x.size()) == 5, 'the object of position encoding requires 5-dim tensor! '
        grid_dim = torch.linspace(0, self.token_dim - 1, self.token_dim, dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature ** grid_dim
        position = None
        for index in range(len(dim)):
            pos_size = [1, 1, 1, 1, 1, self.token_dim]
            length = x.size(dim[index])
            pos_size[dim[index]] = length

            pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
            pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
            pos_dim = pos_dim.view(pos_size)

            if position is None:
                position = pos_dim
            else:
                position = position + pos_dim
            pass

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position / len(dim)


if __name__ == "__main__":
    net = Network().cuda()
    from thop import profile

    input1 = torch.randn(1, 1, 25, 64, 64).cuda()
    flops, params = profile(net, inputs=(input1,))
    print('   Number of parameters: %.5fM' % (params / 1e6))
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))
