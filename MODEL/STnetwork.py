import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.layer(x)


class SplitChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SplitChannelAttention, self).__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer_frequency = nn.Sequential(
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.layer_spatial = nn.Sequential(
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = self.layer(x)
        return x * self.layer_frequency(squeeze), x * self.layer_spatial(squeeze)


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))

        batch = x.shape[0]

        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv2(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act2(self.bn2(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')

        return output + x


class StructureTextureInteractionModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding_mode, need_bias):
        super(StructureTextureInteractionModule, self).__init__()
        self.sca = SplitChannelAttention(channel=in_channel, ratio=16)

        self.fu = FourierUnit(in_channel, out_channel)

        self.padding_mode = padding_mode
        to_pad = int((kernel_size - 1) / 2)
        if padding_mode == 'reflection':
            self.pad1 = nn.ReflectionPad2d(to_pad)
            self.pad2 = nn.ReflectionPad2d(to_pad)
            to_pad = 0

        self.spatial_branch1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, stride=stride, padding=to_pad, dilation=1, bias=need_bias),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.spatial_branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=to_pad, dilation=1, bias=need_bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x_f, x_s = self.sca(x)
        if self.padding_mode == 'reflection':
            x_s = self.pad1(x_s)
        x_s = self.spatial_branch1(x_s)
        if self.padding_mode == 'reflection':
            x_s = self.pad2(x_s)
        x_s = self.spatial_branch2(x_s)

        return self.fu(x_f) + x_s


class Skip(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding_mode, need_bias):
        super(Skip, self).__init__()
        self.ca = ChannelAttention(channel=in_channel, ratio=16)

        self.padding_mode = padding_mode
        to_pad = int((kernel_size - 1) / 2)
        if padding_mode == 'reflection':
            self.pad = nn.ReflectionPad2d(to_pad)
            to_pad = 0

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=to_pad, dilation=1, bias=need_bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.ca(x)
        if self.padding_mode == 'reflection':
            x = self.pad(x)
        x = self.act(self.bn(self.conv(x)))

        return x


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding_mode, need_bias):
        super(Encoder, self).__init__()
        self.padding_mode = padding_mode
        to_pad = int((kernel_size - 1) / 2)
        if padding_mode == 'reflection':
            self.pad1 = nn.ReflectionPad2d(to_pad)
            self.pad2 = nn.ReflectionPad2d(to_pad)
            to_pad = 0

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride=2, padding=to_pad, dilation=1, bias=need_bias)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride=1, padding=to_pad, dilation=1, bias=need_bias)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.padding_mode == 'reflection':
            x = self.pad1(x)
        x = self.act1(self.bn1(self.conv1(x)))
        if self.padding_mode == 'reflection':
            x = self.pad2(x)
        x = self.act2(self.bn2(self.conv2(x)))

        return x


class SkipS_Encoder_SkipT(nn.Module):
    def __init__(self, in_channel, out_channel_skip, out_channel_encoder, kernel_size, padding_mode, need_bias):
        super(SkipS_Encoder_SkipT, self).__init__()
        self.skipS = Skip(in_channel, out_channel_skip, kernel_size, 1, padding_mode, need_bias)
        self.encoder = Encoder(in_channel, out_channel_encoder, kernel_size, padding_mode, need_bias)
        self.skipT = Skip(in_channel, out_channel_skip, kernel_size, 1, padding_mode, need_bias)

    def forward(self, x):
        skipS = self.skipS(x)
        encoder = self.encoder(x)
        skipT = self.skipT(x)

        return skipS, encoder, skipT


class Decoder(nn.Module):
    def __init__(self, in_channel_up, in_channel_skip, out_channel, kernel_size, padding_mode, need_bias, up_mode):
        super(Decoder, self).__init__()
        self.padding_mode = padding_mode
        to_pad = int((kernel_size - 1) / 2)
        if padding_mode == 'reflection':
            self.pads = nn.ReflectionPad2d(to_pad)
            self.padt = nn.ReflectionPad2d(to_pad)
            to_pad = 0

        # # structure branch
        self.ups = nn.Upsample(scale_factor=2, mode=up_mode)
        self.bns0 = nn.BatchNorm2d(in_channel_up+in_channel_skip)
        self.convs1 = nn.Conv2d(in_channel_up+in_channel_skip, out_channel, kernel_size, stride=1, padding=to_pad, dilation=1, bias=need_bias)
        self.bns1 = nn.BatchNorm2d(out_channel)
        self.acts1 = nn.LeakyReLU(0.2, inplace=True)
        # # 1x1up
        self.convs2 = nn.Conv2d(out_channel, out_channel, 1, stride=1, padding=0, dilation=1, bias=need_bias)
        self.bns2 = nn.BatchNorm2d(out_channel)
        self.acts2 = nn.LeakyReLU(0.2, inplace=True)

        # # texture branch
        self.upt = nn.Upsample(scale_factor=2, mode=up_mode)
        self.bnt0 = nn.BatchNorm2d(in_channel_up+in_channel_skip+in_channel_skip)
        self.convt1 = nn.Conv2d(in_channel_up+in_channel_skip+in_channel_skip, out_channel, kernel_size, stride=1, padding=to_pad, dilation=1, bias=need_bias)
        self.bnt1 = nn.BatchNorm2d(out_channel)
        self.actt1 = nn.LeakyReLU(0.2, inplace=True)
        # # 1x1up
        self.convt2 = nn.Conv2d(out_channel, out_channel, 1, stride=1, padding=0, dilation=1, bias=need_bias)
        self.bnt2 = nn.BatchNorm2d(out_channel)
        self.actt2 = nn.LeakyReLU(0.2, inplace=True)

        # # structure texture interaction module
        self.stim = StructureTextureInteractionModule(out_channel, in_channel_skip, kernel_size, 1, padding_mode, need_bias)

    def forward(self, xs_up, xs_skip, xt_up, xt_skip):
        xs_up = self.ups(xs_up)
        xs = torch.cat((xs_up, xs_skip), dim=1)
        xs = self.bns0(xs)
        if self.padding_mode == 'reflection':
            xs = self.pads(xs)
        xs = self.acts2(self.bns2(self.convs2(self.acts1(self.bns1(self.convs1(xs))))))

        skip = self.stim(xs)

        xt_up = self.upt(xt_up)
        xt = torch.cat((xt_up, xt_skip, skip), dim=1)
        xt = self.bnt0(xt)
        if self.padding_mode == 'reflection':
            xt = self.padt(xt)
        xt = self.actt2(self.bnt2(self.convt2(self.actt1(self.bnt1(self.convt1(xt))))))

        return xs, xt


class network(nn.Module):
    def __init__(self, num_input_channels=64, num_output_channels=1, num_channels_down=128, num_channels_up=128,
                 num_channels_skip=4, filter_size=3, need_bias=True, pad='reflection', upsample_mode='bilinear'):

        super(network, self).__init__()
        self.ses1 = SkipS_Encoder_SkipT(num_input_channels, num_channels_skip, num_channels_down, filter_size, pad,
                                        need_bias)
        self.ses2 = SkipS_Encoder_SkipT(num_channels_down, num_channels_skip, num_channels_down, filter_size, pad,
                                        need_bias)
        self.ses3 = SkipS_Encoder_SkipT(num_channels_down, num_channels_skip, num_channels_down, filter_size, pad,
                                        need_bias)
        self.ses4 = SkipS_Encoder_SkipT(num_channels_down, num_channels_skip, num_channels_down, filter_size, pad,
                                        need_bias)
        self.ses5 = SkipS_Encoder_SkipT(num_channels_down, num_channels_skip, num_channels_down, filter_size, pad,
                                        need_bias)

        self.output_layer_s = nn.Sequential(
            nn.Conv2d(num_channels_up, num_output_channels, kernel_size=1, stride=1, padding=0, bias=need_bias),
            nn.Sigmoid()
        )
        self.output_layer_t = nn.Conv2d(num_channels_up, num_output_channels, 1)

        self.d1 = Decoder(num_channels_up, num_channels_skip, num_channels_up, filter_size, pad, need_bias,
                          upsample_mode)
        self.d2 = Decoder(num_channels_up, num_channels_skip, num_channels_up, filter_size, pad, need_bias,
                          upsample_mode)
        self.d3 = Decoder(num_channels_up, num_channels_skip, num_channels_up, filter_size, pad, need_bias,
                          upsample_mode)
        self.d4 = Decoder(num_channels_up, num_channels_skip, num_channels_up, filter_size, pad, need_bias,
                          upsample_mode)
        self.d5 = Decoder(num_channels_down, num_channels_skip, num_channels_up, filter_size, pad, need_bias,
                          upsample_mode)

    def forward(self, x):
        skipS1, encoder1, skipT1 = self.ses1(x)
        skipS2, encoder2, skipT2 = self.ses2(encoder1)
        skipS3, encoder3, skipT3 = self.ses3(encoder2)
        skipS4, encoder4, skipT4 = self.ses4(encoder3)
        skipS5, encoder5, skipT5 = self.ses5(encoder4)

        outS5, outT5 = self.d5(encoder5, skipS5, encoder5, skipT5)
        outS4, outT4 = self.d4(outS5, skipS4, outT5, skipT4)
        outS3, outT3 = self.d3(outS4, skipS3, outT4, skipT3)
        outS2, outT2 = self.d2(outS3, skipS2, outT3, skipT2)
        outS1, outT1 = self.d1(outS2, skipS1, outT2, skipT1)

        outS = self.output_layer_s(outS1)
        outT = self.output_layer_t(outT1)

        return outS, outT
