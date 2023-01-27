from __future__ import print_function

import cv2
import torch.optim
import math as ma
import scipy.io as sio

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from utils.common_utils import *
from MODEL.STnetwork import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# torch.cuda.set_device(1)
dtype = torch.cuda.FloatTensor


path = './data/Set11/barbara.tif'  # the data path
img = cv2.imread(path, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
img = img[:, :, 0]
row = img.shape[0]
col = img.shape[1]

MR = 0.1               # meaurement rate: from 0.01 to 0.5
block_size = 32
input_depth = 64
num_iter = 6000
LR = 0.001
row = row - np.mod(row, block_size)
col = col - np.mod(col, block_size)
img = img[:row, :col]

img_numpy = np.asarray(img).astype(np.float32)/255
print(img_numpy.shape)

imgsize = block_size * block_size
size_y = round(int(imgsize)*MR)

data = sio.loadmat('measurement_matrix/Phi_50.mat')
phi = data['Phi']
phi = phi[0:size_y, :]
phi = torch.from_numpy(phi).float().cuda()

INPUT = 'noise'
OPT_OVER = 'net'
reg_noise_std = 0.03

net_input = get_noise(input_depth, INPUT, (row, col)).type(dtype).detach()
net = network(num_input_channels=input_depth, num_output_channels=1, num_channels_down=128, num_channels_up=128,
              num_channels_skip=4, filter_size=3, need_bias=True, pad='reflection', upsample_mode='bilinear')
net = net.type(dtype)

# Loss
l1loss = torch.nn.L1Loss(reduction='sum').type(dtype)


class TVloss(nn.Module):
    def __init__(self, TVloss_weight=0.024):
        super(TVloss, self).__init__()
        self.TVloss_weight = TVloss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()

        return self.TVloss_weight * (h_tv + w_tv) / batch_size


tv_loss = TVloss()


def closure():
    global i, torchout_img, net_input, out_imgS, out_imgT
    total_loss = 0
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out_imgS, out_imgT = net(net_input)  # the output of the network
    torchout_img = (out_imgS + out_imgT).view(row, col)

    total_loss += tv_loss(out_imgS)

    out_imgS = out_imgS.view(row, col)   # structure component
    out_imgT = out_imgT.view(row, col)   # texture component

    for num_r in range(1, int(ma.ceil(row / block_size)) + 1):
        for num_c in range(1, int(ma.ceil(col / block_size)) + 1):
            img_block = torch.from_numpy(img_numpy)[(num_r - 1) * block_size:num_r * block_size,
                        (num_c - 1) * block_size:num_c * block_size]
            image = img_block.reshape(block_size * block_size, 1)

            outimg_block = torchout_img[(num_r - 1) * block_size:num_r * block_size,
                           (num_c - 1) * block_size:num_c * block_size]
            outimage = outimg_block.reshape(block_size * block_size, 1)

            outimg_blockS = out_imgS[(num_r - 1) * block_size:num_r * block_size,
                           (num_c - 1) * block_size:num_c * block_size]
            outimageS = outimg_blockS.reshape(block_size * block_size, 1)

            y = phi.mm(image.cuda())
            outy = phi.mm(outimage).cuda()
            outyS = phi.mm(outimageS).cuda()
            total_loss += l1loss(outy, y) + 0.1 * l1loss(outyS, y)

    total_loss.backward()
    psnr = compare_psnr(np.asarray(img), torchout_img.detach().cpu().numpy()*255)
    ssim = compare_ssim(np.asarray(img), torchout_img.detach().cpu().numpy()*255)
    print("the %d iteration %.2f  %.4f" % (i, psnr, ssim))
    i += 1

    return total_loss


net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0

p = get_params(OPT_OVER, net, net_input)
optimizer = torch.optim.Adam(p, lr=LR)

for j in range(num_iter):
    optimizer.zero_grad()
    total_loss = closure()
    nn.utils.clip_grad_norm_(p, 5)
    optimizer.step()
