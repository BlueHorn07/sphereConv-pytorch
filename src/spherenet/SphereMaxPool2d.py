import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .GridGenerator import GridGenerator


class SphereMaxPool2d(nn.MaxPool2d):
  """
  kernel_size: (H, W)
  """

  def __init__(self, kernel_size=(3, 3), stride=1, padding=0, dilation=1,
               return_indices: bool = False, ceil_mode: bool = False):
    super(SphereMaxPool2d, self).__init__(
      kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    if isinstance(kernel_size, int):
      self.kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
      self.stride = (stride, stride)

    self.grid_shape = None
    self.grid = None

  def genSamplingPattern(self, h, w):
    gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)

    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape

    if (self.grid_shape is None) or (self.grid_shape != (H, W)):
      self.grid_shape = (H, W)
      self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    x = F.grid_sample(x, grid, align_corners=False, mode='bilinear')  # (B, in_c, H*Kh, W*Kw)

    x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.kernel_size)

    return x  # (B, out_c, H/stride_h, W/stride_w)
