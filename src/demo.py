import matplotlib.pyplot as plt
from spherenet import SphereMaxPool2d, SphereConv2d

import numpy as np
import torch

if __name__ == '__main__':
    """
    This demo code is originated from here "https://github.com/ChiWeiHsiao/SphereNet-pytorch"
    """

    # SphereConv2d
    cnn = SphereConv2d(3, 5, kernel_size=3, stride=1)
    out = cnn(torch.randn(2, 3, 10, 10))
    print('SphereConv2d(3, 5, 1) output shape: ', out.size())

    # SphereMaxPool2d
    h, w = 100, 200
    img = np.ones([h, w, 3])
    for r in range(h):
        for c in range(w):
            img[r, c, 0] = img[r, c, 0] - r/h
            img[r, c, 1] = img[r, c, 1] - c/w
    plt.imsave('demo_original.png', img)
    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, 0)  # (B, C, H, W)

    # pool
    pool = SphereMaxPool2d(kernel_size=3, stride=3)
    out = pool(torch.from_numpy(img).float())

    out = np.squeeze(out.numpy(), 0).transpose([1, 2, 0])
    plt.imsave('demo_pool_3x3.png', out)


