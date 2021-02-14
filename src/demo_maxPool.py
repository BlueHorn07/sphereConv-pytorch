import numpy as np
import cv2
import torch

from spherenet import SphereMaxPool2d

if __name__ == '__main__':
    img = cv2.imread("../demo/equirectangular_earth.png")

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)

    spherePool = SphereMaxPool2d(3, stride=3)
    out = spherePool(torch.from_numpy(img).float())
    out = np.squeeze(out.numpy(), 0).transpose((1, 2, 0))
    cv2.imwrite("sphere_maxPooled.jpg", out)
    cv2.waitKey()
    cv2.destroyAllWindows()




