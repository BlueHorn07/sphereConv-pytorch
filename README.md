# sphereConv-pytorch

Fast and Simple Spherical Convolution PyTorch code 🌏

This Code is an unofficial implementation of "SphereNet: Learning Spherical Representations for Detection and Classification in Omnidirectional Images(ECCV 2018)", and upgrade version of [ChiWeiHsiao/SphereNet-pytorch](https://github.com/ChiWeiHsiao/SphereNet-pytorch).

This Code supports spherical kernel sampling on "Equirectangular Image"!

I wrote the code to be `numpy`-friendly and `torch`-friendly. 😉

- [x] `numpy`-friendly
- [x] `torch`-friendly
- [x] Support all size of kernel shape (ex: `3x3`, `2x2`, `3x4`, ...)
- [x] Super Fast! 👍
- [ ] Omnidirectional Dataset <br/>(If you want Omni-Dataset, use this repo  [ChiWeiHsiao/SphereNet-pytorch](https://github.com/ChiWeiHsiao/SphereNet-pytorch))

## Demo Result

![demo](https://i.imgur.com/CWews2K.png)

Spherical Kernel can cross over the sides, and left side has brighter color. Therefore, in result image, the right side has bright color by "MaxPooling"!


## Quick Start

Before start, you should install `pytorch`!! (This code also run on CPU.)

```
cd src
python demo.py
python demo_maxPool.py
```

## Code Detail

### class `GridGenerator`

This is a class that supports to generate spherical sampling grid on equirectangular image.

``` python
gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
LonLatSamplingPattern = gridGenerator.createSamplingPattern()
```

This code only use `numpy` and is written `numpy`-friendly! However, this code is super `numpy`-friendly you may feel hard to understand the flow of code 😢. 

I attach some comments on my code and explain how the shape of array changes. Good Luck 🤞.


### class `SphereConv2d`

This is an implementation of spherical Convolution. This class inherits `nn.Conv2d`, so you can replace `nn.Conv2d` into this.

``` python
cnn = SphereConv2d(3, 5, kernel_size=3, stride=1)
out = cnn(torch.randn(2, 3, 10, 10))
```

This code support various shape of kernels: `(3x3, 2x2, 3x8, ...)`.

You can test this by using OmniMNIST Dataset from [ChiWeiHsiao/SphereNet-pytorch](https://github.com/ChiWeiHsiao/SphereNet-pytorch). I've tested using this, and got similar or improved result!

### class `SphereMaxPool2d`

This is an implementation of spherical Convolution. This class inherits `nn.MaxPool2d`, so you can replace `nn.MaxPool2d` into this.

``` python
pool = SphereMaxPool2d(kernel_size=3, stride=3)
out = pool(torch.from_numpy(img).float())
```

Also, this code support various shape of pooling shape!

Likewise, you can test this by using OmniMNIST Dataset from [ChiWeiHsiao/SphereNet-pytorch](https://github.com/ChiWeiHsiao/SphereNet-pytorch).

## Further Reading

- Some formulas are inspired by Paul Bourke's work. [link](http://paulbourke.net/dome/dualfish2sphere/)
- If you want to rotate equirectangular image, see my implementation! [BlueHorn07/pyEquirectRotate](https://github.com/BlueHorn07/pyEquirectRotate)  
- If you want more awesome omnidirectional python codes, I recommend this repository!
    - [sunset1995/py360convert](https://github.com/sunset1995/py360convert)
        - I've forked this `py360convert`, and add `p2e`, perspective2equirectangular. <br/>[BlueHorn07/py360convert](https://github.com/BlueHorn07/py360convert), [`p2e`](https://github.com/BlueHorn07/py360convert#p2ep_img-fov_deg-u_deg-v_deg-out_hw-in_rot_deg0)
