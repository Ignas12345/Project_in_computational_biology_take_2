import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors

def generate_noise(image, upscaling_factor, mean = 0, std_dev = 0.2):
  img_size = np.array(image).shape
  # next line taken from : https://stackoverflow.com/questions/1781970/multiplying-a-tuple-by-a-scalar
  img_size = tuple(i * upscaling_factor for i in img_size)
  noise = np.random.normal(mean, std_dev, img_size)
  #noise = (noise-np.min(noise))/(np.max(noise)-np.min(noise))
  return Image.fromarray(noise)

def image_to_tensor(image):
  tensor = v2.PILToTensor()(image)
  tensor = (tensor-torch.min(tensor))/(torch.max(tensor)-torch.min(tensor))
  #tensor = v2.Normalize([0.5], [0.5])(tensor)
  return tensor.unsqueeze(0)

def tensor_to_image(tensor):
  tensor = tensor.squeeze()
  return v2.ToPILImage()(tensor)

def turn_to_grayscale(image):
  return v2.Grayscale()(image)

def downsample(orig_img, downscaling_factor):
  size = tuple(x//downscaling_factor for x in orig_img.size)
  return orig_img.resize(size, resample = Image.Resampling.LANCZOS)

def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()
