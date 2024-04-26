import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors

#this is used for some of the optimization loops
from .downsampler import Downsampler

def generate_noise(image, upscaling_factor = 1, mean = 0, std_dev = 0.2):
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

def crop(image, new_size):
    bbox = [0, 0, new_size[0], new_size[1]]
    return image.crop(bbox)

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

def optimize_vanilla(model, input_img, target_img, n_iterations, lr = 0.001, save_every_n_iters = 25, dtype = torch.FloatTensor):
      
  input_tensor = image_to_tensor(input_img).type(dtype)
  target_tensor = image_to_tensor(target_img).type(dtype)
  optimizer = optim.Adam(model.parameters(), lr = lr)
  mse = nn.MSELoss().type(dtype) 

  for iteration in range(n_iterations+1):

    model.train()

    optimizer.zero_grad()
    output = model(input_tensor)
    loss = mse(output, target_tensor)
    loss.backward()
    optimizer.step()
    
    print(f'iteration [{iteration}/{n_iterations}], Loss: {loss.item()}')

    if iteration % save_every_n_iters == 0:
    
      output_img = tensor_to_image(output)
      output_img.save('DIP_images/High_resolution_iteration_' + str(iteration) + '.jpg')
      plot([target_img, output_img], cmap = 'gray')

def optimize_upscaling(model, input_img, target_img, n_iterations, lr = 0.001, save_every_n_iters = 25, upscaling_factor = 1, dtype = torch.FloatTensor):
      
  input_tensor = image_to_tensor(input_img).type(dtype)
  target_tensor = image_to_tensor(target_img).type(dtype)
  optimizer = optim.Adam(model.parameters(), lr = lr)
  mse = nn.MSELoss().type(dtype) 
  downsampler = Downsampler(n_planes = 1, factor = upscaling_factor, kernel_type= 'lanczos2', preserve_size=True)

  for iteration in range(n_iterations+1):

    model.train()

    optimizer.zero_grad()
    output = model(input_tensor)
    loss = mse(downsampler(output), target_tensor)
    loss.backward()
    optimizer.step()
    
    print(f'iteration [{iteration}/{n_iterations}], Loss: {loss.item()}')

    if iteration % save_every_n_iters == 0:
    
      output_img = tensor_to_image(output)
      output_img.save('DIP_images/High_resolution_iteration_' + str(iteration) + '.jpg')
      plot([target_img, output_img], cmap = 'gray')
