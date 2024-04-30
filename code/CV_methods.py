import numpy as np
import PIL
from PIL import Image, ImageFilter
from skimage.metrics import mean_squared_error

def normalize_array(array):
  return (array - np.min(array)) / (np.max(array) - np.min(array))

def compare_images_mse(image_one, image_two):
  array_one = normalize_array(np.array(image_one))
  array_two = normalize_array(np.array(image_two))
  return mean_squared_error(array_one, array_two)

def upsample_bicubic(orig_img, upscaling_factor = None, new_size = None):
  if upscaling_factor != None:
    new_size = tuple(x * upscaling_factor for x in orig_img.size)
  bicubic_upscaling_img = orig_img.resize(new_size, Image.BICUBIC)
  return bicubic_upscaling_img

def upsample_nearest_neighbours(orig_img, upscaling_factor = None, new_size = None):
  if upscaling_factor != None:
    new_size = tuple(x * upscaling_factor for x in orig_img.size)
  nearest_upscaling_img = orig_img.resize(new_size, Image.NEAREST)
  return nearest_upscaling_img

def upsample_sharp(orig_img, upscaling_factor = None, new_size = None):
  if upscaling_factor != None:
    new_size = tuple(x * upscaling_factor for x in orig_img.size)
  bicubic_upscaling_img = orig_img.resize(new_size, Image.BICUBIC)
  sharp_upscaling_img = bicubic_upscaling_img.filter(PIL.ImageFilter.UnsharpMask())
  return sharp_upscaling_img
