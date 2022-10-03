import sys

from PIL import Image
import numpy as np

image = Image.open('mainimage.png')

print(image.format)
print(image.size)
print(image.mode)

array_image = np.asarray(image)
np.set_printoptions(threshold=sys.maxsize)
print(type(array_image))
print(array_image.shape)
print(array_image)

