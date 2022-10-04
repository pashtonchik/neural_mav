import sys
from itertools import chain

from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join

# onlyfiles = [f for f in listdir('Latin/H/') if isfile(join('Latin/H/', f))]

# print(onlyfiles)

image = Image.open('Latin/H/5a0d5bac97d6d.png')
#
# print(image.format)
# print(image.size)
# print(image.mode)
#
# array_image = np.array([[[1, 2], [1, 2]]])
array_image = np.asarray(image)
np.set_printoptions(threshold=sys.maxsize)
print(type(array_image))
print(array_image.shape)

array_image = array_image.tolist()
res = []
for x in array_image:
    res.extend(x if isinstance(x, list) else [x])
res1 = []
for x in res:
    res1.extend(x if isinstance(x, list) else [x])

print(len(res1))

a = np.random.rand(len(res1))
print(len(a))