import sys
from itertools import chain

from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join

# onlyfiles = [f for f in listdir('Latin/H/') if isfile(join('Latin/H/', f))]

# print(onlyfiles)

# image = Image.open('5a6735484b979.png')
# image = image.resize((20, 20))
# image.save('result.png')
# #
# print(image.format)
# print(image.size)
# print(image.mode)
#
# # array_image = np.array([[[1, 2], [1, 2]]])
# array_image = np.asarray(image)
# print(type(array_image))
# print(array_image.shape)
#
# array_image = array_image.tolist()
# res = []
#
# for x in array_image:
#     res.extend(x if isinstance(x, list) else [x])
#
# res1 = []
# for x in res:
#     res1.append(0 if x == [0, 0, 0, 0] else 1)
#
# print(len(res1))
# print(res1.count(0))
# print(res1)

# a = np.random.rand(len(res))
# print(len(a)

W1 = np.random.randint(-10, 10, (2, 10)) / 10
print(W1)
print(W1[0, :])