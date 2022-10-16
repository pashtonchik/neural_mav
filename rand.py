import random
from PIL import Image
import os
from os.path import isfile, join
from os import listdir


directories = ['dataset_400/' + chr(i) + '/' for i in range(65, 91)]
for dir in directories:
    print(dir)
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    print(files)
    indexes = [a for a in range(len(files))]
    print(type(indexes))
    random.shuffle(indexes)
    print(indexes)
    image_path = f"dataset_300/{dir[len(dir) - 2]}/"
    print(image_path)
    os.mkdir(image_path)
    for index in indexes[:300]:
        image = Image.open(f'{dir}{files[index]}')
        image = image.save(f"{image_path}/{dir[len(dir) - 2]}_{index}.png")

