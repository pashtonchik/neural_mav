import random
from PIL import Image
import os
from os.path import isfile, join
from os import listdir


directories = ['Latin/' + chr(i) + '/' for i in range(65, 91)]
for dir in directories:
    print(dir)
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    print(files)
    indexes = [1, 2, 3, 4, 5, 6]
    print(type(indexes))
    random.shuffle(indexes)
    print(indexes)
    image_path = f"test_dataset/{dir[len(dir) - 2]}/"
    print(image_path)
    os.mkdir(image_path)
    for index in indexes:
        image = Image.open(f'{dir}{files[index]}')
        image = image.save(f"{image_path}/{dir[len(dir) - 2]}_{index}.png")

