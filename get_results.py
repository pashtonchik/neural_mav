import json
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image

W1 = json.load(open("weights_200.json", "r"))['W1']
W2 = json.load(open("weights_200.json", "r"))['W2']

def f_activation(x):
    return 1 / (1 + np.exp(-x))


def der_f_activation(x):
    return f_activation(x) * (1 - f_activation(x))


def go_forward(inp):
    s = np.dot(W1, inp)
    out = np.array([f_activation(x) for x in s])

    s = np.dot(W2, out)
    y = f_activation(s)
    # print(y.shape)
    # print(out.shape)
    return y, out


if __name__ == '__main__':
    correct = 0
    directories = ['test_dataset/' + chr(i) + '/' for i in range(65, 91)]
    all = 0
    for dir in directories:
        print(dir)
        files = [f for f in listdir(dir) if isfile(join(dir, f))]
        i = 0
        all += len(files)
        for file in files:
            # if i % 50 == 0:
            #     print(f'{i}/{len(files)}')
            image = Image.open(f'{dir}{file}').resize((20, 20))
            image_array = np.asarray(image).tolist()


            res = []
            for x in image_array:
                res.extend(x if isinstance(x, list) else [x])
            res1 = []
            for x in res:
                res1.append(0 if x == [0, 0, 0, 0] else 1)

            true_output = dir[len(dir) - 2]
            neural_output, out = go_forward(res1)
            index_max = 0
            for k in range(len(neural_output)):
                if neural_output[k] > neural_output[index_max]:
                    index_max = k
            neural_output = chr(index_max + 65)
            if neural_output == true_output:
                correct += 1
    print(correct / all)

