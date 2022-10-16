import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from app import f_activation
from app import der_f_activation


def go_forward(inp):
    s = np.dot(W1, inp)
    out = np.array([f_activation(x) for x in s])

    s = np.dot(W2, out)
    y = f_activation(s)

    return y, out

W1 = json.load(open("weights.json", "r"))['W1']
W2 = json.load(open("weights.json", "r"))['W2']

if __name__ == "__main__":
    example = Image.open('test/o.png').resize((20, 20))
    example = np.asarray(example).tolist()
    res = []
    for x in example:
        res.extend(x if isinstance(x, list) else [x])
    example = []
    for x in res:
        example.append(0 if x == [0, 0, 0, 0] else 1)

    print('Пример:')

    print(go_forward(example)[0])
    answer = go_forward(example)[0]
    max1 = 0
    index = 0
    for i in range(len(answer)):
        if answer[i] > max1:
            max1 = answer[i]
            index = i
    print(chr(index + 65))

    fig, ax = plt.subplots()

    x = np.array([chr(i) for i in range(65, 91)])
    y = np.array(answer)
    ax.bar(x, y)

    ax.set_facecolor('seashell')
    fig.set_facecolor('floralwhite')
    fig.set_figwidth(12)  # ширина Figure
    fig.set_figheight(6)  # высота Figure

    plt.show()
