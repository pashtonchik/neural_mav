import json
import matplotlib.pyplot as plt

correct = json.load(open("results.json", "r"))
print(correct.keys())
print(list(correct.keys())[1])
correct_percents = [correct[key] for key in list(correct.keys())]
print(correct_percents)

fig, ax = plt.subplots()

ax.bar(['по 100', 'по 200', 'по 300', 'по 400'], correct_percents)
ax.set_title('Зависимость корректности ответов НС от размера обучающей выборки')
ax.set_ylim([0, 100])
fig.set_figwidth(16)  # ширина Figure
fig.set_figheight(8)
ax.text('по 100', correct_percents[0] + 10, f'{round(correct_percents[0], 3)} %', fontsize=15)
ax.text('по 200', correct_percents[1] + 10, f'{round(correct_percents[1], 3)} %', fontsize=15)
ax.text('по 300', correct_percents[2] + 10, f'{round(correct_percents[2], 3)} %', fontsize=15)
ax.text('по 400', correct_percents[3] + 10, f'{round(correct_percents[3], 3)} %', fontsize=15)
plt.show()

