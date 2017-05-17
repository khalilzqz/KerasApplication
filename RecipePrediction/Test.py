# -*- encoding:utf-8 -*-
import numpy as np
datalist = [[9.99957919e-01, 6.16392811e-07],
            [9.95113790e-01, 1.41417910e-03],
            [9.99989867e-01,  1.21732683e-06],
            [9.99037385e-01, 1.53857206e-06]]

data = [1, 2, 1, 0]
a = np.array(datalist)
# 可视化
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

tps_idx = {'全部': 0, '东北菜': 1, '湘菜': 3, '鲁菜': 2}
_, ax = plt.subplots(figsize=(15, 12))
for tp, idx in tps_idx.items():
    ax.scatter(a[data == idx, 0], a[data == idx, 1],
               c=cm.Vega20.colors[idx], label=tp)

ax.set_xlim(-20, 40)
ax.set_ylim(-20, 20)
ax.set_xticks([])
ax.set_yticks([])

ax.legend(loc=0)
ltext = ax.get_legend().get_texts()
font_song = FontProperties(fname="D:/DataForMining/SIMSUN.TTC")
for t in ltext:
    t.set_font_properties(font_song)
    t.set_fontsize("xx-large")

plt.show()
