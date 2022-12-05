import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
# plt.rcParams["figure.figsize"] = (5,5)

df = pd.read_csv('results_classify.csv')
ax = sns.barplot(x='model', y='value', hue='classify_topics', data=df)
# df = pd.read_csv('results_ft.csv')
# ax = sns.barplot(x='metric', y='value', hue='model', data=df)
ax.set_xlabel("")
ax.set_ylabel("")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.legend([],[], frameon=False)

plt.xticks(rotation=20)
plt.tight_layout()
# plt.legend(loc='upper center')
# plt.legend(ncol=2).set_title('')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.).set_title('')
plt.show()
# plt.savefig("fig.png")
