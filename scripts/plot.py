"""
plot results

Created on 04/13/2020

@author: RH
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results = pd.read_csv('../Results/meta_summary.csv', header=0)
results['name'] = results['model']+'_'+results['state'].str.get(0)

# All in one
a = results[['name', 'AUROC_tile']]
a['metrics'] = 'AUROC_tile'
a = a.rename(columns={'AUROC_tile': 'value'})
b = results[['name', 'AUPRC_tile']]
b['metrics'] = 'AUPRC_tile'
b = b.rename(columns={'AUPRC_tile': 'value'})
c = results[['name', 'accuracy_tile']]
c['metrics'] = 'accuracy_tile'
c = c.rename(columns={'accuracy_tile': 'value'})
d = results[['name', 'precision_tile']]
d['metrics'] = 'precision_tile'
d = d.rename(columns={'precision_tile': 'value'})
e = results[['name', 'recall_tile']]
e['metrics'] = 'recall_tile'
e = e.rename(columns={'recall_tile': 'value'})
f = results[['name', 'F1_tile']]
f['metrics'] = 'F1_tile'
f = f.rename(columns={'F1_tile': 'value'})

recombined = pd.concat([a, d, b, e, c, f])

grid = sns.catplot(x="name", y="value", hue="metrics", kind="bar", data=recombined, height=5, aspect=4)
grid.set_xticklabels(rotation=45, horizontalalignment='right', fontsize='medium', fontweight='light')
grid.set(ylim=(0, 1))
plt.show()


# Scatter plots
a = results[['name', 'AUROC_tile', 'accuracy_tile']]
a = a.rename(columns={'AUROC_tile': 'AUROC', 'accuracy_tile': 'accuracy'})
a['metrics'] = 'AUROC vs accuracy'
b = results[['name', 'precision_tile', 'recall_tile']]
b = b.rename(columns={'precision_tile': 'precision', 'recall_tile': 'recall'})
b['metrics'] = 'precision vs recall'
c = results[['name', 'F1_tile', 'AUPRC_tile']]
c = c.rename(columns={'F1_tile': 'F1', 'AUPRC_tile': 'AUPRC'})
c['metrics'] = 'F1 vsv AUPRC'

a = a.fillna(0)
b = b.fillna(0)
c = c.fillna(0)

ga = sns.FacetGrid(a, hue="name", col="metrics", height=4,
                  margin_titles=True, palette=sns.color_palette("muted"))
ga.map(plt.scatter, "AUROC", "accuracy", edgecolor="white", s=15, lw=0.1)
ga.set(xlim=(0, 1), ylim=(0, 1))
ax = ga.axes.ravel()[0]
ax.legend(fontsize='xx-small', ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True)
plt.show()

gb = sns.FacetGrid(b, hue="name", col="metrics", height=4,
                  margin_titles=True, palette=sns.color_palette("muted"))
gb.map(plt.scatter, "precision", "recall", edgecolor="white", s=15, lw=0.1)
gb.set(xlim=(0, 1), ylim=(0, 1))
ax = gb.axes.ravel()[0]
ax.legend(fontsize='xx-small', ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True)
plt.show()

gc = sns.FacetGrid(c, hue="name", col="metrics", height=4,
                  margin_titles=True, palette=sns.color_palette("muted"))
gc.map(plt.scatter, "F1", "AUPRC", edgecolor="white", s=15, lw=0.1)
gc.set(xlim=(0, 1), ylim=(0, 1))
ax = gc.axes.ravel()[0]
ax.legend(fontsize='xx-small', ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True)
plt.show()


