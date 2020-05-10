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

# # Scatter plots
# a = results[['name', 'AUROC_tile', 'AUROC_slide']]
# a = a.rename(columns={'AUROC_tile': 'tile', 'AUROC_slide': 'slide'})
# a['metrics'] = 'AUROC'
# b = results[['name', 'AUPRC_tile', 'AUPRC_slide']]
# b = b.rename(columns={'AUPRC_tile': 'tile', 'AUPRC_slide': 'slide'})
# b['metrics'] = 'AUPRC'
# c = results[['name', 'accuracy_tile', 'accuracy_slide']]
# c = c.rename(columns={'accuracy_tile': 'tile', 'accuracy_slide': 'slide'})
# c['metrics'] = 'accuracy'
# d = results[['name', 'F1_tile', 'F1_slide']]
# d = d.rename(columns={'F1_tile': 'tile', 'F1_slide': 'slide'})
# d['metrics'] = 'F1'
#
# recombined = pd.concat([a, b, c, d])
# recombined = recombined.fillna(0)
# g = sns.FacetGrid(recombined, hue="name", col="metrics", height=4,
#                   margin_titles=True, palette=sns.color_palette("muted"))
# g.map(plt.scatter, "slide", "tile", edgecolor="white", s=15, lw=0.1)
# g.set(xlim=(0.5, 1), ylim=(0.5, 1))
# ax = g.axes.ravel()[0]
# ax.legend(fontsize='xx-small', ncol=13, loc='upper center', bbox_to_anchor=(2, -0.15),
#           fancybox=True, shadow=True)
# plt.show()