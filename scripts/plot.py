"""
plot results

Created on 04/13/2020

@author: RH
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results = pd.read_csv('../Summary.csv', header=0)
results['name'] = results['model']+'_'+results['state'].str.get(0)

# All in one
a = results[['name', 'AUROC_image']]
a['metrics'] = 'AUROC_image'
a = a.rename(columns={'AUROC_image': 'value'})
b = results[['name', 'AUPRC_image']]
b['metrics'] = 'AUPRC_image'
b = b.rename(columns={'AUPRC_image': 'value'})
c = results[['name', 'accuracy_image']]
c['metrics'] = 'accuracy_image'
c = c.rename(columns={'accuracy_image': 'value'})
d = results[['name', 'AUROC_patient']]
d['metrics'] = 'AUROC_patient'
d = d.rename(columns={'AUROC_patient': 'value'})
e = results[['name', 'AUPRC_patient']]
e['metrics'] = 'AUPRC_patient'
e = e.rename(columns={'AUPRC_patient': 'value'})
f = results[['name', 'accuracy_patient']]
f['metrics'] = 'accuracy_patient'
f = f.rename(columns={'accuracy_patient': 'value'})

recombined = pd.concat([a, d, b, e, c, f])

grid = sns.catplot(x="name", y="value", hue="metrics", kind="bar", data=recombined, height=5, aspect=4)
grid.set_xticklabels(rotation=45, horizontalalignment='right', fontsize='medium', fontweight='light')
grid.set(ylim=(0.45, 1))
plt.show()

# # Scatter plots
# a = results[['name', 'AUROC_image', 'AUROC_patient']]
# a = a.rename(columns={'AUROC_image': 'image', 'AUROC_patient': 'patient'})
# a['metrics'] = 'AUROC'
# b = results[['name', 'AUPRC_image', 'AUPRC_patient']]
# b = b.rename(columns={'AUPRC_image': 'image', 'AUPRC_patient': 'patient'})
# b['metrics'] = 'AUPRC'
# c = results[['name', 'accuracy_image', 'accuracy_patient']]
# c = c.rename(columns={'accuracy_image': 'image', 'accuracy_patient': 'patient'})
# c['metrics'] = 'accuracy'
# d = results[['name', 'F1_image', 'F1_patient']]
# d = d.rename(columns={'F1_image': 'image', 'F1_patient': 'patient'})
# d['metrics'] = 'F1'
#
# recombined = pd.concat([a, b, c, d])
# recombined = recombined.fillna(0)
# g = sns.FacetGrid(recombined, hue="name", col="metrics", height=4,
#                   margin_titles=True, palette=sns.color_palette("muted"))
# g.map(plt.scatter, "patient", "image", edgecolor="white", s=15, lw=0.1)
# g.set(xlim=(0.5, 1), ylim=(0.5, 1))
# ax = g.axes.ravel()[0]
# ax.legend(fontsize='xx-small', ncol=13, loc='upper center', bbox_to_anchor=(2, -0.15),
#           fancybox=True, shadow=True)
# plt.show()