import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pca_data = pd.read_csv('clustering.eigenvec', sep=" ", header=None)
samples = pd.read_csv('pop_superpop.tsv', sep='\t')
ids = samples.iloc[:, 0].tolist()
new_df = pd.DataFrame(pca_data)
new_df.index = ids
final_df = new_df.iloc[:, [2, 3]]
final_df.to_csv('pca_data.csv', sep=',', index=True, encoding='utf-8', index_label='sample_id')