import pandas as pd
import seaborn as sns

df_q = pd.read_csv('annotatedfiles.5.Q', sep=' ', header=None)
# automatically generate column names based on number of columns in Q (pop1, pop2, pop3, etc.)
names = ["pop{}".format(i) for i in range(1, df_q.shape[1]+1)]

# add column names to dataframe
df_q.columns = names

samples = pd.read_csv('pop_superpop.tsv', sep='\t')
ids = samples.iloc[:, 0].tolist()

#set the dataframe index to the sample names 
new_df = pd.DataFrame(df_q)
new_df.index = ids

#assign each individual to a population, based on highest proportion of ancestry
new_df['assignment'] = new_df.idxmax(axis=1)

#sort populations
def sort_df_by_pops(df):
    temp_dfs = []
    for pop in sorted(df['assignment'].unique()):
        temp = df.loc[df['assignment'] == pop].sort_values(by=[pop], ascending=False)
        temp_dfs.append(temp)
    return pd.concat(temp_dfs)

df_sorted_q = sort_df_by_pops(new_df)

df_sorted_q.to_csv('df_sorted_q.csv', sep=',', index=True, encoding='utf-8', index_label='sample_id')