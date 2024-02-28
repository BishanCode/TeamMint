import pandas as pd

pop = pd.read_csv("pop_superpop.tsv", sep="\t")
#get population labels
pop_name = pop.population.tolist()
pop_name = list(dict.fromkeys(pop_name))

acb = pd.read_csv("plink2.ACB.afreq", sep="\t")
#get snp ids
snp_id = acb.ID.tolist()
final_df = pd.DataFrame({"ID": snp_id})

#for each population file get the allele frequencies and merge them into one dataframe
for pop in pop_name:
    filename = f"allele_freq/plink2.{pop}.afreq"
    pop_df = pd.read_csv(filename, sep="\t")
    allele_freq = pop_df.ALT_FREQS.tolist()
    final_df[f"{pop}"] = allele_freq
    
final_df.to_csv("allele_freq_df.csv", index=False)