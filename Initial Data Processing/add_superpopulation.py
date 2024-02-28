import as pd
pop_data=pd.read_csv('sample_pop.tsv', sep='\t')
superpop = {'ACB':'AFR','ASW':'AFR','ESN':'AFR','GWD':'AFR','LWK':'AFR','MSL':'AFR',
            'YRI':'AFR','CLM':'AMR','MXL':'AMR','PEL':'AMR','PUR':'AMR',
            'CDX':'EAS','CHB':'EAS','CHS':'EAS','JPT':'EAS','KHV':'EAS',
            'GBR':'EUR', 'FIN':'EUR','IBS':'EUR','TSI':'EUR','CEU':'EUR','SIB':'EUR',
            'BEB':'SAS','GIH':'SAS','ITU':'SAS','PJL':'SAS','STU':'SAS'}

pop_data['superpopulation'] = pop_data['population'].map(superpop) 
print (pop_data) 
pop_data.to_csv('pop_superpop.tsv', sep='\t', index=False, header=True)