import as pd
pop_data=pd.read_csv('sample_pop.tsv', sep='\t')

#dictionary where population are mapped to their superpopulations
superpop = {'ACB':'AFR','ASW':'AFR','ESN':'AFR','GWD':'AFR','LWK':'AFR','MSL':'AFR',
            'YRI':'AFR','CLM':'AMR','MXL':'AMR','PEL':'AMR','PUR':'AMR',
            'CDX':'EAS','CHB':'EAS','CHS':'EAS','JPT':'EAS','KHV':'EAS',
            'GBR':'EUR', 'FIN':'EUR','IBS':'EUR','TSI':'EUR','CEU':'EUR','SIB':'EUR',
            'BEB':'SAS','GIH':'SAS','ITU':'SAS','PJL':'SAS','STU':'SAS'}

#map samples to their superpopoulation
pop_data['superpopulation'] = pop_data['population'].map(superpop) 

#save final dataframe as a csv file
pop_data.to_csv('pop_superpop.tsv', sep='\t', index=False, header=True)
