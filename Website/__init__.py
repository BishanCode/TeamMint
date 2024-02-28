import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import mysql.connector
from flask import Flask, render_template, redirect, url_for, request, send_file
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, SelectMultipleField, StringField
from wtforms.validators import Optional, NumberRange
import matplotlib
matplotlib.use("AGG")
import numpy as np
from itertools import combinations

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'Teammint123@'

    # Database connection
    connection = mysql.connector.connect(
        user="root",
        password="Teammint123@",
        host='localhost',
        database='final'
    )

    # Define color maps for populations and superpopulations
    population_colours = {
        'ACB': 'red',
        'ASW': 'blue',
        'ESN': 'green',
        'GWD': 'orange',
        'LWK': 'purple',
        'MSL': 'cyan',
        'YRI': 'magenta',
        'CLM': 'gold',
        'MXL': 'lightblue',
        'PEL': 'lightgreen',
        'PUR': 'pink',
        'CDX': 'lightcoral',
        'CHB': 'lightskyblue',
        'CHS': 'lightseagreen',
        'JPT': 'lightsalmon',
        'KHV': 'lime',
        'GBR': 'lightpink',
        'FIN': 'lightcyan',
        'IBS': 'lightgreen',
        'TSI': 'lightgrey',
        'CEU': 'darkred',
        'SIB': 'darkblue',
        'BEB': 'darkgreen',
        'GIH': 'darkorange',
        'ITU': 'darkviolet',
        'PJL': 'darkcyan',
        'STU': 'darkmagenta',
    }

    superpopulation_colours = {
        'AFR': 'red',
        'AMR': 'blue',
        'EAS': 'green',
        'EUR': 'orange',
        'SAS': 'purple',
    }
 # Function to perform PCA and generate plot based on selected populations
    def generate_pca_plot(selected_populations, population=True):
        print("Generating PCA plot for selected populations:", selected_populations)
        # Determine the color map based on the selected populations
        colour_map = population_colours if population else superpopulation_colours

        # Query Data
        query = """
                SELECT pop_superpop.sample_id, pca_data.pc1, pca_data.pc2, pop_superpop.population, pop_superpop.superpopulation
                FROM pop_superpop
                INNER JOIN pca_data ON pop_superpop.sample_id = pca_data.sample_id
                """
        data = pd.read_sql_query(query, connection)
        print("Data loaded from database:", data.head())

        # Filter Data based on user selection
        if population == True:
            filtered_data = data[data['population'].isin(selected_populations)]
        else:
            filtered_data = data[data['superpopulation'].isin(selected_populations)]
        print("Filtered data based on selected populations:", filtered_data.head())
        
        # Plot PCA
        print("plotting")
        plt.figure(figsize=(8, 6))
        print(selected_populations)
        for pop in selected_populations:
            print("Filtering data based on population:", pop)
            if population == True:
                population_data = filtered_data[filtered_data['population'] == pop]
            else:
                population_data = filtered_data[filtered_data['superpopulation'] == pop]
            print("Population data for", pop, ":", population_data.head())
            plt.scatter(population_data['pc1'], population_data['pc2'],
                        label=pop, color=colour_map.get(pop, 'black'))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Plot')
        plt.legend()

        # Save plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        # Encode the plot to base64
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return plot_data

    # Route for the home page
    @app.route('/')
    def index():
        populations = list(population_colours.keys())
        return render_template('home.html', populations=populations)

    # Route for generating and displaying the PCA plot based on selected populations
    @app.route('/plot/population', methods=['POST'])
    def plot_population():
        selected_populations = request.form.getlist('populations')
        print("Selected populations for population plot:", selected_populations)
        plot_data = generate_pca_plot(selected_populations, population=True)
        return render_template('clustering_population_plot.html', plot_data=plot_data)

    # Route for generating and displaying the PCA plot based on selected superpopulations
    @app.route('/plot/superpopulation', methods=['POST'])
    def plot_superpopulation():
        selected_superpopulations = request.form.getlist('superpopulations')
        print("Selected superpopulations for superpopulation plot:", selected_superpopulations)
        plot_data = generate_pca_plot(selected_superpopulations, population=False)
        return render_template('clustering_superpopulation_plot.html', plot_data=plot_data)
    
    @app.route('/admixture')
    def admixture():
        return render_template('admixture.html')
    
    @app.route('/allele_genotype_frequency')
    def allele_genotype_frequency():
        return render_template('allele_genotype_frequency.html')
    
    @app.route('/clustering')
    def clustering():
        return render_template('clustering.html')
    
    @app.route('/clustering/clustering_superpopulation')
    def clustering_superpopulation():
        superpopulations = list(superpopulation_colours.keys()) 
        return render_template('clustering_superpopulation.html', superpopulations=superpopulations)
    
    @app.route('/clustering/clustering_population')
    def clustering_population():
        populations = list(population_colours.keys()) 
        return render_template('clustering_population.html', populations=populations)
    
    # Route for generating and displaying the PCA plot based on selected superpopulations
    @app.route('/clustering/clustering_superpopulation_plot', methods=['POST'])
    def clustering_superpopulation_plot():
        selected_superpopulations = request.form.getlist('superpopulations')
        #print("Selected superpopulations:", selected_superpopulations)  # Add this print statement
        plot_data = generate_pca_plot(selected_superpopulations, population=False)
        return render_template('clustering_superpopulation_plot.html', plot_data=plot_data)
    
    # Function to perform PCA and generate plot
    def generate_admixture_plot(selected_populations):
        # Step 1: Query Data
        query = """
                SELECT pop_superpop.sample_id, admixture_data.pop1, admixture_data.pop2,
                    admixture_data.pop3, admixture_data.pop4, admixture_data.pop5,
                    pop_superpop.population, pop_superpop.superpopulation
                FROM pop_superpop
                INNER JOIN admixture_data ON pop_superpop.sample_id = admixture_data.sample_id
                WHERE pop_superpop.population IN ({})
                """.format(', '.join(['%s'] * len(selected_populations)))

        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, selected_populations)
        data = pd.DataFrame(cursor.fetchall())

        # Step 2: Plotting
        num_populations = len(selected_populations)
        fig, axes = plt.subplots(num_populations, 1, figsize=(25, 5*num_populations))

        for i, population in enumerate(selected_populations):
            population_data = data[data['population'] == population]

            pal = ['red', 'blue', 'green', 'orange', 'purple']
            ax = population_data.plot.bar(ax=axes[i], stacked=True, color=pal, width=1,
                                        fontsize='x-small', edgecolor='black', linewidth=0.5)

            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_xlabel('')  # Remove x-axis label
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium', labelspacing=0.5, frameon=False)
 
            ax.set_title(f'Admixture Plot for Population: {population}')
            ax.set_xlabel('Individual')
            ax.set_ylabel('Ancestry')

        plt.tight_layout()
        
        # Convert plot to image
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return plot_url
# Admixture for population
    @app.route('/admixture_home_pop', methods=['GET', 'POST'])
    def admixture_pop():
        populations = ['ACB', 'ASW', 'ESN', 'GWD', 'LWK', 'MSL', 'YRI', 'CLM', 'MXL', 'PEL', 'PUR', 'CDX', 'CHB',
                    'CHS', 'JPT', 'KHV', 'GBR', 'FIN', 'IBS', 'TSI', 'CEU', 'SIB', 'BEB', 'GIH', 'ITU', 'PJL',
                    'STU']
        return render_template('admixture_home_pop.html', populations=populations)

    @app.route('/admixture_pop_plot', methods=['POST'])
    def admixture_pop_plot():
        selected_populations = request.form.getlist('superpopulations')
        plot_data = generate_admixture_plot(selected_populations)
        return render_template('admixture_pop_plot.html', plot_data=plot_data)


    def generate_admixture_plot_super(selected_populations):
        plot_urls = []

        for superpopulation in selected_populations:
            # Query Data
            query = """
                    SELECT pop_superpop.sample_id, admixture_data.pop1, admixture_data.pop2,
                        admixture_data.pop3, admixture_data.pop4, admixture_data.pop5,
                        pop_superpop.superpopulation
                    FROM pop_superpop
                    INNER JOIN admixture_data ON pop_superpop.sample_id = admixture_data.sample_id
                    WHERE pop_superpop.superpopulation = '{}'
                    """.format(superpopulation)

            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            data = pd.DataFrame(cursor.fetchall())

            # Plotting
            pal = ['red', 'blue', 'green', 'orange', 'purple']

            fig, ax = plt.subplots(figsize=(25, 5))
            data.plot.bar(stacked=True,
                        ax=ax,
                        color=pal,
                        width=1,
                        fontsize='x-small',
                        edgecolor='black',
                        linewidth=0.5)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Set x-axis labels to sample IDs
            sample_ids = data['sample_id'].unique()
            ax.set_xticks([])
            ax.set_xlabel('') 

            # Place the legend slightly to the right of the graph
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium', labelspacing=0.5, frameon=False)
            # Center the title
            ax.set_title(f'Admixture Plot for Superpopulation: {superpopulation}', fontsize='large', ha='center')

            # Set Y-axis label
            ax.set_ylabel('Ancestry')
            ax.set_xlabel('individual')

            # Convert plot to image
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')  # Use bbox_inches='tight' to prevent cropping of title
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plot_urls.append(plot_url)

            plt.close()  # Close the plot to avoid overlapping

        return plot_urls

# Admixture Superpopulation
    @app.route('/admixture_home_superpop', methods=['GET', 'POST'])
    def admixture_superpop():
        superpopulations = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']
        return render_template('admixture_home_superpop.html', populations=superpopulations)

    @app.route('/admixture_superpop_plot', methods=['POST'])
    def admixture_superpop_plot():
        selected_populations = request.form.getlist('superpopulations')
        plot_data = generate_admixture_plot_super(selected_populations)
        superpopulations = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']  # Define your list of superpopulations here
        return render_template('admixture_superpop_plot.html', plot_data=plot_data, populations=superpopulations)
    
    class SNPForm(FlaskForm):
        start_position = IntegerField('Start Position:', validators=[Optional(), NumberRange(min=12782, max=248936926)])
        end_position = IntegerField('End Position:', validators=[Optional(), NumberRange(min=12782, max=248936926)])
        populations = SelectMultipleField('Select populations to include:', coerce=str)
        submit = SubmitField('Submit')

    @app.route('/position_search', methods=['GET', 'POST'])
    def position_search():
        form = SNPForm()
        form.populations.choices = get_population_labels()
        if form.validate_on_submit():
            start_position = form.start_position.data
            end_position = form.end_position.data
            selected_populations = form.populations.data
            return redirect(url_for('search_snp', start_position=start_position, end_position=end_position, populations=selected_populations))
        return render_template('snp_position.html', form=form)

    @app.route('/search_snp', methods=['GET', 'POST'])
    def search_snp():
        start_position = request.args.get('start_position', type=int)
        end_position = request.args.get('end_position', type=int)
        selected_populations = request.args.getlist('populations')

        if start_position is None or end_position is None:
            return "Please enter both start and end positions."

        if start_position >= end_position:
            return "End position must be greater than start position."

        pop_list = []  
        alt_freq_values = [[] for _ in range(len(selected_populations))]  

        try:
            cursor = connection.cursor(dictionary=True)

            snp_data = []
            query = f"SELECT snp_id, ref_base, alt_base, disease_name, classification FROM snp_char WHERE position BETWEEN {start_position} AND {end_position}"
            cursor.execute(query)
            for snp_char_row in cursor.fetchall():
                snp_id = snp_char_row['snp_id']
                allele_freq_data = {}
                for i, population in enumerate(selected_populations):
                    query = f"SELECT {population} FROM allele_freq WHERE snp_id = '{snp_id}'"
                    cursor.execute(query)
                    frequency = cursor.fetchone()
                    if frequency:
                        allele_freq_data[population] = frequency[population]
                        alt_freq_values[i].append(frequency[population])

                ref_allele_freq_data  = {population: 1 - freq for population, freq in allele_freq_data.items()}
                genotype_freq_data = {}
                for population in selected_populations:
                    alt_freq = allele_freq_data.get(population, 0)
                    ref_freq = ref_allele_freq_data.get(population, 0)
                    homo_alt_freq = alt_freq ** 2
                    homo_ref_freq = ref_freq ** 2
                    hetero_freq = 2 * ref_freq * alt_freq
                    genotype_freq_data[population] = {'Homozygous Alt': homo_alt_freq, 'Homozygous Ref': homo_ref_freq, 'Heterozygous': hetero_freq}

                snp_char_data = {
                    'ref_allele': snp_char_row['ref_base'],
                    'alt_allele': snp_char_row['alt_base'],
                    'disease': snp_char_row['disease_name'],
                    'classification': snp_char_row['classification']
                }

                snp_data.append({'id_name': snp_id, 'snp_char_data': snp_char_data, 'allele_freq_data': allele_freq_data, 'ref_allele_freq_data': ref_allele_freq_data, 'genotype_freq_data': genotype_freq_data})

            if not snp_data:
                return "No data found within this range."

            fst_matrix = calculate_fst(alt_freq_values)
            plot_url = visualize_fst(fst_matrix, selected_populations)
            
            cursor.close()

            single_population = len(selected_populations) == 1  
            
            # Create a DataFrame from the Fst matrix data
            fst_matrix_df = pd.DataFrame(fst_matrix, columns=selected_populations, index=selected_populations)

            # Save the DataFrame to a text file with tab-separated values (TSV)
            fst_matrix_df.to_csv('fst_matrix.txt', sep='\t')

            # Render the template and send the response
            return render_template('search_snp_result.html', snp_data=snp_data, pop_list=pop_list, alt_freq_values=alt_freq_values, snp_char_data=snp_char_data, fst_matrix_image=plot_url, single_population=single_population)

        except mysql.connector.Error as error:
            return f"Error while connecting to MySQL: {error}"
        except UnboundLocalError:
            return "No data found within this range."

    def calculate_fst(alt_freq_values):
        num_populations = len(alt_freq_values)
        num_snps = len(alt_freq_values[0])  

        fst_matrix = np.zeros((num_populations, num_populations))

        for i, j in combinations(range(num_populations), 2):
            sum_squared_diff = 0.0
            for snp in range(num_snps):
                pi_total = 0
                for freq_list in alt_freq_values:
                    pi_total += freq_list[snp]
                pi_total /= num_populations
                
                # Check if pi_total is non-zero before performing the division
                if pi_total != 0:
                    sum_squared_diff = 0.0
                    for freq in alt_freq_values[i][snp], alt_freq_values[j][snp]:
                        pi_within = freq ** 2
                        sum_squared_diff += 2 * (pi_total - pi_within)
                        
                    fst = sum_squared_diff / (num_snps * pi_total)
                    fst_matrix[i][j] = fst
                    fst_matrix[j][i] = fst  

        return fst_matrix


    def visualize_fst(fst_matrix, pop_list):
        plt.figure(figsize=(8, 6))
        plt.imshow(fst_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Fst')
        plt.xticks(np.arange(len(pop_list)), pop_list, rotation=45)
        plt.yticks(np.arange(len(pop_list)), pop_list)
        plt.title('Pairwise Fst Matrix')
        plt.xlabel('Population')
        plt.ylabel('Population')
        plt.tight_layout()
        for i in range(len(pop_list)):
            for j in range(len(pop_list)):
                plt.text(j, i, f'{fst_matrix[i, j]:.2f}', ha='center', va='center', color='white')   
        img = BytesIO() 
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()  
        
        return plot_url

    def get_population_labels():
        try:
            cursor = connection.cursor()

            cursor.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'allele_freq'")
            column_names = cursor.fetchall()

            population_labels = []
            for column_name_tuple in column_names:
                column_name = column_name_tuple[0]
                if column_name != 'snp_id':
                    population_labels.append((column_name, column_name.replace('_', ' ')))

            cursor.close()

            return population_labels

        except mysql.connector.Error as error:
            print(f"Error while fetching population labels from MySQL: {error}")
            return []

    @app.route('/download_results')
    def download_results():
        # Load the Fst matrix data
        # Assuming you have already calculated and saved it as 'fst_matrix.txt'
        try:
            fst_matrix_df = pd.read_csv('fst_matrix.txt', sep='\t', index_col=0)
        except FileNotFoundError:
            return "Fst matrix file not found.", 404

        # Send the file as a downloadable attachment (Will need to change the dir)
        return send_file('../fst_matrix.txt', as_attachment=True, mimetype='text/plain')
    
    class IDForm(FlaskForm):
        id_names = StringField('Enter SNP ids or RSids (comma-separated):', validators=[Optional()])
        populations = SelectMultipleField('Select populations to include:', coerce=str)
        submit = SubmitField('Submit')

    @app.route('/ID_search', methods=['GET', 'POST'])
    def ID_search():
        form = IDForm()
        form.populations.choices = get_population_labels()
        if form.validate_on_submit():
            id_names = form.id_names.data.split(',')
            selected_populations = form.populations.data
            return redirect(url_for('ID_snp', id_names=','.join(id_names), populations=selected_populations))
        return render_template('snp_id.html', form=form)

    @app.route('/ID_snp', methods=['GET', 'POST'])
    def ID_snp():
        id_names = request.args.get('id_names')
        selected_populations = request.args.getlist('populations')

        if id_names is not None:
            id_names = id_names.split(',')
        else:
            id_names = []

        pop_list = []  
        alt_freq_values = [[] for _ in range(len(selected_populations))]  

        try:
            # Modified connection with direct parameters
            connection = mysql.connector.connect(
                user="root",
                password="Teammint123@",
                host='localhost',
                database='final'
            )
            cursor = connection.cursor(dictionary=True)

            snp_data = []
            for id_name in id_names:
                allele_freq_data = {}
                for i, population in enumerate(selected_populations):
                    query = f"SELECT {population} FROM allele_freq WHERE snp_id = '{id_name}'"
                    cursor.execute(query)
                    frequency = cursor.fetchone()
                    if frequency:
                        allele_freq_data[population] = frequency[population]
                        alt_freq_values[i].append(frequency[population])

                ref_allele_freq_data = {population: 1 - freq for population, freq in allele_freq_data.items()}
                genotype_freq_data = {}
                for population in selected_populations:
                    alt_freq = allele_freq_data.get(population, 0)
                    ref_freq = ref_allele_freq_data.get(population, 0)
                    homo_alt_freq = alt_freq ** 2
                    homo_ref_freq = ref_freq ** 2
                    hetero_freq = 2 * ref_freq * alt_freq
                    genotype_freq_data[population] = {'Homozygous Alt': homo_alt_freq, 'Homozygous Ref': homo_ref_freq, 'Heterozygous': hetero_freq}

                snp_char_data = {}
                snp_char_query = f"SELECT ref_base, alt_base, disease_name, classification FROM snp_char WHERE snp_id = '{id_name}'"
                cursor.execute(snp_char_query)
                snp_char_row = cursor.fetchone()
                if snp_char_row:
                    snp_char_data['ref_allele'] = snp_char_row['ref_base']
                    snp_char_data['alt_allele'] = snp_char_row['alt_base']
                    snp_char_data['disease'] = snp_char_row['disease_name']
                    snp_char_data['classification'] = snp_char_row['classification']

                snp_data.append({'id_name': id_name, 'allele_freq_data': allele_freq_data, 'ref_allele_freq_data': ref_allele_freq_data, 'genotype_freq_data': genotype_freq_data, 'snp_char_data': snp_char_data})
            
            if not snp_data:  # If no data found for any SNP ID
                return render_template('search_snp_result.html', warning_message="No data found for the entered SNP ID(s). Please check the ID(s) and try again.")

            fst_matrix = calculate_fst(alt_freq_values)
            plot_url = visualize_fst(fst_matrix, selected_populations)
            
            cursor.close()
            connection.close()

            single_population = len(selected_populations) == 1  
            fst_matrix_df = pd.DataFrame(fst_matrix, columns=selected_populations, index=selected_populations)
            fst_matrix_df.to_csv('fst_matrix.txt', sep='\t')
            return render_template('search_snp_result.html', snp_data=snp_data, pop_list=pop_list, alt_freq_values=alt_freq_values, snp_char_data=snp_char_data, fst_matrix_image=plot_url, single_population=single_population)

        except mysql.connector.Error as error:
            return f"Error while connecting to MySQL: {error}"

    class GeneNameForm(FlaskForm):  # Changed from SNPForm to GeneNameForm
        gene_names = StringField('Enter gene names (comma-separated):', validators=[Optional()])
        populations = SelectMultipleField('Select populations to include:', coerce=str)
        submit = SubmitField('Submit')

    @app.route('/GeneName_search', methods=['GET', 'POST'])  # Changed from /snp_search to /GeneName_search
    def GeneName_search():  # Changed from snp_search to GeneName_search
        form = GeneNameForm()  # Changed from SNPForm to GeneNameForm
        form.populations.choices = get_population_labels()
        if form.validate_on_submit():
            gene_names = form.gene_names.data.split(',')
            selected_populations = form.populations.data
            return redirect(url_for('GeneName_snp', gene_names=','.join(gene_names), populations=selected_populations))  # Changed from search_snp to GeneName_snp
        return render_template('gene_name.html', form=form)

    @app.route('/GeneName_snp', methods=['GET', 'POST'])  # Changed from /search_snp to /GeneName_snp
    def GeneName_snp():  # Changed from search_snp to GeneName_snp
        gene_names = request.args.get('gene_names')
        selected_populations = request.args.getlist('populations')

        if gene_names is not None:
            gene_names = gene_names.split(',')
        else:
            gene_names = []

        id_names = get_snp_ids_from_gene_names(gene_names)

        pop_list = []  
        alt_freq_values = [[] for _ in range(len(selected_populations))]  

        try:
            cursor = connection.cursor(dictionary=True)

            snp_data = []
            for id_name in id_names:
                allele_freq_data = {}
                for i, population in enumerate(selected_populations):
                    query = f"SELECT {population} FROM allele_freq WHERE snp_id = '{id_name}'"
                    cursor.execute(query)
                    frequency = cursor.fetchone()
                    if frequency:
                        allele_freq_data[population] = frequency[population]
                        alt_freq_values[i].append(frequency[population])

                ref_allele_freq_data = {population: 1 - freq for population, freq in allele_freq_data.items()}
                genotype_freq_data = {}
                for population in selected_populations:
                    alt_freq = allele_freq_data.get(population, 0)
                    ref_freq = ref_allele_freq_data.get(population, 0)
                    homo_alt_freq = alt_freq ** 2
                    homo_ref_freq = ref_freq ** 2
                    hetero_freq = 2 * ref_freq * alt_freq
                    genotype_freq_data[population] = {'Homozygous Alt': homo_alt_freq, 'Homozygous Ref': homo_ref_freq, 'Heterozygous': hetero_freq}

                snp_char_data = {}
                snp_char_query = f"SELECT ref_base, alt_base, disease_name, classification FROM snp_char WHERE snp_id = '{id_name}'"
                cursor.execute(snp_char_query)
                snp_char_row = cursor.fetchone()
                if snp_char_row:
                    snp_char_data['ref_allele'] = snp_char_row['ref_base']
                    snp_char_data['alt_allele'] = snp_char_row['alt_base']
                    snp_char_data['disease'] = snp_char_row['disease_name']
                    snp_char_data['classification'] = snp_char_row['classification']

                snp_data.append({'id_name': id_name, 'allele_freq_data': allele_freq_data, 'ref_allele_freq_data': ref_allele_freq_data, 'genotype_freq_data': genotype_freq_data, 'snp_char_data': snp_char_data})
            
            fst_matrix = calculate_fst(alt_freq_values)
            plot_url = visualize_fst(fst_matrix, selected_populations)
            
            cursor.close()

            single_population = len(selected_populations) == 1  

            # Create a DataFrame from the Fst matrix data
            fst_matrix_df = pd.DataFrame(fst_matrix, columns=selected_populations, index=selected_populations)

            # Save the DataFrame to a text file with tab-separated values (TSV)
            fst_matrix_df.to_csv('fst_matrix.txt', sep='\t')
            return render_template('search_snp_result.html', snp_data=snp_data, pop_list=pop_list, alt_freq_values=alt_freq_values, snp_char_data=snp_char_data, fst_matrix_image=plot_url, single_population=single_population)

        except mysql.connector.Error as error:
            return f"Error while connecting to MySQL: {error}"
        except UnboundLocalError:
            return "No data found for the provided gene name."  
        
    def get_snp_ids_from_gene_names(gene_names):
        try:
            cursor = connection.cursor()

            snp_ids = []

            for gene_name in gene_names:
                query = f"SELECT snp_id FROM snp_char WHERE position IN (SELECT position FROM gene_names WHERE gene_name = '{gene_name}')"
                cursor.execute(query)
                snp_ids.extend([result[0] for result in cursor.fetchall()])

            cursor.close()

            return snp_ids

        except mysql.connector.Error as error:
            print(f"Error while fetching SNP IDs from MySQL: {error}")
            
            return []
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
    
