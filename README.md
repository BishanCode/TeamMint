# Team Mint application 
This web application provides a user-friendly interface to explore the SNPs found on chromosome 1 in 3929 samples from different populations around the world. Specifically, it allows users to perform two population genetic structure analyses on the data set provided; clustering using Principal Component Analysis and Admixture analysis. In these two analyses the user is able to select which populations or superpopulations to analyse each time. Additionally, the user can use a search feature to retrieve allele and genotype frequencies as well as other relevant information, such as clinical information for the SNPs of interest in the populations of interest. The user also has the option to search for genes of interest in the populations he chooses and all SNPs associated with those genes along with the relevant information will be shown. This can also be done by choosing a region of interest on chromosome 1. If multiple populations are selected in the latter three analyses an Fst matrix is automatically created showing how much the selected populations differ and the user has the option to download the matrix as a text file to be used in further analyses. 
## Requirements
These are the installations required to run the web application:
- Flask
- pandas
- mySQL.connector
- NumPy
- Matplotlib & Matplotlib.pyplot
- io & BytesIO
- Base64
- wtforms
- wtforms.validators
- Itertools
<!-- end of the list -->
If you wish to recreate the SQL database these additional packages are required:
- PyMySQL
- Zarr
- Scikit-allel
- Numcodecs
<!-- end of the list -->
## Installation
- Use python 64bit (v3.11.7 64bit).
- Download the ‘Website’ folder and ‘main.py’. Run main.py using your choice of Python IDE or open up a terminal and type cd /path/to/Website and python main.py. Ensure MYSQL 8.0.36 database is created via downloading all the files in “Make the database” and running make_database.ipynb or if you want to start from scratch to download all the files in “Initial Data Processing”.
- For Mac ensure on the top where the imports matplotlib.use(“AGG”) is coded.
