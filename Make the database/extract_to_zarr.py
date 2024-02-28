# This py file allows to convert vcf.gz file to zarr arrays, this will be very useful
# when data are too large to fit into main memory, zarr arrays can provide fast on-disk
# storage and retrieval of numerical arrays.
import allele
import numcodecs
import sys

# Specify the path here if needed 
vcf_path = 'annotated.vcf.gz'
zarr_path = 'annotated.zarr'
clinvar_vcf_path = 'clinvar.vcf.gz'
clinvar_zarr_path = 'clinvar.zarr'

# Change the vcf_path, zarr_path to clinvar_vcf_path and clinvar_zarr_path
# when convert clinvar.vcf.gz to zarr file.
allel.vcf_to_zarr(vcf_path, zarr_path, group='1',
                  fields='*', alt_number=1, log=sys.stdout,
                  compressor=numcodecs.Blosc(cname='zstd', clevel=1, shuffle=False))


