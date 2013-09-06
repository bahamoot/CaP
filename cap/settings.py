import os

"""

This module is all about constant value that are used in this application

"""

# > > > > > > > > > > > > > development files & folders < < < < < < < < < <
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

SCRIPT_DIR = os.path.join(PROJECT_ROOT, 'script')
WRAPPED_SUMMARIZE_ANNOVAR = os.path.join(SCRIPT_DIR, '/home/jessada/development/CMM/projects/linkage_analysis/script/wrapped_summarize_annovar')

REF_DB_FILE_PREFIX = '/home/jessada/development/scilifelab/tools/annovar/humandb/hg19_snp137'
CHR6_BEGIN_MARKER = 'rs1001015'
CHR6_END_MARKER = 'rs3734693'
CHR18_BEGIN_MARKER = 'rs1013785'
CHR18_END_MARKER = 'rs1010800'
CHR19_BEGIN_MARKER = 'rs8109631'
CHR19_END_MARKER = 'rs1529958'
UPPSALA_BWA_VCF_TABIX_FILE = '/home/jessada/development/CMM/master_data/CRC_screen4/bwa_GATK.vcf.gz'
UPPSALA_MOSAIK_VCF_TABIX_FILE = '/home/jessada/development/CMM/master_data/realign/merged/Mosaik_Samtools.vcf.gz'
AXEQ_VCF_TABIX_FILE = '/home/jessada/development/CMM/master_data/axeq/merged/Axeq.vcf.gz'
GLOBAL_WORKING_DIR = '/home/jessada/development/CMM/projects/linkage_analysis/tmp'
SA_OUT_DIR = '/home/jessada/development/CMM/projects/linkage_analysis/data/summarize_annovar'
UPPSALA_FAMILY_FILE = '/home/jessada/development/CMM/projects/linkage_analysis/data/family/uppsala_family.txt'
AXEQ_FAMILY_FILE = '/home/jessada/development/CMM/projects/linkage_analysis/data/family/axeq_family.txt'
XLS_OUT_DIR = '/home/jessada/development/CMM/projects/linkage_analysis/xls_out'

# > > > > > > > > > > > > > patient groups < < < < < < < < < <
TYPE1_ALL = 'ALL'
TYPE2_RECTAL = 'RECTAL'
TYPE2_NON_RECTAL = 'NON_RECTAL'
TYPE3_COLON = 'COLON'
TYPE3_NON_COLON = 'NON_COLON'
TYPE4_CAFAM = 'CAFAM'
TYPE4_NON_CAFAM = 'NON_CAFAM'


## > > > > > > > > > > > > > development files & folders < < < < < < < < < <
#PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
#
## > > > > > > > > > > > > > CSV annotation format configuration < < < < < < < < < <
#
##annotation index
##0-based index, used by python
#SNP_0_IDX_GENE_NAME   = 1
#SNP_0_IDX_EXONIC_FUNC = 2
#SNP_0_IDX_MAF         = 7
#SNP_0_IDX_MARKER      = 8
#SNP_0_IDX_CHROM       = 21
#SNP_0_IDX_START_POS   = 22
#SNP_0_IDX_END_POS     = 23
#SNP_0_IDX_REF         = 24
#SNP_0_IDX_ALT         = 25
#
#
###'central' data for testing and for demo
##CBV_SAMPLE_DATA_ROOT   = os.path.join(PROJECT_ROOT, 'combivep/data')
##CBV_SAMPLE_DATASET_DIR = os.path.join(CBV_SAMPLE_DATA_ROOT, 'dataset')
##CBV_SAMPLE_CBV_DIR     = os.path.join(CBV_SAMPLE_DATA_ROOT, 'CBV')
##CBV_SAMPLE_VCF_DIR     = os.path.join(CBV_SAMPLE_DATA_ROOT, 'VCF')
##CBV_SAMPLE_UCSC_DIR    = os.path.join(CBV_SAMPLE_DATA_ROOT, 'UCSC')
##CBV_SAMPLE_LJB_DIR     = os.path.join(CBV_SAMPLE_DATA_ROOT, 'LJB')
##CBV_SAMPLE_CFG_FILE    = os.path.join(CBV_SAMPLE_DATA_ROOT, 'config.txt')
##CBV_SAMPLE_PARAM_DIR   = os.path.join(CBV_SAMPLE_DATA_ROOT, 'params')
##CBV_SAMPLE_PARAM_FILE  = os.path.join(CBV_SAMPLE_PARAM_DIR, 'params.npz')
##
##
### > > > > > > > > > > > > > User files & folders < < < < < < < < < <
##USER_DATA_ROOT = os.path.expanduser('~/.CombiVEP')
##
###to keep user data produced by CombiVEP engine
##USER_PARAMS_DIR  = os.path.join(USER_DATA_ROOT, 'params')
##USER_PARAMS_FILE = os.path.join(USER_PARAMS_DIR, 'params.npz')
##
###to keep the reference database from UCSC and LJB
##USER_UCSC_REF_DB_DIR = os.path.join(USER_DATA_ROOT, 'ref/UCSC')
##USER_LJB_REF_DB_DIR  = os.path.join(USER_DATA_ROOT, 'ref/LJB')
##
##
