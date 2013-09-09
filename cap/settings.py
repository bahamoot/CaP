import os

"""

This module is all about constant value that are used in this application

"""

# > > > > > > > > > > > > > random values < < < < < < < < < <
DFLT_SEED = None
DEMO_SEED = 20
TEST_SEED = DEMO_SEED

# > > > > > > > > > > > > > development files & folders < < < < < < < < < <
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

#SCRIPT_DIR = os.path.join(PROJECT_ROOT, 'script')
#WRAPPED_SUMMARIZE_ANNOVAR = os.path.join(SCRIPT_DIR, '/home/jessada/development/CMM/projects/linkage_analysis/script/wrapped_summarize_annovar')
#
#REF_DB_FILE_PREFIX = '/home/jessada/development/scilifelab/tools/annovar/humandb/hg19_snp137'
#CHR6_BEGIN_MARKER = 'rs1001015'
#CHR6_END_MARKER = 'rs3734693'
#CHR18_BEGIN_MARKER = 'rs1013785'
#CHR18_END_MARKER = 'rs1010800'
#CHR19_BEGIN_MARKER = 'rs8109631'
#CHR19_END_MARKER = 'rs1529958'
#UPPSALA_BWA_VCF_TABIX_FILE = '/home/jessada/development/CMM/master_data/CRC_screen4/bwa_GATK.vcf.gz'
#UPPSALA_MOSAIK_VCF_TABIX_FILE = '/home/jessada/development/CMM/master_data/realign/merged/Mosaik_Samtools.vcf.gz'
#AXEQ_VCF_TABIX_FILE = '/home/jessada/development/CMM/master_data/axeq/merged/Axeq.vcf.gz'
#GLOBAL_WORKING_DIR = '/home/jessada/development/CMM/projects/linkage_analysis/tmp'
#SA_OUT_DIR = '/home/jessada/development/CMM/projects/linkage_analysis/data/summarize_annovar'
#UPPSALA_FAMILY_FILE = '/home/jessada/development/CMM/projects/linkage_analysis/data/family/uppsala_family.txt'
#AXEQ_FAMILY_FILE = '/home/jessada/development/CMM/projects/linkage_analysis/data/family/axeq_family.txt'
#XLS_OUT_DIR = '/home/jessada/development/CMM/projects/linkage_analysis/xls_out'

# > > > > > > > > > > > > > model parameters < < < < < < < < < <
DFLT_MAP_SIZE = 4
DFLT_STEP_SIZE = 0.2
DFLT_MAX_NBH_SIZE = 3


