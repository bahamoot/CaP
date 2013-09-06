import linkana.settings as lka_const
from linkana.template import LinkAnaBase
from linkana.db.connectors import SummarizeAnnovarDB
from linkana.db.connectors import VcfDB
from linkana.db.connectors import FamilyDB
from linkana.db.connectors import ZYGOSITY_UNKNOWN
from linkana.db.connectors import ZYGOSITY_NONE
from linkana.settings import TYPE1_ALL
from linkana.settings import TYPE2_RECTAL
from linkana.settings import TYPE2_NON_RECTAL
from linkana.settings import TYPE3_COLON
from linkana.settings import TYPE3_NON_COLON
from linkana.settings import TYPE4_CAFAM
from linkana.settings import TYPE4_NON_CAFAM


class PatientRecord(object):
    """ to automatically parse VCF data"""

    def __init__(self):
        self.genotype_fields = {}
        self.patient_code = ''
        self.type2 = None
        self.type3 = None
        self.type4 = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.get_raw_repr())

    def get_raw_repr(self):
        return {"patient code": self.patient_code}


class AbstractSummarizeAnnovarDB(LinkAnaBase):
    """

    #1. an abstract connection to Summarize Annovar databases(not yet implemented)
    #2. able to handle many SummarizeAnnovarDB connectors(not yet implemented)
    3. a mutation annotation record can be accessed using mutation key

    """

    def __init__(self):
        LinkAnaBase.__init__(self)
        self.__connector = None
        self.__mutations = {}

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str({'Connector': self.__get_connector(),
                    })

    def __get_connector(self):
        return self.__connector

    def add_connector(self, summarize_annovar_db_connector):
        self.__connector = summarize_annovar_db_connector
        self.__need_update = True

    def __update_mutaitions_table(self):
        self.__mutations = {}
        #create table
        for record in self.__connector.records:
            self.__mutations[record.key] = record
        self.__need_update = False

    @property
    def mutations(self):
        if self.__need_update:
            self.__update_mutaitions_table()
        return self.__mutations


class AbstractVcfDB(LinkAnaBase):
    """

    1. an abstract connection to VCF databases
    2. able to handle many VcfDB connectors
    3. build up 2D mutations table (mutation, patient)
    4. provide accesses to the content of VcfDB
        - using mutation key
        - using patient code
    5. provide a filtering function
        - common mutations given patient code(s)

    """

    def __init__(self):
        LinkAnaBase.__init__(self)
        self.__connectors = []
        self.__mutations = {}

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str({'Connectors': self.__get_connectors(),
                    })

    def __get_connectors(self):
        return self.__connectors

    def add_connector(self, vcf_db_connector):
        self.__connectors.append(vcf_db_connector)
        self.__need_update = True

    def __update_mutaitions_table(self):
        """

        assume that all overlapped records from different VcfDBs
        have the same content

        """

        self.info("build VCF mutations table")
        self.__mutations = {}
        self.__patients = {}
        for connector in self.__connectors:
            header = connector.header
            #init patients content
            for patient_code in header.patient_codes:
                if patient_code not in self.__patients:
                    self.__patients[patient_code] = PatientRecord()
                    self.__patients[patient_code].patient_code = patient_code
            #create table
            for record in connector.records:
                mutation_genotype_fields = {}
                for i in xrange(len(header.patient_codes)):
                    patient_code = header.patient_codes[i]
                    genotype_fields = record.genotype_fields[i]
                    #add pointer to patient record(column)
                    genotype_fields.patient = self.__patients[patient_code]
                    #add pointer to mutation record(row)
                    genotype_fields.mutation = record
                    #give mutation an access to genotype field using patient code as a key
                    mutation_genotype_fields[patient_code] = genotype_fields
                    #give patient an access to genotype field using mutaion key as a key
                    self.__patients[patient_code].genotype_fields[record.key] = genotype_fields
                record.genotype_fields = mutation_genotype_fields
                self.__mutations[record.key] = record
        self.__need_update = False

    def common_mutations(self, patient_codes, exom_only=False):
        """

        return dict of mutations that are found in all patient
        given patient codes

        """

        common_mutations = {}
        for mutation_key in self.mutations:
            mutation = self.mutations[mutation_key]
            common_mutation = True
            print mutation_key
            for patient_code in patient_codes:
                zygosity = mutation.genotype_fields[patient_code].zygosity
                print mutation.genotype_fields[patient_code]
                if zygosity == ZYGOSITY_UNKNOWN:
                    common_mutation = False
                    break
                if zygosity == ZYGOSITY_NONE:
                    common_mutation = False
                    break
            if common_mutation:
                common_mutations[mutation_key] = mutation
        return common_mutations


    @property
    def patients(self):
        if self.__need_update:
            self.__update_mutaitions_table()
        return self.__patients

    @property
    def mutations(self):
        if self.__need_update:
            self.__update_mutaitions_table()
        return self.__mutations


class AbstractFamilyDB(LinkAnaBase):
    """

    #1. an abstract connection to Family databases(not yet implemented)
    #2. able to handle many FamilyDB connectors(not yet implemented)
    3. a family (members) record can be accessed by family code

    """

    def __init__(self):
        LinkAnaBase.__init__(self)
        self.__connector = None
        self.__families = {}
        self.__group_members_count = {}

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str({'Connectors': self.__get_connector(),
                    })

    def __get_connector(self):
        return self.__connector

    def add_connector(self, family_db_connector):
        self.__connector = family_db_connector
        self.__need_update = True

    def __update_group_members_count(self):
        group_members_count = {}
        group_members_count[TYPE1_ALL] = 0
        group_members_count[TYPE2_RECTAL] = 0
        group_members_count[TYPE2_NON_RECTAL] = 0
        group_members_count[TYPE3_COLON] = 0
        group_members_count[TYPE3_NON_COLON] = 0
        group_members_count[TYPE4_CAFAM] = 0
        group_members_count[TYPE4_NON_CAFAM] = 0
        for record in self.__connector.records:
            if record.type2 == TYPE2_RECTAL:
                group_members_count[TYPE2_RECTAL] += len(record.patient_codes)
            else:
                group_members_count[TYPE2_NON_RECTAL] += len(record.patient_codes)
            if record.type3 == TYPE3_COLON:
                group_members_count[TYPE3_COLON] += len(record.patient_codes)
            else:
                group_members_count[TYPE3_NON_COLON] += len(record.patient_codes)
            if record.type4 == TYPE4_CAFAM:
                group_members_count[TYPE4_CAFAM] += len(record.patient_codes)
            else:
                group_members_count[TYPE4_NON_CAFAM] += len(record.patient_codes)
            group_members_count[TYPE1_ALL] += len(record.patient_codes)
        self.__group_members_count = group_members_count
        self.__need_update = False

    def __update_families_table(self):
        self.__families = {}
        #create table
        for record in self.__connector.records:
            self.__families[record.family_code] = record
        self.__need_update = False

    @property
    def group_members_count(self):
        if self.__need_update:
            self.__update_families_table()
            self.__update_group_members_count()
        return self.__group_members_count

    @property
    def families(self):
        if self.__need_update:
            self.__update_families_table()
            self.__update_group_members_count()
        return self.__families


class DBManager(LinkAnaBase):
    """

    1. to handle all the connection to all databases
    2. to provide the simplest interface to downstream classes
        - provide abstract connection for each db type

    """

    def __init__(self):
        LinkAnaBase.__init__(self)
        self.__abs_sa_db = AbstractSummarizeAnnovarDB()
        self.__abs_vcf_db = AbstractVcfDB()
        self.__abs_fam_db = AbstractFamilyDB()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str({'SummarizeAnnovarDB': self.__get_summarize_annovar_db_connection(),
                    'VcfDB': self.__get_vcf_db_connection(),
                    'FamilyDB': self.__get_family_db_connection(),
                    })

    def __get_summarize_annovar_db_connection(self):
        return self.__abs_sa_db

    def __get_vcf_db_connection(self):
        return self.__abs_vcf_db

    def __get_family_db_connection(self):
        return self.__abs_fam_db

    def __connect_summarize_annovar_db(self, csv_file, delimiter='\t'):
        self.info("create summarize-annovar db connection to " + csv_file)
        sa_db = SummarizeAnnovarDB()
        sa_db.open_db(csv_file, delimiter)
        self.__abs_sa_db.add_connector(sa_db)

    def connect_summarize_annovar_db(self, csv_file, delimiter='\t'):
        return self.__connect_summarize_annovar_db(csv_file, delimiter)

    def __connect_vcf_db(self, vcf_db_gz_file, chrom, begin_pos, end_pos):
        self.info("create vcf db connection to " + vcf_db_gz_file)
        vcf_db = VcfDB()
        vcf_db.open_db(vcf_db_gz_file, chrom, begin_pos, end_pos)
        self.__abs_vcf_db.add_connector(vcf_db)

    def connect_vcf_db(self, vcf_db_gz_file, chrom, begin_pos, end_pos):
        return self.__connect_vcf_db(vcf_db_gz_file, chrom, begin_pos, end_pos)

    def __connect_family_db(self, family_db_file):
        self.info("create family db connection to " + family_db_file)
        fam_db = FamilyDB()
        fam_db.open_db(family_db_file)
        self.__abs_fam_db.add_connector(fam_db)

    def connect_family_db(self, family_db_file):
        return self.__connect_family_db(family_db_file)

    @property
    def summarize_annovar_db(self):
        return self.__abs_sa_db

    @property
    def vcf_db(self):
        return self.__abs_vcf_db

    @property
    def family_db(self):
        return self.__abs_fam_db

    @property
    def valid_patient_codes(self):
        """

        to check if members that appear in family file are also in
        Vcf database

        """

        vcf_patients = self.vcf_db.patients
        families = self.family_db.families

        for family_code in families:
            for patient_code in families[family_code].patient_codes:
                if patient_code not in vcf_patients:
                    self.info("patient code " + patient_code + " not found")
                    return False
        return True
