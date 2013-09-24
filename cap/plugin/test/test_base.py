import os
import cap.plugin.base
from cap.plugin.base import SamplesLoader
from cap.settings import TYPE_TRAINING_SAMPLE
from cap.settings import TYPE_TEST_SAMPLE
from cap.plugin.test.template import SafePluginTester


class TestMisc(SafePluginTester):
    """ to test misc stuffs in base.py espectially 'load_base' """

    def __init__(self, test_name):
        SafePluginTester.__init__(self, test_name)

    def setUp(self):
        self.test_class = 'Misc'

    def test_load_samples(self):
        """ to test if the base samples are correctly loaded """

        self.init_test(self.current_func_name)

        test_features_file = os.path.join(self.data_dir,
                                          self.current_func_name + '_features.txt')
        test_classes_file = os.path.join(self.data_dir,
                                         self.current_func_name + '_classes.txt')
        test_samples = cap.plugin.base.load_samples(test_features_file,
                                                    test_classes_file,
                                                    samples_type=TYPE_TEST_SAMPLE,
                                                    )
        test_sample_idx = None
        for i in xrange(len(test_samples)):
            if test_samples[i].name == 'TCGA-AA-3672':
                test_sample_idx = i
        self.assertTrue(test_sample_idx is not None,
                         'Invalid sample index')
        test_sample = test_samples[test_sample_idx]
        self.assertEqual(test_sample.features[8],
                         1.51486,
                         "Invalid sample's feature")
        self.assertEqual(test_sample.features[9],
                         0,
                         "Invalid sample's feature")
        self.assertEqual(test_sample.features[11],
                         -1.92865e-16,
                         "Invalid sample's feature")
        self.assertEqual(test_sample.classes['days_to_last_known_alive'],
                         '28',
                         "Invalid sample's class")
        self.assertEqual(test_sample.classes['tumor_site'],
                         '2 - transverse colon',
                         "Invalid sample's class")
        self.assertEqual(test_sample.classes['tumor_stage'],
                         'Stage IIIB',
                         "Invalid sample's class")
        self.assertEqual(len(test_samples),
                         10,
                         'Invalid number of samples')
        self.assertEqual(len(test_sample.features),
                         12,
                         'Invalid number of features')
        self.assertEqual(len(test_sample.classes),
                         44,
                         'Invalid number of classes')
        self.assertEqual(test_sample.sample_type,
                         TYPE_TEST_SAMPLE,
                         'Invalid sample type')


class TestSamplesLoader(SafePluginTester):
    """ to test base class """

    def __init__(self, test_name):
        SafePluginTester.__init__(self, test_name)

    def setUp(self):
        self.test_class = 'SamplesLoader'

    def test_get_samples_hash(self):
        """ to test if the base hash samples are correctly loaded """

        self.init_test(self.current_func_name)

        test_features_file = os.path.join(self.data_dir,
                                          self.current_func_name + '_features.txt')
        test_classes_file = os.path.join(self.data_dir,
                                         self.current_func_name + '_classes.txt')
        sl = SamplesLoader(features_file=test_features_file,
                           classes_file=test_classes_file,
                           samples_type=TYPE_TEST_SAMPLE,
                           )
        test_samples = sl.get_samples_hash()
        self.assertEqual(test_samples['TCGA-AG-A02X'].name,
                         'TCGA-AG-A02X',
                         'Invalid sample name')
        self.assertEqual(test_samples['TCGA-AG-A02X'].features[8],
                         2.87636,
                         "Invalid sample's feature")
        self.assertEqual(test_samples['TCGA-AG-A02X'].features[11],
                         -2.89298e-14,
                         "Invalid sample's feature")
        self.assertEqual(test_samples['TCGA-AA-3532'].features[9],
                         0.227783,
                         "Invalid sample's feature")
        self.assertEqual(test_samples['TCGA-AG-A01L'].classes['days_to_last_known_alive'],
                         '365',
                         "Invalid sample's class")
        self.assertEqual(test_samples['TCGA-AA-3672'].classes['tumor_site'],
                         '2 - transverse colon',
                         "Invalid sample's class")
        self.assertEqual(test_samples['TCGA-AA-3532'].classes['tumor_stage'],
                         'Stage IIA',
                         "Invalid sample's class")
        self.assertEqual(len(test_samples),
                         10,
                         'Invalid number of samples')
        self.assertEqual(len(test_samples['TCGA-AA-3532'].features),
                         12,
                         'Invalid number of features')
        self.assertEqual(len(test_samples['TCGA-AA-3532'].classes),
                         44,
                         'Invalid number of classes')
        self.assertEqual(test_samples['TCGA-AA-3532'].sample_type,
                         TYPE_TEST_SAMPLE,
                         'Invalid sample type')

    def test_get_samples_list(self):
        """ to test if the base list samples are correctly loaded """

        self.init_test(self.current_func_name)

        test_features_file = os.path.join(self.data_dir,
                                          self.current_func_name + '_features.txt')
        test_classes_file = os.path.join(self.data_dir,
                                         self.current_func_name + '_classes.txt')
        sl = SamplesLoader(features_file=test_features_file,
                           classes_file=test_classes_file,
                           )
        test_samples = sl.get_samples_list()
        test_sample_idx = None
        for i in xrange(len(test_samples)):
            if test_samples[i].name == 'TCGA-AA-3672':
                test_sample_idx = i
        self.assertTrue(test_sample_idx is not None,
                         'Invalid sample index')
        test_sample = test_samples[test_sample_idx]
        self.assertEqual(test_sample.features[8],
                         1.51486,
                         "Invalid sample's feature")
        self.assertEqual(test_sample.features[9],
                         0,
                         "Invalid sample's feature")
        self.assertEqual(test_sample.features[11],
                         -1.92865e-16,
                         "Invalid sample's feature")
        self.assertEqual(test_sample.classes['days_to_last_known_alive'],
                         '28',
                         "Invalid sample's class")
        self.assertEqual(test_sample.classes['tumor_site'],
                         '2 - transverse colon',
                         "Invalid sample's class")
        self.assertEqual(test_sample.classes['tumor_stage'],
                         'Stage IIIB',
                         "Invalid sample's class")
        self.assertEqual(len(test_samples),
                         10,
                         'Invalid number of samples')
        self.assertEqual(len(test_sample.features),
                         12,
                         'Invalid number of features')
        self.assertEqual(len(test_sample.classes),
                         44,
                         'Invalid number of classes')
        self.assertEqual(test_sample.sample_type,
                         TYPE_TRAINING_SAMPLE,
                         'Invalid sample type')
