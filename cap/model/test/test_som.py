import os
from cap.model.test.template import SafeModelTester
from cap.model.som import SOMBase
from cap.settings import DFLT_SEED
from cap.settings import TEST_SEED
from cap.settings import DFLT_MAP_SIZE
from cap.settings import DFLT_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE


class TestSOMBase(SafeModelTester):

    def __init__(self, test_name):
        SafeModelTester.__init__(self, test_name)

    def setUp(self):
        self.test_class = 'SomBase'

#    def __create_model_instance(self):
#        model = SummarizeAnnovarModel()
#        return model

    def test_init_default(self):
        """ to check if SOMBase initialize correctly (default) """

        self.init_test(self.current_func_name)
        model = SOMBase(5)
        self.assertEqual(model.features_size,
                         5,
                         'Incorrect number of features')
        self.assertEqual(model.map_size,
                         DFLT_MAP_SIZE,
                         'Incorrect size of model mapping')
        self.assertEqual(model.step_size,
                         DFLT_STEP_SIZE,
                         'Incorrect step size')
        self.assertEqual(model.max_nbh_size,
                         DFLT_MAX_NBH_SIZE,
                         'Incorrect maximum number of neighborhood')
        self.assertTrue(model.random_seed is DFLT_SEED,
                        'Invalid random seed')

    def test_init_custom(self):
        """ to check if SOMBase initialize correctly (custom) """

        self.init_test(self.current_func_name)
        model = SOMBase(200,
                        step_size=0.3,
                        map_size=100,
                        max_nbh_size=70,
                        random_seed=TEST_SEED
                        )
        self.assertEqual(model.features_size,
                         200,
                         'Incorrect number of features')
        self.assertEqual(model.map_size,
                         100,
                         'Incorrect size of model mapping')
        self.assertEqual(model.step_size,
                         0.3,
                         'Incorrect step size')
        self.assertEqual(model.max_nbh_size,
                         70,
                         'Incorrect maximum number of neighborhood')
        self.assertEqual(model.random_seed,
                         20,
                         'Invalid random seed')

    def test_init_random_weight_map(self):
        """ to check if SOMBase initialize random weight_map correctly """

        self.init_test(self.current_func_name)
        model = SOMBase(5,
                        random_seed=TEST_SEED
                        )
        self.assertEqual(round(model.weight_map[0][1], 4),
                         0.8977,
                         'Invalid random weight map')

    def test_train(self):
        """ to see if SOMBase can correctly train testing samples """

        self.init_test(self.current_func_name)
        model = SOMBase(5,
                        random_seed=TEST_SEED
                        )
        model.train(None)
