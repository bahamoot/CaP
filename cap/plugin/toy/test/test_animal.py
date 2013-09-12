import os
import cap.plugin.toy.animal
from cap.plugin.toy.test.template import SafeToyTester
from cap.settings import DFLT_SEED
from cap.settings import TEST_SEED
from cap.settings import DFLT_MAP_SIZE
from cap.settings import DFLT_WEIGHT_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE


class TestAnimal(SafeToyTester):

    def __init__(self, test_name):
        SafeToyTester.__init__(self, test_name)

    def setUp(self):
        self.test_class = 'Animal'

    def test_load_animals(self):
        """ to test if the toy sample is correctly loaded """

        self.init_test(self.current_func_name)

        animals = cap.plugin.toy.animal.load_animals()

        self.assertEqual(len(animals),
                         32,
                         'Invalid number of animals')
        self.assertEqual(len(animals[0].props),
                         84,
                         'Invalid number of animal properties')
