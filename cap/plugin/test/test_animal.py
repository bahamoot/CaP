import os
import cap.plugin.animal
from cap.plugin.test.template import SafePluginTester
from cap.plugin.animal import Animal
from cap.settings import DFLT_SEED
from cap.settings import TEST_SEED
from cap.settings import DFLT_MAP_SIZE
from cap.settings import DFLT_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE


class TestAnimal(SafePluginTester):

    def __init__(self, test_name):
        SafePluginTester.__init__(self, test_name)

    def setUp(self):
        self.test_class = 'Animal'

    def test_load_animals(self):
        """ to test if the toy sample is correctly loaded """

        self.init_test(self.current_func_name)

        animals = cap.plugin.animal.load_animals()

        self.assertEqual(len(animals),
                         32,
                         'Invalid number of animals')
        self.assertEqual(len(animals[0].props),
                         84,
                         'Invalid number of animal properties')
