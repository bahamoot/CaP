import unittest
import os
from cap.template import SafeTester
from cap.template import RiskyTester


class SafeToyTester(SafeTester):
    """ General template for safe "Toy" modules testing """

    def __init__(self, test_name):
        SafeTester.__init__(self, test_name)

    def set_dir(self):
        self.working_dir = os.path.join(os.path.join(os.path.join(os.path.dirname(__file__),
                                                                  'tmp'),
                                                     self.test_class),
                                        self.test_function)
        self.data_dir = os.path.join(os.path.join(os.path.dirname(__file__),
                                                  'data'),
                                     self.test_class)


class RiskyToyTester(RiskyTester):
    """ General template for risky "Toy" modules testing """

    def __init__(self, test_name):
        RiskyTester.__init__(self, test_name)

    def set_dir(self):
        self.working_dir = cap_settings.cap_WORKING_DIR
        self.data_dir = os.path.join(os.path.join(os.path.dirname(__file__),
                                                  'big_data'),
                                     self.test_class)
