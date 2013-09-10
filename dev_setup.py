from setuptools import setup
import sys
import glob
import pkgutil


setup(
    name='CaP_dev',
    version='0.1.1',
    author='Jessada Thutkawkorapin',
    author_email='jessada.thutkawkorapin@gmail.com',
    packages=['cap',
              'cap.test',
              'cap.model',
              'cap.model.test',
              'cap.plugin',
              'cap.plugin.test',
              'cap.devtools',
              'cap.devtools.test',
              ],
    scripts=['bin/CaP_demo_toy_training',
             ],
    package=['CaP'],
    package_data={},
#    package_data={'': ['data/CBV/*.cbv'],
#                  '': ['data/CBV/*.scores'],
#                  '': ['data/CBV/*.clean'],
#                  },
    data_files=[],
#    data_files=[('cap/data/CBV', ['cap/data/CBV/training.cbv',
#                                       'cap/data/CBV/test.cbv',
#                                       'cap/data/CBV/training.cbv.clean',
#                                       'cap/data/CBV/test.cbv.clean',
#                                       'cap/data/CBV/training.cbv.scores',
#                                       'cap/data/CBV/test.cbv.scores',
#                                       ]),
#                ],
    url='http://pypi.python.org/pypi/cap_dev/',
    license='LICENSE.txt',
    description='Cancer Predictor',
    long_description=open('README.md').read(),
    install_requires=[],
#    install_requires=["pysam >= 0.7"],
)