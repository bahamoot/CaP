import os
import cap.plugin.base
import matplotlib.pyplot as plt
import datetime
from cap.model.som import SOM2D
from cap.settings import DFLT_WEIGHT_STEP_SIZE
from cap.settings import DFLT_NBH_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE
from cap.settings import DFLT_MAP_ROWS
from cap.settings import DFLT_MAP_COLS
from cap.settings import DFLT_SEED
from cap.settings import PROJECT_ROOT
from cap.settings import FIGS_TMP_OUT_DIR
from cap.settings import TERM_TMP_OUT_DIR


ROOT_DEMO_DATA = '/home/jessada/development/scilifelab/projects/CaP/cap/data/'
DEMO_TRAINING_FEATURES = os.path.join(ROOT_DEMO_DATA,
                                      'demo_training_features.txt')
DEMO_TRAINING_CLASSES = os.path.join(ROOT_DEMO_DATA,
                                     'demo_training_classes.txt')
DEMO_TEST_FEATURES = os.path.join(ROOT_DEMO_DATA,
                                  'demo_test_features.txt')
DEMO_TEST_CLASSES = os.path.join(ROOT_DEMO_DATA,
                                 'demo_test_classes.txt')
PARADIGM_WEIGHT_STEP_SIZE = 0.2
PARADIGM_NBH_STEP_SIZE = 10
PARADIGM_MAX_NBH_SIZE = 8
PARADIGM_MAP_ROWS = 10
PARADIGM_MAP_COLS = 10
PARADIGM_RANDOM_SEED = 156806204
#PARADIGM_RANDOM_SEED = None

def get_time_stamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

#Run som2d using Paradigm configuration and using MSI status as
#classification criteria
def som2d_paradigm_msi_status():
    out = som2d_paradigm(DEMO_TRAINING_FEATURES,
                         DEMO_TRAINING_CLASSES,
                         'MSI_status',
                         class_plt_style={'MSI-H': 'r^',
                                          'MSI-L': 'b*',
                                          'MSS': 'mo',
                                          },
                         test_features_file=DEMO_TEST_FEATURES,
                         )
    return out["figure name"]

#Run som2d using Paradigm configuration and using tumor stage as
#classification criteria
def som2d_paradigm_tumor_stage():
    out = som2d_paradigm(DEMO_TRAINING_FEATURES,
                         DEMO_TRAINING_CLASSES,
                         'tumor_stage',
                         class_plt_style={'Stage I': 'r^',
                                          'Stage IIA': 'b*',
                                          'Stage IIB': 'yD',
                                          'Stage IIIA': 'mH',
                                          'Stage IIIB': 'co',
                                          'Stage IIIC': 'gv',
                                          'Stage IV': 'mx',
                                          },
                         test_features_file=DEMO_TEST_FEATURES,
                         )
    return out["figure name"]

#A wrapped layer to call som2d using Paradigm configuration
def som2d_paradigm(training_features_file,
                   training_classes_file,
                   group_criteria,
                   class_plt_style={},
                   test_features_file=None,
                   figure_name=None,
                   terminal_file=None,
                   ):
    current_time = get_time_stamp()
    figure_name = os.path.join(FIGS_TMP_OUT_DIR,
                               'fig_'+group_criteria+'_'+current_time+'.eps',
                               )
    terminal_file = os.path.join(TERM_TMP_OUT_DIR,
                                 'term_'+group_criteria+'_'+current_time+'.txt',
                                 )
    return som2d(training_features_file,
                 training_classes_file,
                 group_criteria,
                 class_plt_style=class_plt_style,
                 figure_name=figure_name,
                 terminal_file=terminal_file,
                 test_features_file=test_features_file,
                 map_rows=PARADIGM_MAP_ROWS,
                 map_cols=PARADIGM_MAP_COLS,
                 weight_step_size=PARADIGM_WEIGHT_STEP_SIZE,
                 nbh_step_size=PARADIGM_NBH_STEP_SIZE,
                 max_nbh_size=PARADIGM_MAX_NBH_SIZE,
                 random_seed=PARADIGM_RANDOM_SEED,
                 )

#wrap all require stuffs to run SOM2D
def som2d(training_features_file,
          training_classes_file,
          group_criteria,
          class_plt_style={},
          test_features_file=None,
          figure_name=None,
          terminal_file=None,
          map_rows=DFLT_MAP_ROWS,
          map_cols=DFLT_MAP_COLS,
          weight_step_size=DFLT_WEIGHT_STEP_SIZE,
          nbh_step_size=DFLT_NBH_STEP_SIZE,
          max_nbh_size=DFLT_MAX_NBH_SIZE,
          random_seed=DFLT_SEED,
          ):
    training_samples = cap.plugin.base.load_samples(training_features_file,
                                                    training_classes_file)
    if test_features_file is not None:
        test_samples = cap.plugin.base.load_samples(test_features_file)
    else:
        test_samples = None

    features_size = len(training_samples[0].features)
    model = SOM2D(features_size,
                  map_rows=map_rows,
                  map_cols=map_cols,
                  weight_step_size=weight_step_size,
                  nbh_step_size=nbh_step_size,
                  max_nbh_size=max_nbh_size,
                  random_seed=random_seed,
                  )
    #shorten training samples name
    for training_sample in training_samples:
        training_sample.name = training_sample.name.replace("TCGA-", "")
    #train and visualize
    model.train(training_samples)
    out_terminal = model.visualize_terminal(training_samples,
                                            terminal_str_width=15,
                                            output_file=terminal_file,
                                            test_samples=test_samples,
                                            )
    out_plt = model.visualize_plt(training_samples,
                                  group_criteria,
                                  class_plt_style,
                                  figure_name=figure_name,
                                  test_samples=test_samples,
                                  )
    return {"figure name": out_plt,
            "terminal file": out_terminal,
            }
