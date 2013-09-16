import os
import cap.plugin.base
import matplotlib.pyplot as plt
from cap.model.som import SOM2D
from cap.settings import DFLT_WEIGHT_STEP_SIZE
from cap.settings import DFLT_NBH_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE
from cap.settings import DFLT_MAP_ROWS
from cap.settings import DFLT_MAP_COLS
from cap.settings import DFLT_SEED
from cap.settings import PROJECT_ROOT

DEMO_TRAINING_FEATURES = '/home/jessada/development/scilifelab/projects/CaP/cap/data/demo_training_features.txt'
DEMO_TRAINING_CLASSES = '/home/jessada/development/scilifelab/projects/CaP/cap/data/demo_training_classes.txt'
DEMO_TEST_FEATURES = os.path.join('/home/jessada/development/scilifelab/projects/CaP/',
                                  '/cap/data/demo_training_features.txt')


def demo_SOM2D_training():
    SOM2D_training(DEMO_TRAINING_FEATURES, DEMO_TRAINING_CLASSES)

def SOM2D_training(training_features_file,
                   training_classes_file=None,
                   test_features_file=None,
                   map_rows=DFLT_MAP_ROWS,
                   map_cols=DFLT_MAP_COLS,
                   weight_step_size=DFLT_WEIGHT_STEP_SIZE,
                   nbh_step_size=DFLT_NBH_STEP_SIZE,
                   max_nbh_size=DFLT_MAX_NBH_SIZE,
                   random_seed=DFLT_SEED,
                   ):
    training_samples = cap.plugin.base.load_samples(training_features_file,
                                                    training_classes_file)
    features_size = len(training_samples[0].features)
    model = SOM2D(features_size,
                  map_rows=map_rows,
                  map_cols=map_cols,
                  weight_step_size=weight_step_size,
                  nbh_step_size=nbh_step_size,
                  max_nbh_size=max_nbh_size,
                  random_seed=random_seed,
                  )
    model.train(training_samples)
    model.visualize_terminal(training_samples)
    model.visualize_plt(training_samples,
                        class_name='tumor_stage',
                        class_plt_style={'Stage I': 'r^',
                                         'Stage IIA': 'b*',
                                         'Stage IIB': 'yD',
                                         'Stage IIIA': 'mH',
                                         'Stage IIIB': 'co',
                                         'Stage IIIC': 'gv',
                                         'Stage IV': 'mx',
#                        class_plt_style={'Stage I': 'r^',
#                                         'Stage IIA': 'b*',
#                                         'Stage IIB': 'b*',
#                                         'Stage IIIA': 'co',
#                                         'Stage IIIB': 'co',
#                                         'Stage IIIC': 'co',
#                                         'Stage IV': 'mx',
                                         }
                        )
