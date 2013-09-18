import os
import cap.plugin.toy.animal
import cap.plugin.toy.extra_animal
import matplotlib.pyplot as plt
from cap.model.som import SOMBase
from cap.model.som import SOM2D
from cap.settings import TEST_SEED
from cap.settings import DFLT_SEED
from cap.settings import DFLT_MAP_SIZE
from cap.settings import DFLT_WEIGHT_STEP_SIZE
from cap.settings import DFLT_NBH_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE
from cap.settings import DFLT_MAP_ROWS
from cap.settings import DFLT_MAP_COLS


class SOM2DAnimal(SOM2D):

    def __init__(self,
                 props_size,
                 map_rows=DFLT_MAP_ROWS,
                 map_cols=DFLT_MAP_COLS,
                 weight_step_size=DFLT_WEIGHT_STEP_SIZE,
                 nbh_step_size=DFLT_NBH_STEP_SIZE,
                 max_nbh_size=DFLT_MAX_NBH_SIZE,
                 random_seed=DFLT_SEED,
                 ):
        SOM2D.__init__(self,
                         props_size,
                         map_rows=map_rows,
                         map_cols=map_cols,
                         weight_step_size=weight_step_size,
                         nbh_step_size=nbh_step_size,
                         max_nbh_size=max_nbh_size,
                         random_seed=random_seed,
                         )

    def get_raw_repr(self):
        return {"props size": self.props_size,
                "map rows": self.map_rows,
                "map columns": self.map_cols,
                "step size": self.step_size,
                "max neighborhod size": self.max_nbh_size,
                "random seed": self.random_seed,
                }

#    def to_str(self, list_item):
#        fmt = '{:<11}'
#        if len(list_item) == 0:
#            return fmt.format('.')
#        else:
#            return fmt.format(', '.join(list_item))


def demo_toy_training():
    animals = cap.plugin.toy.animal.load_animals()
    extra_animals = cap.plugin.toy.extra_animal.load_animals()
    features_size = len(animals[0].features)
    model = SOM2DAnimal(features_size,
                        max_nbh_size=9,
                        nbh_step_size=0.3,
                        map_rows=17,
                        map_cols=17,
                        )
    model.train(animals)
    model.visualize_terminal(animals, test_samples=[extra_animals[8]])
    model.visualize_plt(animals,
                        29,
                        class_plt_style={0: 'r^',
                                         1: 'b*',
                                         },
                        test_samples=[extra_animals[8]],
                        )
