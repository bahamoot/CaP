import os
import cap.plugin.animal
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

    def to_str(self, list_item):
        fmt = '{:<9}'
        if len(list_item) == 0:
            return fmt.format('.')
        else:
            return fmt.format(', '.join(list_item))

    def visualize_terminal(self, samples):
        out = []
        for i in xrange(self.map_rows):
            out_row = []
            for j in xrange(self.map_cols):
                out_row.append([])
            out.append(out_row)

        for sample in samples:
            winner, diff = self.calc_similarity(sample.props)
            row, col = self.to_grid(winner)
            out[row][col].append(sample.name)

        for row_items in out:
            line = " ".join(map(lambda x: self.to_str(x), row_items))
            print line
#        out_samples = sorted(samples, cmp=self.__compare)
#        for sample in out_samples:
#            print sample.name


def demo_toy_training():
    animals = cap.plugin.animal.load_animals()
    prop_size = len(animals[0].props)
    model = SOM2DAnimal(prop_size,
                        max_nbh_size=15,
#                        random_seed=TEST_SEED,
                        map_rows=20,
                        map_cols=20,
                        )
    model.train(animals)
    model.visualize_terminal(animals)


#def demo_toy_training():
#    animals = cap.plugin.animal.load_animals()
#    prop_size = len(animals[0].props)
#    model = SOMBase(prop_size,
#                    random_seed=TEST_SEED,
#                    max_nbh_size=50,
#                    map_size=100,
#                    )
#    model.train(animals)
#    model.visualize(animals)
