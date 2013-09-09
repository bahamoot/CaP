import numpy as np
from cap.template import CaPBase
from cap.settings import DFLT_SEED
from cap.settings import DFLT_MAP_SIZE
from cap.settings import DFLT_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE


class SOMBase(CaPBase):
    """ to automatically parse VCF data"""

    def __init__(self,
                 features_size,
                 map_size=DFLT_MAP_SIZE,
                 step_size=DFLT_STEP_SIZE,
                 max_nbh_size=DFLT_MAX_NBH_SIZE,
                 random_seed=DFLT_SEED,
                 ):
        CaPBase.__init__(self)
        self.__features_size = features_size
        self.__map_size = map_size
        self.__step_size = step_size
        self.__max_nbh_size = max_nbh_size
        self.__random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.__weight_map = np.random.rand(self.map_size, self.features_size)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<SOMBase Object> ' + str(self.get_raw_repr())

    def get_raw_repr(self):
        return {"features size": self.features_size,
                "map size": self.map_size,
                "step size": self.step_size,
                "max neighborhod size": self.max_nbh_size,
                "random seed": self.random_seed,
                }

    @property
    def features_size(self):
        return self.__features_size

    @property
    def map_size(self):
        return self.__map_size

    @property
    def step_size(self):
        return self.__step_size

    @property
    def random_seed(self):
        return self.__random_seed

    @property
    def max_nbh_size(self):
        return self.__max_nbh_size

    @property
    def weight_map(self):
        return self.__weight_map

    def train(self, samples):
        for nbh in xrange(self.max_nbh_size):
            print nbh
#            for animal in a.animals:
#                diff = a.props[animal] - weight
#                dist = np.sum(diff*diff, axis=1)
#                winner = np.argmin(dist)
#        
#                min_neighbor = int(winner - update_size)
#                if min_neighbor < 0:
#                    min_neighbor = 0
#                max_neighbor = int(winner + update_size)
#                if max_neighbor > map_rows:
#                    max_neighbor = map_rows
#                for i in xrange(min_neighbor, max_neighbor):
#                    weight[i] += diff[i] * 0.2



