import numpy as np
import math
import matplotlib.pyplot as plt
from cap.template import CaPBase
from cap.settings import DFLT_SEED
from cap.settings import DFLT_MAP_SIZE
from cap.settings import DFLT_WEIGHT_STEP_SIZE
from cap.settings import DFLT_NBH_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE
from cap.settings import DFLT_MAP_ROWS
from cap.settings import DFLT_MAP_COLS


DEFAULT_TRAINING_CLASS_STYLE = 'kp'
DEFAULT_TEST_CLASS_STYLE = 'ki+'

class SOMBase(CaPBase):
    """ to automatically parse VCF data"""

    def __init__(self,
                 features_size,
                 map_size=DFLT_MAP_SIZE,
                 weight_step_size=DFLT_WEIGHT_STEP_SIZE,
                 nbh_step_size=DFLT_NBH_STEP_SIZE,
                 max_nbh_size=DFLT_MAX_NBH_SIZE,
                 random_seed=DFLT_SEED,
                 ):
        CaPBase.__init__(self)
        self.__features_size = features_size
        self.__map_size = map_size
        self.__weight_step_size = weight_step_size
        self.__nbh_step_size = nbh_step_size
        self.__max_nbh_size = max_nbh_size
        self.__random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.weight_map = np.random.rand(self.map_size, self.features_size)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<' + self.__class__.__name__ + ' Object> ' + str(self.get_raw_repr())

    def get_raw_repr(self):
        return {"features size": self.features_size,
                "map size": self.map_size,
                "weight step size": self.weight_step_size,
                "neighborhood step size": self.nbh_step_size,
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
    def weight_step_size(self):
        return self.__weight_step_size

    @property
    def nbh_step_size(self):
        return self.__nbh_step_size

    @property
    def random_seed(self):
        return self.__random_seed

    @property
    def max_nbh_size(self):
        return self.__max_nbh_size

    def calc_similarity(self, features):
        diff = features - self.weight_map
        dist = np.sum(diff*diff, axis=1)
        winner = np.argmin(dist)
        return winner, diff

    def nbhs(self, winner, nbh):
        for i in xrange(winner-nbh, winner+nbh+1):
            if (i<0) or (i >= self.map_size):
                continue
            yield i

    def nbh_range(self):
        val = self.max_nbh_size
        while val >= 0:
            yield val
            val -= self.nbh_step_size

    def train(self, samples):
        for nbh in self.nbh_range():
            for sample in samples:
                winner, diff = self.calc_similarity(sample.features)
                #update winner and neighbors
                for i in self.nbhs(winner, nbh):
                    self.weight_map[i] += diff[i] * self.weight_step_size


    def __compare(self, x, y):
        order = self.__order
        if order[x.name] < order[y.name]:
            return -1
        if order[x.name] > order[y.name]:
            return 1
        return 0

    def visualize(self, samples):
        self.__order = {}
        for sample in samples:
            winner, diff = self.calc_similarity(sample.features)

            self.__order[sample.name] = winner

        out_samples = sorted(samples, cmp=self.__compare)
        for sample in out_samples:
            print sample.name


class SOM2D(SOMBase):

    def __init__(self,
                 features_size,
                 map_rows=DFLT_MAP_ROWS,
                 map_cols=DFLT_MAP_COLS,
                 weight_step_size=DFLT_WEIGHT_STEP_SIZE,
                 nbh_step_size=DFLT_NBH_STEP_SIZE,
                 max_nbh_size=DFLT_MAX_NBH_SIZE,
                 random_seed=DFLT_SEED,
                 ):
        SOMBase.__init__(self,
                         features_size,
                         map_size=map_rows*map_cols,
                         weight_step_size=weight_step_size,
                         nbh_step_size=nbh_step_size,
                         max_nbh_size=max_nbh_size,
                         random_seed=random_seed,
                         )
        self.__map_rows = map_rows
        self.__map_cols = map_cols

    def get_raw_repr(self):
        return {"features size": self.features_size,
                "map rows": self.map_rows,
                "map columns": self.map_cols,
                "step size": self.step_size,
                "max neighborhod size": self.max_nbh_size,
                "random seed": self.random_seed,
                }

    @property
    def map_rows(self):
        return self.__map_rows

    @property
    def map_cols(self):
        return self.__map_cols

    def to_grid(self, idx):
        return (idx%self.map_rows, idx//self.map_rows)

    def from_grid(self, row, col):
        return row + self.map_rows*col

    def nbhs(self, winner, nbh):
        row, col = self.to_grid(winner)
        ceil_nbh = int(math.ceil(nbh))
        for i in xrange(row-ceil_nbh, row+ceil_nbh+1):
            if (i<0) or (i >= self.map_rows):
                continue
            for j in xrange(col-ceil_nbh, col+ceil_nbh+1):
                if (j<0) or (j >= self.map_cols):
                    continue
                if ((i-row)**2 + (j-col)**2) > nbh**2:
                    continue
                yield self.from_grid(i, j)

    def to_str(self, list_item):
        fmt = '{:<11}'
        if len(list_item) == 0:
            return fmt.format('.')
        else:
            return fmt.format(', '.join(list_item))

    def visualize_terminal(self, training_samples, test_samples=None):
        out = []
        for i in xrange(self.map_rows):
            out_row = []
            for j in xrange(self.map_cols):
                out_row.append([])
            out.append(out_row)
        #create terminal matrix
        for sample in training_samples:
            print sample
            winner, diff = self.calc_similarity(sample.features)
            row, col = self.to_grid(winner)
            out[row][col].append(sample.name)
        if test_samples is not None:
            for sample in test_samples:
                print sample
                winner, diff = self.calc_similarity(sample.features)
                row, col = self.to_grid(winner)
                out[row][col].append(sample.name)
        #throw matrix to stdout
        for row_items in out:
            line = " ".join(map(lambda x: self.to_str(x), row_items))
            print line

    def visualize_plt(self,
                      training_samples,
                      class_name,
                      class_plt_style,
                      test_samples=None,
                      ):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #plot training samples
        for sample in training_samples:
            winner, diff = self.calc_similarity(sample.features)
            row, col = self.to_grid(winner)
            sample_class = sample.classes[class_name]
            if sample_class in class_plt_style:
                ax.plot(col+1, self.map_rows-row, class_plt_style[sample_class])
            else:
                ax.plot(col+1, self.map_rows-row, DEFAULT_TRAINING_CLASS_STYLE)
        #plot testing samples
        if test_samples is not None:
            for sample in test_samples:
                winner, diff = self.calc_similarity(sample.features)
                row, col = self.to_grid(winner)
                ax.plot(col+1, self.map_rows-row, DEFAULT_TEST_CLASS_STYLE)
        ax.set_ylim([0, self.map_rows+1])
        ax.set_xlim([0, self.map_cols+1])
        plt.show()
