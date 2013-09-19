import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cap.template import CaPBase
from cap.settings import DFLT_SEED
from cap.settings import DFLT_MAP_SIZE
from cap.settings import DFLT_WEIGHT_STEP_SIZE
from cap.settings import DFLT_NBH_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE
from cap.settings import DFLT_MAP_ROWS
from cap.settings import DFLT_MAP_COLS
from collections import defaultdict
from collections import OrderedDict
from random import randint


DFLT_TRAINING_CLASS_STYLE = 'kp'
DFLT_TEST_CLASS_STYLE = 'k+'
DFLT_TERMINAL_STR_WIDTH = 11

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
        if self.random_seed is None:
            self.__random_seed = randint(-sys.maxint-1, sys.maxint)
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
                "weight step size": self.weight_step_size,
                "neighborhood step size": self.nbh_step_size,
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

    def to_str(self, list_item, terminal_str_width):
        fmt = '{:<' + str(terminal_str_width) + '}'
        if len(list_item) == 0:
            return fmt.format('.')
        else:
            return fmt.format(', '.join(list_item))

    def visualize_terminal(self,
                           training_samples,
                           terminal_str_width=DFLT_TERMINAL_STR_WIDTH,
                           test_samples=None,
                           output_file=None,
                           ):
        out = []
        for i in xrange(self.map_rows):
            out_row = []
            for j in xrange(self.map_cols):
                out_row.append([])
            out.append(out_row)
        #create terminal matrix
        for sample in training_samples:
            winner, diff = self.calc_similarity(sample.features)
            row, col = self.to_grid(winner)
            out[row][col].append(sample.name)
        if test_samples is not None:
            for sample in test_samples:
                winner, diff = self.calc_similarity(sample.features)
                row, col = self.to_grid(winner)
                out[row][col].append(sample.name)
        #redirect stdout if output_file is presented
        if output_file is not None:
            sys.stdout = open(output_file, 'w')
        #throw matrix to stdout
        for row_items in out:
            line = " ".join(map(lambda x: self.to_str(x, terminal_str_width),
                                row_items
                                ))
            print line
        #redirect stdout back to the normal one
        if output_file is not None:
            sys.stdout.flush()
            sys.stdout = sys.__stdout__
        return output_file

    def visualize_plt(self,
                      training_samples,
                      group_criteria,
                      class_plt_style,
                      test_samples=None,
                      figure_name=None,
                      ):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])
        #record training samples
        x_coods = defaultdict(list)
        y_coods = defaultdict(list)
        for sample in training_samples:
            winner, diff = self.calc_similarity(sample.features)
            row, col = self.to_grid(winner)
            plt_col = col + 1
            plt_row = self.map_rows - row
            sample_class = sample.classes[group_criteria]
            if sample_class in class_plt_style:
                x_coods[sample_class].append(plt_col)
                y_coods[sample_class].append(plt_row)
            else:
                x_coods['unknown'].append(plt_col)
                y_coods['unknown'].append(plt_row)
        #record test samples
        if test_samples is not None:
            for sample in test_samples:
                winner, diff = self.calc_similarity(sample.features)
                row, col = self.to_grid(winner)
                plt_col = col + 1
                plt_row = self.map_rows - row
                x_coods['test data'].append(plt_col)
                y_coods['test data'].append(plt_row)
        #plot samples
        class_plt_style['unknown'] = DFLT_TRAINING_CLASS_STYLE
        class_plt_style['test data'] = DFLT_TEST_CLASS_STYLE
        plots = OrderedDict()
        for sample_class in sorted(x_coods.keys()):
            p = ax.plot(x_coods[sample_class],
                        y_coods[sample_class],
                        class_plt_style[sample_class],
                        label = sample_class,
                        )
            plots[sample_class] = p
        ax.set_ylim([0, self.map_rows+1])
        ax.set_xlim([0, self.map_cols+1])
        ax.legend(map(lambda x: plots[x][0], plots),
                  plots,
                  bbox_to_anchor=(0., 1.02, 1., .102),
                  loc=3,
                  mode="expand",
                  ncol=2,
                  borderaxespad=0.,
                  )
        #add text to descript training attributes
        training_samples_size = len(training_samples)
        if test_samples is not None:
            test_samples_size = len(test_samples)
        else:
            test_samples_size = 0
        training_iterations = int(math.ceil(float(self.max_nbh_size)/self.nbh_step_size))
        plt_txt_fmt = "{caption:<28}:{value:>15}"
        plt_txt = []
        plt_txt.append(plt_txt_fmt.format(caption="number of training samples",
                                          value=training_samples_size))
        plt_txt.append(plt_txt_fmt.format(caption="number of test samples",
                                          value=test_samples_size))
        plt_txt.append(plt_txt_fmt.format(caption="features size",
                                          value=self.features_size))
        plt_txt.append(plt_txt_fmt.format(caption="classification property",
                                          value=group_criteria))
        plt_txt.append(plt_txt_fmt.format(caption="training iterations",
                                          value=training_iterations))
        plt_txt.append('')
        plt_txt.append('')
        plt_txt.append(plt_txt_fmt.format(caption="map rows",
                                          value=self.map_rows))
        plt_txt.append(plt_txt_fmt.format(caption="map cols",
                                          value=self.map_cols))
        plt_txt.append(plt_txt_fmt.format(caption="max neighborhod size",
                                          value=self.max_nbh_size))
        plt_txt.append(plt_txt_fmt.format(caption="neighborhood step size",
                                          value=self.nbh_step_size))
        plt_txt.append(plt_txt_fmt.format(caption="random seed",
                                          value=self.random_seed))
        txt_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.02,
                0.99,
                "\n".join(plt_txt),
                transform=ax.transAxes,
                family='monospace',
                fontsize=10,
                verticalalignment='top',
                bbox=txt_props,
                )
        if figure_name is not None:
            fig.savefig(figure_name, bbox_inches='tight', pad_inches=0.1)
        else:
            plt.show()
        return figure_name
