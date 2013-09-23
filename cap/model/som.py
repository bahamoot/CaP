import numpy as np
import math
import sys
import os
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
from matplotlib.backends.backend_pdf import PdfPages


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
            self.__random_seed = randint(1, sys.maxint)
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
        #self.__pdf_pages = None

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

    def to_str(self, list_item, txt_width):
        fmt = '{:<' + str(txt_width) + '}'
        if len(list_item) == 0:
            return fmt.format('.')
        else:
            return fmt.format(', '.join(map(lambda x: x.name, list_item)))

    def __get_grid_coord(self, features):
        winner, diff = self.calc_similarity(features)
        return self.to_grid(winner)

    def __get_terminal_grid_coord(self, features):
        return self.__get_grid_coord(features)

    def __to_plt(self, row, col):
        plt_row = self.map_rows - row
        plt_col = col + 1
        return plt_row, plt_col

    def __get_plt_grid_coord(self, features):
        row, col = self.__get_grid_coord(features)
        return self.__to_plt(row, col)

    def __calc_samples_coord(self,
                             training_samples,
                             test_samples=None,
                             ):
        for sample in training_samples:
            if sample.term_coord is None:
                row, col = self.__get_terminal_grid_coord(sample.features)
                sample.set_term_coord(row=row, col=col)
                plt_row, plt_col = self.__to_plt(row, col)
                sample.set_plt_coord(row=plt_row, col=plt_col)
        for sample in test_samples:
            if sample.term_coord is None:
                row, col = self.__get_terminal_grid_coord(sample.features)
                sample.set_term_coord(row=row, col=col)
                plt_row, plt_col = self.__to_plt(row, col)
                sample.set_plt_coord(row=plt_row, col=plt_col)

    def __generate_samples_matrix(self,
                                  training_samples,
                                  test_samples=None,
                                  ):
        self.__sm = []
        for i in xrange(self.map_rows+1):
            samples_row = []
            for j in xrange(self.map_cols+1):
                samples_row.append([])
            self.__sm.append(samples_row)
        #add samples to matrix
        for sample in training_samples:
            x, y = sample.plt_coord
            self.__sm[y][x].append(sample)
        if test_samples is not None:
            for sample in test_samples:
                x, y = sample.plt_coord
                self.__sm[y][x].append(sample)


    def load_visualize_samples(self,
                               training_samples,
                               test_samples=None,
                               ):
        self.__calc_samples_coord(training_samples, test_samples)
        self.__generate_samples_matrix(training_samples, test_samples)

    def visualize_txt(self,
                      ax,
                      col1_txt_list,
                      col2_txt_list,
                      ):
        ax.axis('off')
        col1_rows = len(col1_txt_list)
        col2_rows = len(col2_txt_list)
        if col1_rows > col2_rows:
            txt_rows = col1_rows
        else:
            txt_rows = col2_rows
        plt_txt_fmt = "{col1_txt:<45}{col2_txt:>45}"
        plt_txt = []
        for i in xrange(txt_rows):
            if i >= col1_rows:
                col1_txt = ''
            else:
                col1_txt = col1_txt_list[i]
            if i >= col2_rows:
                col2_txt = ''
            else:
                col2_txt = col2_txt_list[i]
            plt_txt.append(plt_txt_fmt.format(col1_txt=col1_txt,
                                              col2_txt=col2_txt))
        txt_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.5,
                1,
                "\n".join(plt_txt),
                transform=ax.transAxes,
                family='monospace',
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='top',
                bbox=txt_props,
                )
        return ax

    def visualize_terminal(self,
                           txt_width=DFLT_TERMINAL_STR_WIDTH,
                           out_folder=None,
                           ):
        #redirect stdout if output folder is presented
        if out_folder is not None:
            terminal_out = os.path.join(out_folder,
                                        'terminal_out.txt')
            sys.stdout = open(terminal_out,
                              'w')
        else:
            terminal_out = None
        #throw matrix to stdout
        for row_items in self.__sm[1:len(self.__sm)]:
            line = " ".join(map(lambda x: self.to_str(x, txt_width),
                                row_items[1:len(row_items)]
                                ))
            print line
        #redirect stdout back to the normal one
        if out_folder is not None:
            sys.stdout.flush()
            sys.stdout = sys.__stdout__
        return terminal_out

    def visualize_sample_name(self,
                              ax,
                              out_file_name=None,
                              txt_size=6,
                              ):
        sm = self.__sm
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9, linewidth=0.1)
        for y in xrange(len(sm)):
            for x in xrange(len(sm[y])):
                if len(sm[y][x]) > 0:
                    ax.text(x,
                            y,
                            ", ".join(map(lambda x: x.name, sm[y][x])),
                            ha="center",
                            va="center",
                            size=2,
                            bbox=bbox_props)
        ax.set_xlim([0, self.map_cols+1])
        ax.set_ylim([0, self.map_rows+1])
        ax.set_title("samples name", fontsize=txt_size)
        return ax

    def visualize_new_plt(self,
                          ax,
                          training_samples,
                          group_name,
                          class_plt_style,
                          test_samples=None,
                          marker_size=3,
                          txt_size=6,
                          ):
        self.__calc_samples_coord(training_samples, test_samples)
        samples_count = []
        for i in xrange(self.map_rows+1):
            samples_row = []
            for j in xrange(self.map_cols+1):
                samples_row.append([])
            samples_count.append(samples_row)
        #create frequency matrix
        for sample in training_samples:
            x, y = sample.plt_coord
            samples_count[y][x].append(sample)
        if test_samples is not None:
            for sample in test_samples:
                x, y = sample.plt_coord
                samples_count[y][x]['test data'] += 1

        for y in len(xrange(samples_count)):
            for x in len(xrange(samples_count[y])):
                for sample_group in samples_count[y][x]:
                    pass


        #record training samples
        x_coords = defaultdict(list)
        y_coords = defaultdict(list)
        for sample in training_samples:
            x, y = sample.plt_coord
            sample_class = sample.classes[group_name]
            if sample_class in class_plt_style:
                x_coords[sample_class].append(x)
                y_coords[sample_class].append(y)
            else:
                x_coords['unknown'].append(x)
                y_coords['unknown'].append(y)
        #record test samples
        if test_samples is not None:
            for sample in test_samples:
                x, y = sample.plt_coord
                x_coords['test data'].append(x)
                y_coords['test data'].append(y)
        #plot samples
        class_plt_style['unknown'] = DFLT_TRAINING_CLASS_STYLE
        class_plt_style['test data'] = DFLT_TEST_CLASS_STYLE
        plots = OrderedDict()
        for sample_class in sorted(x_coords.keys()):
            p = ax.plot(x_coords[sample_class],
                        y_coords[sample_class],
                        class_plt_style[sample_class],
                        label=sample_class,
                        markersize=marker_size,
                        )
            plots[sample_class] = p
        ax.set_ylim([0, self.map_rows+1])
        ax.set_xlim([0, self.map_cols+1])
        ax.set_title(group_name, fontsize=txt_size)
        txt_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.legend(map(lambda x: plots[x][0], plots),
                  plots,
                  bbox_to_anchor=(1., 1.02),
                  loc=2,
                  ncol=1,
                  prop={'size':txt_size},
                  )
        return ax

    def visualize_plt(self,
                      ax,
                      training_samples,
                      group_name,
                      class_plt_style,
                      test_samples=None,
                      marker_size=3,
                      txt_size=6,
                      ):
        self.__calc_samples_coord(training_samples, test_samples)
        #record training samples
        x_coords = defaultdict(list)
        y_coords = defaultdict(list)
        for sample in training_samples:
            x, y = sample.plt_coord
            sample_class = sample.classes[group_name]
            if sample_class in class_plt_style:
                x_coords[sample_class].append(x)
                y_coords[sample_class].append(y)
            else:
                x_coords['unknown'].append(x)
                y_coords['unknown'].append(y)
        #record test samples
        if test_samples is not None:
            for sample in test_samples:
                x, y = sample.plt_coord
                x_coords['test data'].append(x)
                y_coords['test data'].append(y)
        #plot samples
        class_plt_style['unknown'] = DFLT_TRAINING_CLASS_STYLE
        class_plt_style['test data'] = DFLT_TEST_CLASS_STYLE
        plots = OrderedDict()
        for sample_class in sorted(x_coords.keys()):
            p = ax.plot(x_coords[sample_class],
                        y_coords[sample_class],
                        class_plt_style[sample_class],
                        label=sample_class,
                        markersize=marker_size,
                        )
            plots[sample_class] = p
        ax.set_ylim([0, self.map_rows+1])
        ax.set_xlim([0, self.map_cols+1])
        ax.set_title(group_name, fontsize=txt_size)
        txt_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.legend(map(lambda x: plots[x][0], plots),
                  plots,
                  bbox_to_anchor=(1., 1.02),
                  loc=2,
                  ncol=1,
                  prop={'size':txt_size},
                  )
        return ax
