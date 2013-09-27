import numpy as np
import math
import sys
import os
import matplotlib as mpl
from cap.template import CaPBase
from cap.settings import DFLT_SEED
from cap.settings import DFLT_MAP_SIZE
from cap.settings import DFLT_WEIGHT_STEP_SIZE
from cap.settings import DFLT_NBH_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE
from cap.settings import DFLT_MAP_ROWS
from cap.settings import DFLT_MAP_COLS
from cap.settings import TYPE_TRAINING_SAMPLE
from cap.settings import TYPE_TEST_SAMPLE
from collections import defaultdict
from collections import OrderedDict
from random import randint


DFLT_TRAINING_CLASS_STYLE = 'kp'
DFLT_TEST_CLASS_STYLE = 'k+'
DFLT_TERMINAL_STR_WIDTH = 11
DFLT_TXT_SIZE = 6

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


class VisualizeLegend(CaPBase):
    """ to store legend information """

    def __init__(self,
                 hide=False,
                 random_seed=DFLT_SEED,
                 ):
        CaPBase.__init__(self)
        self.lg_txt = None
        self.mk_size = None
        self.style = None
        self.__hide = hide
        self.x_coords = []
        self.y_coords = []

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<' + self.__class__.__name__ + ' Object> ' + str(self.get_raw_repr())

    def get_raw_repr(self):
        return {"legend text": self.lg_txt,
                "marker size": self.mk_size,
                "style": self.style,
                "hide": self.hide,
                "x coordinates": self.x_coords,
                "y coordinates": self.y_coords,
                }

    @property
    def hide(self):
        return self.__hide


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

    def __get_term_grid_coord(self, features):
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
                row, col = self.__get_term_grid_coord(sample.features)
                sample.set_term_coord(row=row, col=col)
                plt_row, plt_col = self.__to_plt(row, col)
                sample.set_plt_coord(row=plt_row, col=plt_col)
        for sample in test_samples:
            if sample.term_coord is None:
                row, col = self.__get_term_grid_coord(sample.features)
                sample.set_term_coord(row=row, col=col)
                plt_row, plt_col = self.__to_plt(row, col)
                sample.set_plt_coord(row=plt_row, col=plt_col)

    def __generate_samples_matrix(self,
                                  training_samples,
                                  test_samples=[],
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
        for sample in test_samples:
            x, y = sample.plt_coord
            self.__sm[y][x].append(sample)


    def load_visualize_samples(self,
                               training_samples,
                               test_samples=[],
                               ):
        self.__calc_samples_coord(training_samples, test_samples)
        self.__generate_samples_matrix(training_samples, test_samples)

    def visualize_txt(self,
                      ax,
                      col1_txt_list,
                      col2_txt_list,
                      txt_size=DFLT_TXT_SIZE,
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
                size=txt_size,
                bbox=txt_props,
                )
        return ax

    def visualize_term(self,
                       txt_width=DFLT_TERMINAL_STR_WIDTH,
                       out_file=None,
                       ):
        #redirect stdout if output folder is presented
        if out_file is not None:
            sys.stdout = open(out_file,
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
        if out_file is not None:
            sys.stdout.flush()
            sys.stdout = sys.__stdout__
        return out_file

    def visualize_sample_name(self,
                              ax,
                              txt_size=DFLT_TXT_SIZE,
                              ):
        sm = self.__sm
        bbox_props = dict(boxstyle="round",
                          fc="w",
                          ec="0.5",
                          alpha=0.6,
                          linewidth=0.1)
        for y in xrange(len(sm)):
            for x in xrange(len(sm[y])):
                if len(sm[y][x]) > 0:
                    ax.text(x,
                            y,
                            "\n".join(map(lambda x: x.name, sm[y][x])),
                            ha="center",
                            va="center",
                            size=2,
                            bbox=bbox_props)
        ax.set_xlim([0, self.map_cols+1])
        ax.set_ylim([0, self.map_rows+1])
        ax.set_title("samples name", fontsize=txt_size)
        return ax

    def debugging_contour_txt(self,
                              ax,
                              prop_name,
                              txt_size=DFLT_TXT_SIZE,
                              ):
        sm = self.__sm
        bbox_props = dict(boxstyle="round",
                          fc="w",
                          ec="0.5",
                          alpha=0.6,
                          linewidth=0.1)
        for y in xrange(len(sm)):
            for x in xrange(len(sm[y])):
                items = filter(lambda x: x.classes is not None, sm[y][x])
                if len(items) > 0:
                    ax.text(x,
                            y,
                            "\n".join(map(lambda x: x.classes[prop_name],
                                          items)),
                            ha="center",
                            va="center",
                            size=2,
                            bbox=bbox_props)
        ax.set_xlim([0, self.map_cols+1])
        ax.set_ylim([0, self.map_rows+1])
        title_txt = []
        title_txt.append("debugging content of")
        title_txt.append("'"+prop_name+"'")
        ax.set_title("\n".join(title_txt), fontsize=txt_size)
        return ax

    def debugging_contour_filter(self,
                                 ax,
                                 prop_name,
                                 min_cutoff=300,
                                 max_cutoff=720,
                                 txt_size=DFLT_TXT_SIZE,
                                 ):
        x_range = np.arange(0, self.map_cols+2, 1)
        y_range = np.arange(0, self.map_rows+2, 1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros((y_range.shape[0], x_range.shape[0]))
        sm = self.__sm
        bbox_props = dict(boxstyle="round",
                          fc="w",
                          ec="0.5",
                          alpha=0.6,
                          linewidth=0.1)
        for y in xrange(len(sm)):
            for x in xrange(len(sm[y])):
                items = filter(lambda x: x.classes is not None,
                                   sm[y][x])
                items = filter(lambda x: int(x.classes[prop_name])>min_cutoff,
                                   items)
                items = map(lambda x: int(x.classes[prop_name]),
                                items)
                if len(items) > 0:
                    ax.text(x,
                            y,
                            reduce(lambda a, b: a+b,
                                   items)/len(items),
                            ha="center",
                            va="center",
                            size=2,
                            bbox=bbox_props)
        ax.set_xlim([0, self.map_cols+1])
        ax.set_ylim([0, self.map_rows+1])
        title_txt = []
        title_txt.append("debugging filter+average")
        title_txt.append("'"+prop_name+"'>"+str(min_cutoff))
        ax.set_title("\n".join(title_txt), fontsize=txt_size)
        return ax

    def visualize_contour(self,
                          ax,
                          prop_name,
                          min_cutoff=300,
                          max_cutoff=720,
                          color_level_step_size=5,
                          txt_size=DFLT_TXT_SIZE,
                          ):
        #initailize grid
        x_range = np.arange(0, self.map_cols+2, 1)
        y_range = np.arange(0, self.map_rows+2, 1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros((y_range.shape[0], x_range.shape[0]))
        sm = self.__sm
        #map informative values to the grid
        for y in xrange(len(sm)):
            for x in xrange(len(sm[y])):
                #select only items that can be categorized
                items = filter(lambda x: x.classes is not None,
                                   sm[y][x])
                #remove items that are non-informative
                items = filter(lambda x: int(x.classes[prop_name])>min_cutoff,
                                   items)
                #cast items as integer
                items = map(lambda x: int(x.classes[prop_name]),
                                items)
                if len(items) > 0:
                    Z[y][x] = reduce(lambda a, b: a+b,
                                     items)/len(items)
        min_Z = abs(Z).min()
        max_Z = abs(Z).max()
        levels = np.append(np.array([0]),
                           np.arange(min_cutoff,
                                     max_Z+(max_Z-min_cutoff)*0.15,
                                     color_level_step_size,
                                     ),
                           )
        norm = mpl.colors.Normalize(vmax=max_Z, vmin=min_Z)
        cmap = mpl.cm.PRGn
        out_plt = ax.contourf(X, Y, Z,
                              levels,
                              cmap=mpl.cm.get_cmap(cmap, len(levels)-1),
                              norm=norm,
                              )
        ax.set_xlim([0, self.map_cols+1])
        ax.set_ylim([0, self.map_rows+1])
        ax.set_title(prop_name, fontsize=txt_size)
        return out_plt

    def visualize_plt(self,
                      ax,
                      prop_name,
                      plt_style,
                      mk_size=3,
                      txt_size=DFLT_TXT_SIZE,
                      ):
        sm = self.__sm
        #generate default legend
        lg_list = defaultdict(VisualizeLegend)
        plt_style['unknown'] = DFLT_TRAINING_CLASS_STYLE
        plt_style['test data'] = DFLT_TEST_CLASS_STYLE
        #count class frequency for each xy and generate new legend if any
        for y in xrange(len(sm)):
            for x in xrange(len(sm[y])):
                #count class frequency
                sample_class_count = defaultdict(lambda: 0)
                for sample in sm[y][x]:
                    if sample.sample_type == TYPE_TRAINING_SAMPLE:
                        sample_class = sample.classes[prop_name]
                        if sample_class in plt_style:
                            sample_class_count[sample_class] += 1
                        else:
                            sample_class_count['unknown'] += 1
                    elif sample.sample_type == TYPE_TEST_SAMPLE:
                        sample_class_count['test data'] += 1
                #generate new legend if any
                for sample_class in sample_class_count:
                    #generate legend key
                    sample_count = sample_class_count[sample_class]
                    if sample_count == 1:
                        lg_key = sample_class
                    else:
                        lg_key = sample_class + ' (' + str(sample_count) + ')'
                    #look up, add any, and add coordinate to the key
                    if lg_key not in lg_list:
                        lg_list[lg_key].lg_txt = sample_class
                        lg_list[lg_key].mk_size = mk_size + (sample_count-1)*2
                        lg_list[lg_key].style = plt_style[sample_class]
                    lg_list[lg_key].x_coords.append(x)
                    lg_list[lg_key].y_coords.append(y)
        #plot samples from each class
        for lg_key in lg_list:
            p = ax.plot(lg_list[lg_key].x_coords,
                        lg_list[lg_key].y_coords,
                        lg_list[lg_key].style,
                        label=lg_key,
                        markersize=lg_list[lg_key].mk_size,
                        )
            lg_list[lg_key].plot = p
        ax.set_ylim([0, self.map_rows+1])
        ax.set_xlim([0, self.map_cols+1])
        ax.set_title(prop_name, fontsize=txt_size)
        txt_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        lg_list=OrderedDict(sorted(lg_list.items(), key=lambda x: x[0]))
        ax.legend(map(lambda x: lg_list[x].plot[0], lg_list),
                  lg_list,
                  bbox_to_anchor=(1., 1.02),
                  loc=2,
                  ncol=1,
                  prop={'size':txt_size},
                  )
        return ax
