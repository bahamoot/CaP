import os
import math
import cap.plugin.base
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
from cap.model.som import SOM2D
from cap.settings import DFLT_WEIGHT_STEP_SIZE
from cap.settings import DFLT_NBH_STEP_SIZE
from cap.settings import DFLT_MAX_NBH_SIZE
from cap.settings import DFLT_MAP_ROWS
from cap.settings import DFLT_MAP_COLS
from cap.settings import DFLT_SEED
from cap.settings import FIGS_TMP_OUT_DIR
from cap.settings import TERM_TMP_OUT_DIR
from cap.settings import TYPE_TRAINING_SAMPLE
from cap.settings import TYPE_TEST_SAMPLE


ROOT_DEMO_DATA = '/home/jessada/development/scilifelab/projects/CaP/cap/data/'
DEMO_TRAINING_FEATURES = os.path.join(ROOT_DEMO_DATA,
                                      'demo_training_features.txt')
DEMO_TRAINING_CLASSES = os.path.join(ROOT_DEMO_DATA,
                                     'demo_training_classes.txt')
DEMO_TEST_FEATURES = os.path.join(ROOT_DEMO_DATA,
                                  'demo_test_features.txt')
DEMO_TEST_CLASSES = os.path.join(ROOT_DEMO_DATA,
                                 'demo_test_classes.txt')
DEMO_OUT_DIR = '/home/jessada/development/scilifelab/projects/CaP/out/tmp/'
PARADIGM_WEIGHT_STEP_SIZE = 0.2
PARADIGM_NBH_STEP_SIZE = 8
PARADIGM_MAX_NBH_SIZE = 5
PARADIGM_MAP_ROWS = 10
PARADIGM_MAP_COLS = 10
PARADIGM_RANDOM_SEED = None

def get_time_stamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

#running demo version of SOM2D using Paradigm data
def demo_som2d_paradigm():
    visualize_params = []
    visualize_params.append({'type': 'terminal',
                             'txt_width': 15,
                             })
    visualize_params.append({'type': 'scatter',
                             'group_name': 'MSI_status',
                             'class_plt_style': {'MSI-H': 'r^',
                                                 'MSI-L': 'b*',
                                                 'MSS': 'mo',
                                                 },
                             })
    visualize_params.append({'type': 'scatter',
                             'group_name': 'methylation_subtype',
                             'class_plt_style': {'CIMP.H': 'r^',
                                                 'CIMP.L': 'b*',
                                                 'Cluster3': 'gv',
                                                 'Cluster4': 'mo',
                                                 },
                             })
    visualize_params.append({'type': 'scatter',
                             'group_name': 'tumor_stage',
                             'class_plt_style': {'Stage I': 'r^',
                                                 'Stage IIA': 'b*',
                                                 'Stage IIB': 'yD',
                                                 'Stage IIIA': 'mH',
                                                 'Stage IIIB': 'co',
                                                 'Stage IIIC': 'gv',
                                                 'Stage IV': 'mx',
                                                 },
                             })
    visualize_params.append({'type': 'scatter',
                             'group_name': 'anatomic_organ_subdivision',
                             'class_plt_style': {'Ascending Colon': 'r^',
                                                 'Cecum': 'b*',
                                                 'Descending Colon': 'yD',
                                                 'Hepatic Flexure': 'mH',
                                                 'Rectosigmoid Junction': 'co',
                                                 'Rectum': 'gv',
                                                 'Sigmoid Colon': 'mx',
                                                 'Transverse Colon': 'bp',
                                                 },
                             })
    visualize_params.append({'type': 'scatter',
                             'group_name': 'tumor_site',
                             'class_plt_style': {'1 - right colon': 'r^',
                                                 '2 - transverse colon': 'b*',
                                                 '3 - left colon': 'mo',
                                                 '4 - rectum': 'gv',
                                                 },
                             })
    out = som2d_paradigm(DEMO_TRAINING_FEATURES,
                         DEMO_TRAINING_CLASSES,
                         test_features_file=DEMO_TEST_FEATURES,
                         visualize_params=visualize_params,
                         )
    return out

#A wrapped layer to call som2d using Paradigm configuration
def som2d_paradigm(training_features_file,
                   training_classes_file,
                   test_features_file=None,
                   visualize_params=None,
                   ):
    current_time = get_time_stamp()
    out_folder = os.path.join(DEMO_OUT_DIR,
                              'Paradigm/'+current_time)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    training_samples = cap.plugin.base.load_samples(training_features_file,
                                                    training_classes_file)
    if test_features_file is not None:
        test_samples = cap.plugin.base.load_samples(test_features_file,
                                                    samples_type=TYPE_TEST_SAMPLE)
    else:
        test_samples = None
    #shorten training samples name
    for training_sample in training_samples:
        training_sample.name = training_sample.name.replace("TCGA-", "")
    return som2d(training_samples,
                 test_samples,
                 visualize_params=visualize_params,
                 out_folder=out_folder,
                 map_rows=PARADIGM_MAP_ROWS,
                 map_cols=PARADIGM_MAP_COLS,
                 weight_step_size=PARADIGM_WEIGHT_STEP_SIZE,
                 nbh_step_size=PARADIGM_NBH_STEP_SIZE,
                 max_nbh_size=PARADIGM_MAX_NBH_SIZE,
                 random_seed=PARADIGM_RANDOM_SEED,
                 )

#wrap all require stuffs to run SOM2D
def som2d(training_samples,
          test_samples=None,
          visualize_params=None,
          out_folder=None,
          map_rows=DFLT_MAP_ROWS,
          map_cols=DFLT_MAP_COLS,
          weight_step_size=DFLT_WEIGHT_STEP_SIZE,
          nbh_step_size=DFLT_NBH_STEP_SIZE,
          max_nbh_size=DFLT_MAX_NBH_SIZE,
          random_seed=DFLT_SEED,
          ):
    features_size = len(training_samples[0].features)
    model = SOM2D(features_size,
                  map_rows=map_rows,
                  map_cols=map_cols,
                  weight_step_size=weight_step_size,
                  nbh_step_size=nbh_step_size,
                  max_nbh_size=max_nbh_size,
                  random_seed=random_seed,
                  )
    #train and load sample for visualize
    model.train(training_samples)
    model.load_visualize_samples(training_samples, test_samples)

    #generate summary pdf report
    pdf_font = {'family' : 'monospace',
                'size'   : 3}
    matplotlib.rc('font', **pdf_font)
    fig_rows = 2
    fig_cols = 3
    legend_width = 1
    description_height = 1
    fig_width = 2
    fig_height = 2
    plt_rows = fig_rows*fig_height + description_height
    plt_cols = (fig_width+legend_width) * fig_cols
    fig = plt.figure()
    idx = 0
    #plot figures
    for params in visualize_params:
        fig_col = (idx%fig_cols) * (fig_width+legend_width)
        fig_row = ((idx//fig_cols)*fig_height) + description_height
        if params['type'] == 'terminal':
            ax = plt.subplot2grid((plt_rows, plt_cols), (fig_row, fig_col), colspan=fig_width, rowspan=fig_height)
            out_terminal = model.visualize_terminal(txt_width=params['txt_width'],
                                                    out_folder=out_folder,
                                                    )
            model.visualize_sample_name(ax)
        elif params['type'] == 'scatter':
            ax = plt.subplot2grid((plt_rows, plt_cols), (fig_row, fig_col), colspan=fig_width, rowspan=fig_height)
            out_plt = model.visualize_plt(ax,
                                          params['group_name'],
                                          params['class_plt_style'],
                                          )
        idx += 1
    #plot training attributes
    training_samples_size = len(training_samples)
    if test_samples is not None:
        test_samples_size = len(test_samples)
    else:
        test_samples_size = 0
    training_iterations = int(math.ceil(float(model.max_nbh_size)/model.nbh_step_size))
    samples_txt_fmt = "{caption:<28}:{value:>7}"
    samples_txt = []
    samples_txt.append(samples_txt_fmt.format(caption="number of training samples",
                                              value=training_samples_size))
    samples_txt.append(samples_txt_fmt.format(caption="number of test samples",
                                              value=test_samples_size))
    samples_txt.append(samples_txt_fmt.format(caption="features size",
                                              value=model.features_size))
    samples_txt.append(samples_txt_fmt.format(caption="training iterations",
                                              value=training_iterations))
    model_txt_fmt = "{caption:<24}:{value:>12}"
    model_txt = []
    model_txt.append(model_txt_fmt.format(caption="map rows",
                                          value=model.map_rows))
    model_txt.append(model_txt_fmt.format(caption="map cols",
                                          value=model.map_cols))
    model_txt.append(model_txt_fmt.format(caption="max neighborhod size",
                                          value=model.max_nbh_size))
    model_txt.append(model_txt_fmt.format(caption="neighborhood step size",
                                          value=model.nbh_step_size))
    model_txt.append(model_txt_fmt.format(caption="random seed",
                                          value=model.random_seed))
    ax = plt.subplot2grid((plt_rows, plt_cols), (0, 0), colspan=fig_cols*(fig_width+legend_width))
    out_plt = model.visualize_txt(ax, samples_txt, model_txt)
    plt.tight_layout()
    summary_pdf_file_name = os.path.join(out_folder, 'summary.pdf')
    fig.savefig(summary_pdf_file_name, bbox_inches='tight', pad_inches=0.1)
    return {"summary file": summary_pdf_file_name,
            "terminal file": out_terminal,
            }
