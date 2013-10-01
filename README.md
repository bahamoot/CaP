CaP
===

A model to predict cancer


How to prepare data?
--------------------

This model can only understand the matrix format that has the first row as
feature names and the first column as sample names. For example, if you
have 100 samples with 200 features each, in the file, there will be 201
columns and 101 rows.

If you need only to train the model, you will only need one file with
features for each sample. But if you need to virtualize model
performance by showing how the samples are clustered, you need to tell
the model which properties the model should used to visualize. To do
that you need to have a separate file with same matrix format as above
as an 'classes' input. 

And if you need to see the performance by using test data, of course,
you need another features file with features for each test sample.

Summary, to do everything, you may need three files with, training
features, training classes, and test features, which have the same
matrix format as above to feed the model.

The advantage of this format is that it's not much difficult to add more
features to existing samples, basically using 'join'. The drawback is
that it's a little difficult to add more samples.


How to run the model? and what to be awared?
--------------------------------------------

This model has a demo vesion to show how the training works. All you
need to do is to run 'sudo python dev_setup.py install', then run
'CaP_demo_SOM2D_Paradigm' to see the result. The only problem is that
you might not have any data since this model is hardcode data location.


Model architecture?
-------------------

This model consists of three major components, trainer, visualizer, and
data loader. But, somehow, visualizer and trainer are bundled together
because they depend on each other.

The trainer part is in model/som.py, which can be splitted into two layers,
SOMBase and SOM2D. SOMBase is the core of the model. Most model data are
stored at this layer. The training is also done at this layer. What is
lack at this layer is that it's output into one dimensional matrix. So
come SOM2D. Its purpose is to output 2D result. SOM2D modifies
neighborhood function (nbhs) so that it'll search for neighborhoods in
2D direction.

The data loader part is in plugin/base.py. Given the correct input
format, it'll produce the data that can be directly feed to the model.
