import numpy as np
from cap.template import CaPBase


class Sample(CaPBase):
    """ to keep and manipulate sample information """

    def __init__(self, name):
        CaPBase.__init__(self)
        self.__name = name
        self.features = None
        self.classes = None
        self.content = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<' + self.__class__.__name__ + ' Object> ' + str(self.get_raw_repr())

    def get_raw_repr(self):
        return {'name': self.name,
                'content': self.content,
                'features': self.features,
                'classes': self.classes,
                }

    @property
    def name(self):
        return self.__name

    @property
    def content(self):
        return self.__content

    @content.setter
    def content(self, value):
        self.__content = value

    @property
    def features(self):
        return self.__features

    @features.setter
    def features(self, value):
        self.__features = value
        self.__content = None

    @property
    def classes(self):
        return self.__classes

    @classes.setter
    def classes(self, value):
        self.__classes = value


class SamplesLoader(CaPBase):
    """ to load samples and convert them into the format ready for training """

    def __init__(self,
                 features_file=None,
                 classes_file=None,
                 ):
        CaPBase.__init__(self)
        self.__features_file = features_file
        self.__classes_file = classes_file
        self.__samples = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<' + self.__class__.__name__ + ' Object> ' + str(self.get_raw_repr())

    def get_raw_repr(self):
        return {'features_file': self.__features_file,
                'classes_file': self.__classes_file,
                }

    def __process_samples_file(self, samples_file):
        samples_matrix = np.genfromtxt(samples_file, dtype='str', delimiter="\t")
        header = samples_matrix[0][1:len(samples_matrix[0])]
        records = samples_matrix[1:len(samples_matrix)]
        samples = {}
        for record in records:
            sample = Sample(record[0])
            sample.content = record[1:len(record)]
            samples[sample.name] = sample
        return header, samples

    def __process_features_file(self):
        header, samples = self.__process_samples_file(self.__features_file)
        #convert sample content into features
        for sample_name in samples:
            sample = samples[sample_name]
            sample.features = sample.content.astype(np.float)
        return samples

    def __process_classes_file(self):
        header, samples = self.__process_samples_file(self.__classes_file)
        #convert sample content into classes
        for sample_name in samples:
            sample = samples[sample_name]
            content = sample.content
            classes = {}
            for i in xrange(len(content)):
                classes[header[i]] = content[i]
            sample.classes = classes
        return samples

    def __generate_samples(self):
        out_samples = self.__process_features_file()
        if self.__classes_file is not None:
            classes_samples = self.__process_classes_file()
            for sample_name in out_samples:
                out_sample = out_samples[sample_name]
                class_sample = classes_samples[sample_name]
                out_sample.classes = class_sample.classes
        self.__samples = out_samples

    def get_samples_hash(self):
        if self.__samples is None:
            self.__generate_samples()
        return self.__samples

    def get_samples_list(self):
        if self.__samples is None:
            self.__generate_samples()
        return map(lambda x: self.__samples[x], self.__samples)


def load_samples(features_file=None,
                 classes_file=None,
                 ):
    """

    to load datasets under a very simple format,
        - The first row must be the properties/classes name
        - The first column must be the samples name
        - The rest are samples properties/classes

    """

    y = np.loadtxt(join_paradigm_result, dtype='str')
    print y
    return
