import numpy
from cap.template import CaPBase
from collections import namedtuple
from cap.settings import TYPE_TRAINING_SAMPLE


my_animals = ('antelop', 'ape', 'bat', 'bear', 'beetle', 'butterfly', 'camel', 'cat', 'crocodile', 'dog', 'dragonfly',
	   'duck', 'elephant', 'frog', 'giraffe', 'grasshopper', 'horse', 'housefly', 'hyena', 'kangaroo', 'lion',
	   'moskito', 'ostrich', 'pelican', 'penguin', 'pig', 'rabbit', 'rat', 'seaturtle', 'skunk', 'spider', 'walrus')

properties = ('antlered', 'articulations', 'barks', 'big', 'bigears', 'biting', 'black', 'blood', 'brown', 'climbing',
              'cns', 'coldblooded', 'curltail', 'digging', 'eatsanimals', 'eatsanything', 'eatscarrion', 'eatsfish',
              'eatsflies', 'eatsgarbage', 'eatsgrass', 'eggs', 'eightlegged', 'exoskeleton', 'extremelysmall', 'fatbody',
              'feathers', 'feelerless', 'feelers', 'flying', 'fourlegged', 'fourwinged', 'fur', 'gibbous', 'gnawteeth',
              'green', 'grey', 'hoofs', 'humanlike', 'jumping', 'landliving', 'lightbrown', 'lissom', 'livingoffspring',
              'longneck', 'longnosed', 'medium', 'moving', 'nervoussystem', 'nib', 'notflying', 'oddtoed',
              'oxygenconsuming', 'pairtoed', 'pink', 'pipetrakeas', 'plates', 'pouch', 'proboscis', 'robust',
              'ruminanting', 'running', 'shell', 'shortnosed', 'shorttail', 'sixlegged', 'small', 'smellsterrible',
              'spine', 'tail', 'thinbody', 'trakeas', 'tusked', 'twolegged', 'twowinged', 'warmblooded', 'waterliving',
              'verybig', 'verylongears', 'verysmall', 'white', 'wingless', 'wings', 'yellow')

propindex = {}
for i, p in enumerate(properties):
    propindex[p] = i

# Dictionary holding the property array for each animal
my_props = {}

def setProps(a, propList):
    indices = [propindex[p] for p in propList]
    v = numpy.zeros(len(properties))
    v[indices] = 1
    my_props[a] = v

setProps('bat',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'landliving', 'fourlegged',
          'flying', 'eatsflies', 'verysmall', 'grey', 'tail'))

setProps('rat',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'gnawteeth', 'tail',
          'landliving', 'fourlegged', 'eatsgarbage', 'small', 'brown'))

setProps('rabbit',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'landliving', 'fourlegged',
          'jumping', 'shorttail', 'eatsgrass', 'gnawteeth', 'verylongears',
          'small', 'white'))

setProps('elephant',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'fourlegged', 'landliving',
          'eatsgrass', 'robust', 'bigears', 'proboscis', 'tusked', 'tail', 'verybig',
          'grey'))

setProps('horse',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'fourlegged', 'hoofs',
          'eatsgrass', 'landliving', 'oddtoed', 'running', 'big', 'brown', 'tail'))

setProps('antelop',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'fourlegged', 'hoofs',
          'shorttail', 'eatsgrass', 'pairtoed', 'landliving', 'ruminanting',
          'lissom', 'antlered', 'running', 'medium', 'lightbrown'))

setProps('giraffe',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'fourlegged', 'hoofs',
          'eatsgrass', 'pairtoed', 'landliving', 'ruminanting', 'longneck',
          'tail', 'big', 'yellow'))

setProps('camel',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'warmblooded', 'fur', 'livingoffspring', 'fourlegged', 'hoofs',
          'eatsgrass', 'pairtoed', 'landliving', 'gibbous', 'tail', 'big', 'yellow'))

setProps('pig',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'fourlegged', 'hoofs',
          'eatsgrass', 'pairtoed', 'landliving', 'digging', 'tail', 'curltail', 'big', 'pink'))

setProps('walrus',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'warmblooded', 'fur', 'livingoffspring', 'fourlegged',
          'eatsanimals', 'waterliving', 'tusked', 'verybig', 'brown'))

setProps('skunk',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'warmblooded', 'fur', 'livingoffspring', 'fourlegged',
          'landliving', 'eatscarrion', 'tail', 'smellsterrible', 'medium', 'black'))

setProps('hyena',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'fourlegged', 'landliving',
          'longnosed', 'shorttail', 'eatscarrion', 'medium', 'brown'))

setProps('dog',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'fourlegged', 'eatsanimals',
          'landliving', 'longnosed', 'tail', 'lissom', 'medium', 'brown', 'barks'))

setProps('bear',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'fourlegged', 'shorttail',
          'eatsanimals', 'landliving', 'longnosed', 'robust', 'big', 'brown'))

setProps('lion',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'warmblooded', 'fur', 'livingoffspring', 'fourlegged',
          'eatsanimals', 'landliving', 'shortnosed', 'tail', 'lissom', 'climbing',
          'big', 'yellow'))

setProps('cat',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'warmblooded', 'fur', 'livingoffspring', 'fourlegged',
          'eatsanimals', 'landliving', 'shortnosed', 'tail', 'lissom', 'climbing',
          'small', 'black'))

setProps('ape',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'fur', 'livingoffspring', 'landliving', 'shorttail',
          'eatsanything', 'twolegged', 'shortnosed', 'humanlike', 'big', 'black'))

setProps('kangaroo',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'warmblooded', 'fur', 'landliving', 'fourlegged',
          'livingoffspring', 'pouch', 'jumping', 'eatsgrass', 'medium', 'tail',
          'lightbrown'))

setProps('duck',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'warmblooded', 'wings', 'nib', 'twolegged', 'feathers', 'eggs',
          'flying', 'eatsgrass', 'small', 'white'))

setProps('pelican',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'warmblooded', 'wings', 'nib', 'twolegged', 'feathers', 'eggs',
          'flying', 'eatsfish', 'medium', 'white'))

setProps('penguin',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'warmblooded', 'wings', 'nib', 'twolegged', 'feathers', 'eggs',
          'notflying', 'eatsfish', 'waterliving', 'small', 'black'))

setProps('ostrich',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns', 'blood',
          'warmblooded', 'wings', 'nib', 'twolegged', 'feathers', 'eggs',
          'notflying', 'eatsgrass', 'running', 'big', 'black'))

setProps('crocodile',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'coldblooded', 'eggs', 'tail', 'waterliving', 'fourlegged',
          'plates', 'eatsanimals', 'brown', 'big', 'tail'))

setProps('seaturtle',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine',
          'cns', 'blood', 'coldblooded', 'eggs', 'tail', 'shell',
          'fourlegged', 'eatsgrass', 'waterliving', 'big', 'brown'))

setProps('frog',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'spine', 'cns',
          'blood', 'coldblooded', 'waterliving', 'fourlegged', 'eggs',
          'verysmall', 'jumping', 'eatsflies', 'green'))

setProps('housefly',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'exoskeleton',
          'articulations', 'eggs', 'extremelysmall', 'trakeas', 'feelers',
          'sixlegged', 'wings', 'flying', 'proboscis', 'twowinged', 'fatbody',
          'black'))

setProps('moskito',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'exoskeleton',
          'articulations', 'eggs', 'extremelysmall', 'trakeas', 'feelers',
          'sixlegged', 'wings', 'flying', 'proboscis', 'twowinged', 'thinbody',
          'lightbrown'))

setProps('butterfly',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'exoskeleton',
          'articulations', 'eggs', 'extremelysmall', 'trakeas', 'feelers',
          'sixlegged', 'wings', 'flying', 'proboscis', 'fourwinged', 'fatbody',
          'yellow'))

setProps('beetle',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'exoskeleton',
          'articulations', 'eggs', 'extremelysmall', 'trakeas', 'feelers',
          'sixlegged', 'wings', 'flying', 'biting', 'fourwinged', 'fatbody',
          'black', 'shell'))

setProps('dragonfly',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'exoskeleton',
          'articulations', 'eggs', 'extremelysmall', 'trakeas', 'feelers',
          'sixlegged', 'wings', 'flying', 'biting', 'fourwinged', 'thinbody',
          'brown'))

setProps('grasshopper',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'exoskeleton',
          'articulations', 'eggs', 'extremelysmall', 'trakeas', 'feelers',
          'sixlegged', 'wings', 'fourwinged', 'flying', 'biting', 'jumping',
          'fatbody', 'green'))

setProps('spider',
         ('oxygenconsuming', 'moving', 'nervoussystem', 'exoskeleton',
          'articulations', 'eggs', 'extremelysmall', 'wingless', 'pipetrakeas',
          'feelerless', 'eightlegged', 'fatbody', 'black'))


Coord = namedtuple('Coord', ['x', 'y'])

class Animal(CaPBase):
    """ to keep and manipulate animal information """

    def __init__(self, name):
        CaPBase.__init__(self)
        self.__name = name
        self.__term_coord = None
        self.__plt_coord = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<' + self.__class__.__name__ + ' Object> ' + str(self.get_raw_repr())

    def get_raw_repr(self):
        return {'name': self.name,
        #        'features': self.features,
                'terminal coordinate': self.term_coord,
                'plot coordinate': self.plt_coord,
                }

    def set_term_coord(self, row, col):
        self.__term_coord = Coord(col, row)

    def set_plt_coord(self, row, col):
        self.__plt_coord = Coord(col, row)

    @property
    def name(self):
        return self.__name

    @property
    def features(self):
        return self.__features

    @features.setter
    def features(self, value):
        self.__features = value

    @property
    def term_coord(self):
        return self.__term_coord

    @property
    def plt_coord(self):
        return self.__plt_coord


def load_animals(samples_type=TYPE_TRAINING_SAMPLE):
    animals = []
    for animal_name in my_animals:
        animal = Animal(animal_name)
        animal.features = my_props[animal_name]
        animal.classes = my_props[animal_name]
        animal.sample_type = samples_type
        animals.append(animal)
    return animals


# Clean up things that should not be exported
del numpy, setProps, propindex, i, p
