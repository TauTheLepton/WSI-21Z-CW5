# Data
red = 'winequality-red.csv'
white = 'winequality-white.csv'
use_file = red
# Options and settings
hidden_layer_size = 5 # TODO 2 dla normalizacji [0, 1] i sigmoidalnej w ostatniej warstwie, 16 dla [-1,1], bez sigmo i z inną funkcją straty
output_layer_size = 1
era = 1000  # TODO nie wiem jak w tym przypadku tłumaczy się epoka na ang, dlatego dałem era
learning_param = 0.7
bias = False # Use biases (True) or not (False)
A = 0 # Lower bound for input scaling
B = 1 # Upper bound for input scaling
# End of options and settings

