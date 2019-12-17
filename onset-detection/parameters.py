# DATA #####
# directory of the audio file
audio_file_dir = '/Users/gustavocidornelas/Desktop/sound-source/Male_Az_90.csv'

# GENERAL PARAMETERS #####
# number of states for one frequency (n = 2 for decaying_sinusoid and n = 8 for gammatone)
n = 2
# number of frequencies contained in the frequency bank
n_freq = 1
# total number of states
n_states = n * n_freq
# sampling frequency
Fs = 44100.0

# ONSET MODEL #####
# selection of the onset model (either 'decaying_sinusoid' or 'gammatone')
onset_model = 'decaying_sinusoid'

# WINDOW MODEL #####
# selection of the window model (either 'gamma' or 'exponential')
window_model = 'gamma'