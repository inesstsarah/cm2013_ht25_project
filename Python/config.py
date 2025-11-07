# -- Project Configuration --

# Set the current iteration of the project (1-4). 
# This controls which parts of the pipeline are active.
CURRENT_ITERATION = 2

# Set to True to use cached data for preprocessing and feature extraction.
USE_CACHE = False
USE_PARALLEL = True # Use parallel processing where applicable
PARALLEL_N_JOBS = -1  # Number of parallel jobs (-1 uses all available cores)

# -- File Paths --
import os
DATA_DIR = '../data/'
TRAINING_DIR = f'{DATA_DIR}training/'
HOLDOUT_DIR = f'{DATA_DIR}holdout/'
SAMPLE_DIR = f'{DATA_DIR}sample/'
CACHE_DIR = 'cache/'
FIGURE_DATA_DIR = 'figure/Data/'
FIGURES_PREPROCESSING_DIR = 'figure/Preprocessing/'
FIGURES_FEATURE_EXTRACTION_DIR = 'figure/Feature Extraction/'
FIGURES_CLASSIFICATION_DIR = 'figure/Classification/'

# Validate and create directories if needed
def _create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}\nPlease ensure you are running from the correct directory.")
if not os.path.exists(CACHE_DIR):
    print(f"Creating cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

_create_dir_if_not_exists(FIGURE_DATA_DIR)
_create_dir_if_not_exists(FIGURES_PREPROCESSING_DIR)
_create_dir_if_not_exists(FIGURES_FEATURE_EXTRACTION_DIR)
_create_dir_if_not_exists(FIGURES_CLASSIFICATION_DIR)

# -- Preprocessing --
LOW_PASS_FILTER_FREQ = 40  # Hz
NOTCH_FILTER_FREQ = 60 # Hz
NOTCH_FILTER_Q = 30
BANDPASS_FILTER_LOWER_FREQ = 0.5 # Hz
BANDPASS_FILTER_HIGHER_FREQ = 33 # Hz
BANDPASS_FILTER_ORDER = 5
HIGHPASS_FILTER_FREQ = 0.5 # Hz

# -- Feature Extraction --
# (Add feature-specific parameters here)
AR_ORDER = 15
EEG_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    # 'sigma': (12.0, 15.0),
    'beta': (13.0, 30.0),
}
EEG_SE_PERCENTILE = 0.9
EEG_FS = 125
EEG_LOWER = 0.5
EEG_UPPER = 30

WELCH_PARAMETERS = {
    'window' : 'hann',
    'nperseg' : 4*EEG_FS,
    'noverlap' : int(0.5*4*EEG_FS),
    'nfft' : 4*EEG_FS
}
WAVELET_NAME = 'db4'

# -- Classification --
USE_HYPERPARAM_OPTIMAZATION = False
# Iteration-specific parameters - students should modify these based on current iteration
if CURRENT_ITERATION == 1: # TODO: add more hyperparameters for the hyperparameter optimization
    # Iteration 1: Basic pipeline with k-NN
    CLASSIFIER_TYPE = 'knn'
    GRID_PARAMS = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
    BEST_PARAMS = {
        'n_neighbors': 5,
        'weights': 'distance',
        'metric': 'manhattan'
    }

elif CURRENT_ITERATION == 2:
    # Iteration 2: Enhanced EEG processing with SVM
    CLASSIFIER_TYPE = 'svm'
    # SVM Grid Search Parameters
    GRID_PARAMS = {
        'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0], 
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }
    BEST_PARAMS = {
        'C': 5.0,
        'gamma': 'auto',
        'kernel': 'rbf',
        'class_weight': 'balanced'
    }
    
elif CURRENT_ITERATION == 3:
    # Iteration 3: Multi-signal processing with Random Forest
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
elif CURRENT_ITERATION == 4:
    # Iteration 4: Full system optimization
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 5
else:
    raise ValueError(f"Invalid CURRENT_ITERATION: {CURRENT_ITERATION}. Must be 1-4.")


# -- Visualization --
import matplotlib.colors as mcolors
import numpy as np

def _lighten(color, amount=0.5):
    c = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(np.clip(c + (1 - c) * amount, 0, 1))

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = ['red', 'orange', 'green', 'blue', 'purple'] # truth colors
LIGHT_STAGE_COLORS = [_lighten(c, 0.6) for c in STAGE_COLORS] # prediction colors

# -- Submission --
SUBMISSION_FILE = 'submission.csv'