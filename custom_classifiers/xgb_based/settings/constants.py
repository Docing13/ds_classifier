import numpy as np

N_SPLIT = 8
NUM_CLASS = 7
OBJECTIVE = 'multi:softmax'
SCORING = 'f1_macro'
GRID_PARAMS = {
            'max_depth': [i for i in range(1, 5, 1)],
            'n_estimators': [i for i in range(1, 20, 5)],
            'learning_rate': np.linspace(1e-8, 1, 5)
        }
