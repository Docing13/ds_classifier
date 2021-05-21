import os

# data
NUMERIC_FEATURES = [
    'Elevation',
    'Aspect',
    'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

DROP_FEATURES = [
    'Id',
    'Soil_Type7',
    'Soil_Type15'
]

LAST_FEATURE = 'Soil_Type40'
LABELS = "Cover_Type"
# folders
DATA_FOLDER = 'data'
MODELS_FOLDER = 'models'
TRAIN_CSV = os.path.join(DATA_FOLDER, 'train.csv')
VAL_CSV = os.path.join(DATA_FOLDER, 'val.csv')
MODEL = os.path.join(MODELS_FOLDER, 'model.pickle')
# model selector log columns
ACCURACY = 'Accuracy'
CLASSIFIER = 'Classifier'
NAME = 'Name'
LOG_COLS = [NAME,
            ACCURACY,
            CLASSIFIER]
# model selector plot params
COLOR = 'b'
PLOT_TITLE = 'Classifier Accuracy'
