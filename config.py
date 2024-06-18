# URL of the zip file
DATA_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

FIG_PATH = "./figures"
DATA_PATH = "./data"
MODELS_PATH = "./models"
RESULTS_PATH = "./results"
MAJOR_CATEGORIES = f'{DATA_PATH}/meta/major_cat.csv'
META_DATA = f'{DATA_PATH}/meta/esc50.csv'
FINAL_META_DATA = f'{DATA_PATH}/meta/esc50_major_cat.csv'
AUDIO_PATH = f'{DATA_PATH}/audio'
SEMANTIC_LABELS = ["Animals", "Natural soundscapes & water sounds", "Human non-speech sounds", "Interior/domestic sounds", "Exterior/urban noises"]


#hyperparameters
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 20
