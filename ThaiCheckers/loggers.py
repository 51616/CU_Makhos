
from utils import setup_logger
from settings import run_folder

# SET all LOGGER_DISABLED to True to disable logging
# WARNING: the mcts log file gets big quite quickly

LOGGER_DISABLED = {
    'main': False
}

logger_main = setup_logger('logger_main', run_folder + 'logs/logger_main.log')
logger_main.disabled = LOGGER_DISABLED['main']
