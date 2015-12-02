import logging

LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }


def make_logger(cysparse_config):
    # create logger
    logger_name = cysparse_config.get('CODE_GENERATION', 'log_name')
    if logger_name == '':
        logger_name = 'cysparse_generate_code'

    logger = logging.getLogger(logger_name)

    # levels
    log_level = LOG_LEVELS[cysparse_config.get('CODE_GENERATION', 'log_level')]
    console_log_level = LOG_LEVELS[cysparse_config.get('CODE_GENERATION', 'console_log_level')]
    file_log_level = LOG_LEVELS[cysparse_config.get('CODE_GENERATION', 'file_log_level')]

    logger.setLevel(log_level)

    # create console handler and set logging level
    ch = logging.StreamHandler()
    ch.setLevel(console_log_level)

    # create file handler and set logging level
    log_file_name = logger_name + '.log'
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(file_log_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger