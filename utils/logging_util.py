# -*- coding: utf-8 -*-

# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import logging


def get_std_logging(file_path):
    """ Make python logger """
    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    logger = logging.getLogger('H-DAS')
    log_format = '%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S%p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger
