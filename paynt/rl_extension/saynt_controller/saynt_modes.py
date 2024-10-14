from enum import Enum

import logging

logger = logging.getLogger(__name__)

class SAYNT_Modes(Enum):
    BELIEF = 1
    CUTOFF_FSC = 2
    CUTOFF_SCHEDULER = 3