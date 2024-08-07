# Author: David Hudak
# Filename: regex_patterns.py
# Description: Implementation of regex patterns for properties parsing via static class.

import re

class RegexPatterns:
    reward_max = re.compile("\s*R.*max.*")
    reward_min = re.compile("\s*R.*min.*")
    
    @staticmethod
    def check_max_property(property : str = ""):
        return bool(RegexPatterns.reward_max.match(property))
    
    @staticmethod
    def check_min_property(property : str = ""):
        return bool(RegexPatterns.reward_min.match(property))