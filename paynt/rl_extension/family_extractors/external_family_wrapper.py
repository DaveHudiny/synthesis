from paynt.family.family import Family
from paynt.rl_extension.family_extractors.rl_family_extractor import RLFamilyExtractor

from paynt.rl_extension.saynt_rl_tools.agents_wrapper import AgentsWrapper

class ExtractedFamilyWrapper:
    def __init__(self, family: Family, memory_size, agents_wrapper : AgentsWrapper):
        family_w_restrictions = RLFamilyExtractor.get_restricted_family_rl_inference(family, agents_wrapper, agents_wrapper.agent.args, True, True)
        self.extracted_family = family_w_restrictions[0]
        self.subfamily_restrictions = family_w_restrictions[1]
        self.memory_size = memory_size

    def get_family(self):
        return self.extracted_family
    
    def get_subfamily_restrictions(self):
        return self.subfamily_restrictions
    
    def get_memory_size(self):
        return self.memory_size