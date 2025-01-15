from paynt.family.family import Family
from paynt.rl_extension.family_extractors.rl_family_extractor import RLFamilyExtractor

from paynt.synthesizer.synthesizer_agents import Synthesizer_Agents

class ExtractedFamilyWrapper:
    def __init__(self, family: Family, memory_size, rl_synthesiser : Synthesizer_Agents):
        family_w_restrictions = RLFamilyExtractor.get_restricted_family_rl_inference(family, rl_synthesiser, rl_synthesiser.agent.args, True, True)
        self.extracted_family = family_w_restrictions[0]
        self.subfamily_restrictions = family_w_restrictions[1]
        self.memory_size = memory_size

    def get_family(self):
        return self.extracted_family
    
    def get_subfamily_restrictions(self):
        return self.subfamily_restrictions
    
    def get_memory_size(self):
        return self.memory_size