from paynt.synthesizer.synthesizer_rl import Synthesizer_RL
import numpy as np
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.tools.evaluators import evaluate_extracted_fsc
from rl_src.interpreters.fsc_based_interpreter import ExtractedFSCPolicy
import logging
import re

from paynt.family.family import Family

logger = logging.getLogger(__name__)


class RLFamilyExtractor:

    class HoleInfo:
        def __init__(self, hole: int, vector: list[float], mem_number: int, is_update: bool, observation_integer: int,
                     options: list[int], option_labels: list[str]):
            self.hole = hole
            self.vector = vector
            self.mem_number = mem_number
            self.is_update = is_update
            self.observation_integer = observation_integer
            self.options = options
            self.option_labels = option_labels

    @staticmethod
    def parse_hole_observation_info(family: Family, hole: int):

        name = family.hole_name(hole)
        if name[0] == "M":
            is_update = True
        else:
            is_update = False
        vector_match = re.search(r'\[([^\]]+)\]', name)
        if vector_match:
            vector_string = vector_match.group(1)
            vector = []
            for item in vector_string.split('&'):
                if "=" not in item:
                    if "!" in item:
                        vector.append(float(-1.0))
                    else:
                        vector.append(float(1.0))
                else:
                    _, value = item.split('=')
                    value = value.strip()
                    vector.append(float(value))
        last_number_match = re.search(r',(\d+)\)$', name)
        if last_number_match:
            last_number = int(last_number_match.group(1))
        return vector, last_number, is_update

    @staticmethod
    def get_hole_info(restricted_family: Family, hole: int) -> 'RLFamilyExtractor.HoleInfo':
        vector, last_number, is_update = RLFamilyExtractor.parse_hole_observation_info(
            restricted_family, hole)
        options = restricted_family.hole_options(hole)
        labels = [str(restricted_family.hole_to_option_labels[hole][option])
                  for option in options]
        return RLFamilyExtractor.HoleInfo(hole, vector, last_number, is_update,
                                                    restricted_family.get_hole_observation_index(
                                                        hole),
                                                    options, labels)

    @staticmethod
    def get_extracted_fsc_policy(rl_synthesizer: Synthesizer_RL, args: ArgsEmulator):
        extracted_fsc_policy = ExtractedFSCPolicy(rl_synthesizer.agent.wrapper, rl_synthesizer.agent.environment,
                                                  tf_environment=rl_synthesizer.agent.tf_environment, args=args)
        evaluate_extracted_fsc(external_evaluation_result=rl_synthesizer.agent.evaluation_result,
                               agent=rl_synthesizer.agent,
                               extracted_fsc_policy=extracted_fsc_policy)
        return extracted_fsc_policy

    @staticmethod
    def basic_initial_mem_check(hole_info: 'RLFamilyExtractor.HoleInfo', restricted_family: Family, subfamily_restrictions: list[dict]):
        if hole_info.is_update or hole_info.mem_number == 0:
            restricted_family.hole_set_options(hole_info.hole, [0])
            restriction = {"hole": hole_info.hole, "restriction": [0]}
            subfamily_restrictions.append(restriction)
            return True
        elif hole_info.mem_number > 0:
            return True
        else:
            return False

    @staticmethod
    def convert_rl_action_to_hole_option(hole_info: 'RLFamilyExtractor.HoleInfo', action: int, act_keywords: list) -> int:
        action_label = act_keywords[action]
        return hole_info.option_labels.index(action_label)

    @staticmethod
    def convert_hole_option_to_rl_action(hole_info: 'RLFamilyExtractor.HoleInfo', option: int, act_keywords: list) -> int:
        action_label = hole_info.option_labels[option]
        return act_keywords.index(action_label)

    @staticmethod
    def get_rl_action_label(rl_synthesizer: Synthesizer_RL, rl_action):
        return rl_synthesizer.agent.environment.act_to_keywords[rl_action]

    @staticmethod
    def generate_fake_timestep(rl_synthesizer: Synthesizer_RL, hole_info: 'RLFamilyExtractor.HoleInfo'):
        return rl_synthesizer.agent.environment.create_fake_timestep_from_observation_integer(hole_info)

    @staticmethod
    def generate_rl_action(rl_synthesizer: Synthesizer_RL, fake_time_step, extracted_fsc_policy : ExtractedFSCPolicy,
                           hole_info: 'RLFamilyExtractor.HoleInfo', is_first=False):
        if is_first:
            action = extracted_fsc_policy.get_single_action(
                hole_info.observation_integer, hole_info.mem_number)
            action_label = RLFamilyExtractor.get_rl_action_label(rl_synthesizer, action)
        else:
            action = rl_synthesizer.agent.wrapper.action(fake_time_step)
            extracted_fsc_policy.set_single_action(
                hole_info.observation_integer, hole_info.mem_number, action.action.numpy()[0])
            action_label = RLFamilyExtractor.get_rl_action_label(rl_synthesizer, action.action.numpy()[0])
        return action_label
    
    @staticmethod
    def apply_family_action_restriction(hole_info: 'RLFamilyExtractor.HoleInfo', action_label : str, restricted_family: Family,
                                  subfamily_restrictions: list[dict]):
                index_of_action_label = hole_info.option_labels.index(
                    action_label)
                restricted_family.hole_set_options(
                    hole_info.hole, [index_of_action_label])
                restriction = {"hole": hole_info.hole,
                               "restriction": [index_of_action_label]}
                subfamily_restrictions.append(restriction)
        
    
    @staticmethod
    def set_action_for_complete_miss(hole_info: 'RLFamilyExtractor.HoleInfo', rl_synthesizer: Synthesizer_RL,
                                     extracted_fsc_policy: ExtractedFSCPolicy, restricted_family: Family,
                                     subfamily_restrictions: list[dict]):
        selected_action = np.random.choice(hole_info.options)
        rl_action = RLFamilyExtractor.convert_hole_option_to_rl_action(
            hole_info, selected_action, rl_synthesizer.agent.environment.act_to_keywords)
        extracted_fsc_policy.set_single_action(
            hole_info.observation_integer, hole_info.mem_number, rl_action)
        restricted_family.hole_set_options(hole_info.hole, [selected_action])
        restriction = {"hole": hole_info.hole, "restriction": [selected_action]}
        subfamily_restrictions.append(restriction)

    @staticmethod
    def hole_loop_body(hole, restricted_family: Family, subfamily_restrictions: list[dict], rl_synthesizer: Synthesizer_RL,
                       extracted_fsc_policy: ExtractedFSCPolicy, mem_check: callable = basic_initial_mem_check) -> tuple[int, int]:
        hole_info = RLFamilyExtractor.get_hole_info(
            restricted_family, hole)
        if mem_check(hole_info, restricted_family, subfamily_restrictions):
            return 0, 0
        fake_time_step = RLFamilyExtractor.generate_fake_timestep(hole_info)
        i = 0
        miss = 0
        complete_miss = 0
        while True:

            action_label = RLFamilyExtractor.generate_rl_action(
                rl_synthesizer, fake_time_step, 
                extracted_fsc_policy, hole_info, 
                is_first = (i == 0))
            
            if action_label in hole_info.option_labels:
                RLFamilyExtractor.apply_family_action_restriction(
                    hole_info, action_label, restricted_family, subfamily_restrictions)
                break
            elif i == 0:
                miss = 1
            elif i > 10:
                RLFamilyExtractor.set_action_for_complete_miss(
                    hole_info, rl_synthesizer, extracted_fsc_policy, restricted_family, subfamily_restrictions)
                complete_miss = 1
                break
            i += 1
        return miss, complete_miss

    @staticmethod
    def get_restricted_family_rl_inference(original_family: Family, rl_synthesizer: Synthesizer_RL, args: ArgsEmulator):
        # Copy of the original family, because PAYNT uses the original family to other purposes
        restricted_family = original_family.copy()

        # List of dictionaries with the restrictions of the subfamilies, where the keys are the hole numbers and the values are arrays of selected actions.
        subfamily_restrictions = []

        # Statistics
        num_misses = 0
        num_misses_complete = 0

        logger.info("Building family from FFNN...")

        # Extraction and evaluation of original policy
        extracted_fsc_policy = RLFamilyExtractor.get_extracted_fsc_policy(
            rl_synthesizer, args)

        for hole in range(restricted_family.num_holes):
            miss, complete_miss = RLFamilyExtractor.hole_loop_body(
                hole, restricted_family, subfamily_restrictions, rl_synthesizer, extracted_fsc_policy)
            num_misses += miss
            num_misses_complete += complete_miss

        extracted_fsc_policy.recompile_tf_tables()
        evaluate_extracted_fsc(external_evaluation_result=rl_synthesizer.agent.evaluation_result,
                               agent=rl_synthesizer.agent, extracted_fsc_policy=extracted_fsc_policy)
        logger.info(
            f"Number of misses: {num_misses} out of {restricted_family.num_holes}")
        logger.info(
            f"Number of complete misses: {num_misses_complete} out of {restricted_family.num_holes}")
        return restricted_family, subfamily_restrictions
