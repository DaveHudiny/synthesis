from paynt.rl_extension.saynt_rl_tools.agents_wrapper import AgentsWrapper
import numpy as np
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.tools.evaluators import evaluate_extracted_fsc
from rl_src.interpreters.fsc_based_interpreter import ExtractedFSCPolicy
import logging
import re

from tf_agents.trajectories.time_step import TimeStep

from paynt.family.family import Family

logger = logging.getLogger(__name__)


class RLFamilyExtractor:

    def __init__(self):
        pass

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

        def __str__(self):
            as_dict = self.__dict__
            return str(as_dict)

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
    def get_extracted_fsc_policy(agents_wrapper: AgentsWrapper, args: ArgsEmulator):
        extracted_fsc_policy = ExtractedFSCPolicy(agents_wrapper.agent.wrapper, agents_wrapper.agent.environment,
                                                  tf_environment=agents_wrapper.agent.tf_environment, args=args)
        evaluate_extracted_fsc(external_evaluation_result=agents_wrapper.agent.evaluation_result,
                               agent=agents_wrapper.agent,
                               extracted_fsc_policy=extracted_fsc_policy)
        return extracted_fsc_policy

    @staticmethod
    def basic_initial_mem_check(hole_info: 'RLFamilyExtractor.HoleInfo', restricted_family: Family, subfamily_restrictions: list[dict]):
        if hole_info.is_update:
            restricted_family.hole_set_options(hole_info.hole, [0])
            restriction = {"hole": hole_info.hole, "restriction": [0]}
            subfamily_restrictions.append(restriction)
            return True
        elif hole_info.mem_number > 0:
            return True
        else:
            return False

    @staticmethod
    def fill_all_mem_check(hole_info: 'RLFamilyExtractor.HoleInfo', restricted_family: Family, subfamily_restrictions: list[dict]):
        if hole_info.is_update:
            options = hole_info.options
            # random_option = np.random.choice(options)
            # Generate single optinon from geometric distribution -- the first option should have probability 1/2, second 1/4, third 1/8, etc.
            options_range = np.arange(len(options)) + 1.0
            options_prob = 1.0 / (2.0 ** options_range)
            normalized_options_prob = options_prob / np.sum(options_prob)
            random_option = np.random.choice(options, p=normalized_options_prob) 
            subfamily_restrictions.append(
                {"hole": hole_info.hole, "restriction": [random_option]})
            
            restricted_family.hole_set_options(hole_info.hole, [random_option])
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
    def get_rl_action_label(agents_wrapper: AgentsWrapper, rl_action):
        return agents_wrapper.agent.environment.act_to_keywords[rl_action]

    @staticmethod
    def generate_fake_timestep(agents_wrapper: AgentsWrapper, hole_info: 'RLFamilyExtractor.HoleInfo'):
        return agents_wrapper.agent.environment.create_fake_timestep_from_observation_integer(hole_info.observation_integer)

    @staticmethod
    def generate_rl_action(agents_wrapper: AgentsWrapper, fake_time_step, extracted_fsc_policy: ExtractedFSCPolicy,
                           hole_info: 'RLFamilyExtractor.HoleInfo', is_first=False, memory_less=True):
        if is_first and hole_info.mem_number == 0:
            action = extracted_fsc_policy.get_single_action(
                hole_info.observation_integer, hole_info.mem_number)
            action_label = RLFamilyExtractor.get_rl_action_label(
                agents_wrapper, action)
        else:
            # mem_number = hole_info.mem_number if not memory_less else 0
            initial_state = agents_wrapper.agent.wrapper.get_initial_state(1)
            action = agents_wrapper.agent.wrapper.action(fake_time_step, initial_state)
            if not memory_less or hole_info.mem_number == 0:
                extracted_fsc_policy.set_single_action(
                    hole_info.observation_integer, hole_info.mem_number, action.action.numpy()[0])
            action_label = RLFamilyExtractor.get_rl_action_label(
                agents_wrapper, action.action.numpy()[0])
        return action_label

    @staticmethod
    def apply_family_action_restriction(hole_info: 'RLFamilyExtractor.HoleInfo', action_label: str, restricted_family: Family,
                                        subfamily_restrictions: list[dict], memory_only : bool = False):
        index_of_action_label = hole_info.option_labels.index(
            action_label)
        restricted_family.hole_set_options(
            hole_info.hole, [index_of_action_label])
        if not memory_only:
            restriction = {"hole": hole_info.hole,
                           "restriction": [index_of_action_label]}
            subfamily_restrictions.append(restriction)
        else:
            restriction = {"hole": hole_info.hole,
                           "restriction": hole_info.options}
            subfamily_restrictions.append(restriction)
            pass

    @staticmethod
    def set_action_for_complete_miss(hole_info: 'RLFamilyExtractor.HoleInfo', agents_wrapper: AgentsWrapper,
                                     extracted_fsc_policy: ExtractedFSCPolicy, restricted_family: Family,
                                     subfamily_restrictions: list[dict], memory_only : bool = False):
        selected_action = np.random.choice(hole_info.options)
        rl_action = RLFamilyExtractor.convert_hole_option_to_rl_action(
            hole_info, selected_action, agents_wrapper.agent.environment.action_keywords)
        if hole_info.mem_number == 0:
            extracted_fsc_policy.set_single_action(
                hole_info.observation_integer, hole_info.mem_number, rl_action)
        restricted_family.hole_set_options(hole_info.hole, [selected_action])
        if memory_only:
            restriction = {"hole": hole_info.hole,
                        "restriction": [selected_action]}
            subfamily_restrictions.append(restriction)

    @classmethod
    def get_action_greedily(cls, restricted_family: Family, hole_info : HoleInfo, agent_wrapper : AgentsWrapper, fake_timestep : TimeStep):
        reversed_index = -(hole_info.mem_number + 1)
        if not hasattr(cls, "observation_to_logits"):
            cls.observation_to_logits = {}
        if hole_info.observation_integer in cls.observation_to_logits:
            action = np.argsort(cls.observation_to_logits[hole_info.observation_integer])[reversed_index]
            if cls.observation_to_logits[hole_info.observation_integer][action] == -np.inf:
                # pick random action, which is not -np.inf
                action = np.random.choice(np.argwhere(cls.observation_to_logits[hole_info.observation_integer] > -np.inf)[0])
                
        else:
            played_action = agent_wrapper.agent.wrapper.action(fake_timestep, agent_wrapper.agent.wrapper.get_initial_state(1))
            logits = played_action.info["dist_params"]["logits"].numpy()[0]
            allowed_actions = fake_timestep.observation["mask"].numpy()[0]
            logits = np.where(allowed_actions == False, -np.inf, logits)
            cls.observation_to_logits[hole_info.observation_integer] = logits
            action = np.argsort(logits)[reversed_index]
        return action, RLFamilyExtractor.get_rl_action_label(agent_wrapper, action)


    @staticmethod
    def hole_loop_body(hole, restricted_family: Family, subfamily_restrictions: list[dict], agents_wrapper: AgentsWrapper,
                       extracted_fsc_policy: ExtractedFSCPolicy, mem_check: callable = basic_initial_mem_check,
                       memory_less: bool = True, greedy : bool = False, memory_only : bool = False) -> tuple[int, int]:
        hole_info = RLFamilyExtractor.get_hole_info(
            restricted_family, hole)
        if mem_check(hole_info, restricted_family, subfamily_restrictions):
            return 0, 0
        # print(f"Processing hole {hole_info}")
        fake_time_step = RLFamilyExtractor.generate_fake_timestep(
            agents_wrapper, hole_info)
        i = 0
        miss = 0
        complete_miss = 0
        if greedy:
            action, action_label = RLFamilyExtractor.get_action_greedily(restricted_family, hole_info, agents_wrapper, fake_time_step)
            RLFamilyExtractor.apply_family_action_restriction(
                hole_info, action_label, restricted_family, subfamily_restrictions, memory_only)
            return miss, complete_miss
        else:
            while True:

                action_label = RLFamilyExtractor.generate_rl_action(
                    agents_wrapper, fake_time_step,
                    extracted_fsc_policy, hole_info,
                    is_first=(i == 0),
                    memory_less=memory_less)

                if action_label in hole_info.option_labels:
                    RLFamilyExtractor.apply_family_action_restriction(
                        hole_info, action_label, restricted_family, subfamily_restrictions, memory_only)
                    break
                elif i == 0:
                    miss = 1
                elif i > 10:
                    RLFamilyExtractor.set_action_for_complete_miss(
                        hole_info, agents_wrapper, extracted_fsc_policy, restricted_family, subfamily_restrictions, memory_only)
                    complete_miss = 1
                    break
                i += 1
            return miss, complete_miss

    @staticmethod
    def get_restricted_family_rl_inference(original_family: Family, agents_wrapper: AgentsWrapper, args:
                                           ArgsEmulator, fill_all_memory: bool = False, memoryless_rl = True,
                                           greedy : bool = False, memory_only : bool = False) -> tuple[Family, list[dict]]:
        # Copy of the original family, because PAYNT uses the original family to other purposes
        restricted_family = original_family.copy()

        # List of dictionaries with the restrictions of the subfamilies, where the keys are the hole numbers and the values are arrays of selected actions.
        subfamily_restrictions = []

        # Statistics
        num_misses = 0
        num_misses_complete = 0

        logger.info("Building family from FFNN...")

        if not memoryless_rl:
            pass  # TODO: Implement memory version of RLFamilyExtractor

        # Extraction and evaluation of original policy

        extracted_fsc_policy = RLFamilyExtractor.get_extracted_fsc_policy(
            agents_wrapper, args)
        if fill_all_memory:
            mem_check = RLFamilyExtractor.fill_all_mem_check
        for hole in range(restricted_family.num_holes):
            miss, complete_miss = RLFamilyExtractor.hole_loop_body(
                hole, restricted_family, subfamily_restrictions, agents_wrapper, extracted_fsc_policy,
                mem_check=mem_check, memory_less=memoryless_rl, greedy=greedy, memory_only=memory_only)
            num_misses += miss
            num_misses_complete += complete_miss

        extracted_fsc_policy.recompile_tf_tables()
        evaluate_extracted_fsc(external_evaluation_result=agents_wrapper.agent.evaluation_result,
                               agent=agents_wrapper.agent, extracted_fsc_policy=extracted_fsc_policy)
        logger.info(
            f"Number of misses: {num_misses} out of {restricted_family.num_holes}")
        logger.info(
            f"Number of complete misses: {num_misses_complete} out of {restricted_family.num_holes}")
        return restricted_family, subfamily_restrictions