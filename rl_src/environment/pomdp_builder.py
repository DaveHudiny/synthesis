# This file is used to build a POMDP model from a PRISM model and a property.
# Author: David Hudák
# Login: xhudak03
# File: pomdp_builder.py

import stormpy as sp
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import stormpy.pomdp

import logging
logger = logging.getLogger(__name__)


def parse_properties(properties_file):
    """Parses the properties from the properties file. Ignores lines with comments (starting with //).

    Args:
        properties_file (str): The path to the properties file.
    Returns:
        list: The list of properties.
    """
    with open(properties_file, "r") as f:
        lines = f.readlines()
    properties = []
    for line in lines:
        if line.startswith("//"):
            continue
        properties.append(line.strip())
    return properties


class POMDP_arguments:
    def __init__(self, prism, props, constants):
        self.prism = prism
        self.properties = props
        self.constants = constants


class POMDP_builder:
    def build_model(input: POMDP_arguments):
        """ Taken from https://github.com/stevencarrau/safe_RL_POMDPs.
        This function is used to build a POMDP model from a PRISM model and a property.

        Args:
            input (POMDP_arguments): The input arguments for the POMDP model.

        Returns:
            model: The POMDP model.
            prism_program: The PRISM program.
            raw_formula: The raw formula.
        """
        prism_program = sp.parse_prism_program(input.prism)
        prop = sp.parse_properties_for_prism_program(
            input.properties[0], prism_program)[0]
        prism_program, props = stormpy.preprocess_symbolic_input(
            prism_program, [prop], input.constants)
        prop = props[0]
        prism_program = prism_program.as_prism_program()
        raw_formula = prop.raw_formula
        logger.info("Construct POMDP representation...")
        model = POMDP_builder.build_pomdp(prism_program, raw_formula)
        model = sp.pomdp.make_canonic(model)
        return model

    def build_pomdp(program, formula):
        """ Taken from https://github.com/stevencarrau/safe_RL_POMDPs.
        This function is used to build a POMDP model from a PRISM program and a formula.

        Args:
            program: The PRISM program.
            formula: The formula.

        Returns:
            The POMDP model.
        """
        options = stormpy.BuilderOptions([formula])
        options.set_build_state_valuations()
        options.set_build_choice_labels()
        options.set_build_all_labels()
        options.set_build_all_reward_models()
        options.set_build_observation_valuations()
        logger.debug("Start building the POMDP")
        return sp.build_sparse_model_with_options(program, options)


if __name__ == "__main__":
    print("Testing POMDP builder")
