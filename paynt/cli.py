from . import version

import paynt.parser.sketch

import paynt.quotient
import paynt.quotient.pomdp
import paynt.quotient.decpomdp
import paynt.quotient.storm_pomdp_control

import paynt.synthesizer.all_in_one
import paynt.synthesizer.synthesizer
import paynt.synthesizer.synthesizer_cegis
import paynt.synthesizer.policy_tree

import click
import sys
import os
import cProfile, pstats

import logging

import pickle

logger = logging.getLogger(__name__)


def setup_logger(log_path = None):
    ''' Setup routine for logging. '''
    
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # root.setLevel(logging.INFO)

    # formatter = logging.Formatter('%(asctime)s %(threadName)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s')

    handlers = []
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        handlers.append(fh)
    sh = logging.StreamHandler(sys.stdout)
    handlers.append(sh)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    for h in handlers:
        root.addHandler(h)
    return handlers


@click.command()
@click.argument('project', type=click.Path(exists=True))
@click.option("--sketch", default="sketch.templ", show_default=True,
    help="name of the sketch file in the project")
@click.option("--props", default="sketch.props", show_default=True,
    help="name of the properties file in the project")
@click.option("--relative-error", type=click.FLOAT, default="0", show_default=True,
    help="relative error for optimal synthesis")
@click.option("--optimum-threshold", type=click.FLOAT,
    help="known optimum bound")
@click.option("--precision", type=click.FLOAT, default=1e-4,
    help="model checking precision")

@click.option("--export",
    type=click.Choice(['jani', 'drn', 'pomdp']),
    help="export the model to specified format and abort")

@click.option("--method",
    type=click.Choice(['onebyone', 'ar', 'cegis', 'hybrid', 'ar_multicore']),
    default="ar", show_default=True,
    help="synthesis method"
    )

@click.option("--disable-expected-visits", is_flag=True, default=False,
    help="do not compute expected visits for the splitting heuristic")

@click.option("--fsc-synthesis", is_flag=True, default=False,
    help="enable incremental synthesis of FSCs for a (Dec-)POMDP")
@click.option("--fsc-memory-size", default=1, show_default=True,
    help="implicit memory size for (Dec-)POMDP FSCs")
@click.option("--posterior-aware", is_flag=True, default=False,
    help="unfold MDP taking posterior observation of into account")

@click.option("--storm-pomdp", is_flag=True, default=False,
    help="enable running belief analysis in STorm to enhance FSC synthesis for POMDPs (AR only)")
@click.option(
    "--storm-options",
    default="cutoff",
    type=click.Choice(["cutoff", "clip2", "clip4", "small", "refine", "overapp", "2mil", "5mil", "10mil", "20mil", "30mil", "50mil"]),
    show_default=True,
    help="run Storm using pre-defined settings and use the result to enhance PAYNT. Can only be used together with --storm-pomdp flag")
@click.option("--iterative-storm", nargs=3, type=int, show_default=True, default=None,
    help="runs the iterative PAYNT/Storm integration. Arguments timeout, paynt_timeout, storm_timeout. Can only be used together with --storm-pomdp flag")
@click.option("--get-storm-result", default=None, type=int,
    help="runs PAYNT for given amount of seconds and returns Storm result using FSC at cutoff. If time is 0 returns pure Storm result. Can only be used together with --storm-pomdp flag")
@click.option("--prune-storm", is_flag=True, default=False,
    help="only explore the main family suggested by Storm in each iteration. Can only be used together with --storm-pomdp flag. Can only be used together with --storm-pomdp flag")
@click.option("--use-storm-cutoffs", is_flag=True, default=False,
    help="use storm randomized scheduler cutoffs are used during the prioritization of families. Can only be used together with --storm-pomdp flag. Can only be used together with --storm-pomdp flag")
@click.option(
    "--unfold-strategy-storm",
    default="storm",
    type=click.Choice(["storm", "paynt", "cutoff"]),
    show_default=True,
    help="specify memory unfold strategy. Can only be used together with --storm-pomdp flag")

@click.option("--export-fsc-storm", type=click.Path(), default=None,
    help="path to output file for SAYNT belief FSC")
@click.option("--export-fsc-paynt", type=click.Path(), default=None,
    help="path to output file for SAYNT inductive FSC")
@click.option("--export-evaluation", type=click.Path(), default=None,
    help="base filename to output evaluation result")

@click.option(
    "--all-in-one", type=click.Choice(["sparse", "bdd"]), default=None, show_default=True,
    help="use all-in-one MDP abstraction",
)
@click.option("--all-in-one-maxmem", default=4096, type=int,
    help="memory limit (MB) for the all-in-one abstraction")

@click.option("--mdp-split-wrt-mdp", is_flag=True, default=False,
    help="# if set, MDP abstraction scheduler will be used for splitting, otherwise game abstraction scheduler will be used")
@click.option("--mdp-discard-unreachable-choices", is_flag=True, default=False,
    help="# if set, unreachable choices will be discarded from the splitting scheduler")
@click.option("--mdp-use-randomized-abstraction", is_flag=True, default=False,
    help="# if set, randomized abstraction guess-and-verify will be used instead of game abstraction;" +
    " MDP abstraction scheduler will be used for splitting"
)

@click.option(
    "--constraint-bound", type=click.FLOAT, help="bound for creating constrained POMDP for cassandra models",
)

@click.option(
    "--ce-generator", type=click.Choice(["dtmc", "mdp"]), default="dtmc", show_default=True,
    help="counterexample generator",
)
@click.option("--profiling", is_flag=True, default=False,
    help="run profiling")

@click.option("--reinforcement-learning", is_flag=True, default=False,
              help="Run with reinforcement learning oracle. Compatibile only with options --fsc-synthesis and --storm-pomdp. Enables following options.")
@click.option("--load-agent", is_flag=True, default=False,
              help="Load the oracle, only works with --reinforcement-learning. If not set, the oracle will be trained before PAYNT synthesis.")
@click.option("--rl-load-memory-flag", is_flag=True, default=False,
              help="Load the memory of the RL oracle. Only works with --load-agent.")
@click.option("--fsc-cycling", is_flag=True, default=False,
              help="Run the FSC cycling oracle. Not compatible with --load-agent.")
@click.option("--fsc-synthesis-time-limit", type=click.INT, default=60,
              help="Limit the time for FSC cycling . Default is 60.")
@click.option("--soft-fsc", is_flag=True, default=False,
              help="Use soft FSC cycling oracle. Default is hard.")
@click.option("--fsc-training-iterations", type=click.INT, default=100,
                help="Number of training iterations with FSC oracle. Default is 100.")
@click.option("--rl-pretrain-iters", type=click.INT, default=500,
                help="Number of pretraining iterations with RL oracle. Default is 500.")
@click.option("--rl-training-iters", type=click.INT, default=300,
                help="Number of training iterations with RL oracle. Default is 300.")
@click.option("--fsc-multiplier", type=click.FLOAT, default=2.0,
                help="Multiplier for the FSC cycling oracle. Default is 2.0.")
@click.option("--rl-load-path", type=click.Path(), default="./",
                help="Path to the RL oracle to load. Have to be used with --load-agent.")



def paynt_run(
    project, sketch, props, relative_error, optimum_threshold, precision,
    export,
    method,
    disable_expected_visits,
    fsc_synthesis, fsc_memory_size, posterior_aware,
    storm_pomdp, iterative_storm, get_storm_result, storm_options, prune_storm,
    use_storm_cutoffs, unfold_strategy_storm,
    export_fsc_storm, export_fsc_paynt, export_evaluation,
    all_in_one, all_in_one_maxmem,
    mdp_split_wrt_mdp, mdp_discard_unreachable_choices, mdp_use_randomized_abstraction,
    constraint_bound,
    ce_generator,
    profiling,
    reinforcement_learning, load_agent, fsc_cycling, fsc_synthesis_time_limit, soft_fsc,
    fsc_training_iterations, rl_pretrain_iters, rl_training_iters, fsc_multiplier, rl_load_path,
    rl_load_memory_flag):

    if reinforcement_learning and not (fsc_synthesis or storm_pomdp):
        logger.error("Reinforcement learning oracle can be used only with FSC synthesis or Storm POMDP.")
        sys.exit(1)
    elif reinforcement_learning:
        rl_input_dictionary = {
            "reinforcement_learning": reinforcement_learning,
            "load_agent": load_agent,
            "fsc_cycling": fsc_cycling,
            "fsc_synthesis_time_limit": fsc_synthesis_time_limit,
            "soft_fsc": soft_fsc,
            "fsc_training_iterations": fsc_training_iterations,
            "rl_pretrain_iters": rl_pretrain_iters,
            "rl_training_iters": rl_training_iters,
            "fsc_multiplier": fsc_multiplier,
            "rl_load_path": rl_load_path,
            "rl_load_memory_flag": rl_load_memory_flag
        }
    profiler = None
    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    logger.info("This is Paynt version {}.".format(version()))

    # set CLI parameters
    paynt.quotient.quotient.Quotient.disable_expected_visits = disable_expected_visits
    paynt.synthesizer.synthesizer_cegis.SynthesizerCEGIS.conflict_generator_type = ce_generator
    paynt.quotient.pomdp.PomdpQuotient.initial_memory_size = fsc_memory_size
    paynt.quotient.pomdp.PomdpQuotient.posterior_aware = posterior_aware
    paynt.quotient.decpomdp.DecPomdpQuotient.initial_memory_size = fsc_memory_size

    paynt.synthesizer.policy_tree.SynthesizerPolicyTree.split_wrt_mdp_scheduler = mdp_split_wrt_mdp
    paynt.synthesizer.policy_tree.SynthesizerPolicyTree.discard_unreachable_choices = mdp_discard_unreachable_choices
    paynt.synthesizer.policy_tree.SynthesizerPolicyTree.use_randomized_abstraction = mdp_use_randomized_abstraction

    storm_control = None
    if storm_pomdp:
        storm_control = paynt.quotient.storm_pomdp_control.StormPOMDPControl()
        storm_control.set_options(
            storm_options, get_storm_result, iterative_storm, use_storm_cutoffs,
            unfold_strategy_storm, prune_storm, export_fsc_storm, export_fsc_paynt
        )

    sketch_path = os.path.join(project, sketch)
    properties_path = os.path.join(project, props)
    if all_in_one is None:
        quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path, export, relative_error, precision, constraint_bound)
        synthesizer = paynt.synthesizer.synthesizer.Synthesizer.choose_synthesizer(quotient, method, fsc_synthesis, storm_control)
        if reinforcement_learning:
            synthesizer.set_reinforcement_learning(rl_input_dictionary)
        synthesizer.run(optimum_threshold, export_evaluation)
    else:
        all_in_one_program, specification, family = paynt.parser.sketch.Sketch.load_sketch_as_all_in_one(sketch_path, properties_path)
        all_in_one_analysis = paynt.synthesizer.all_in_one.AllInOne(all_in_one_program, specification, all_in_one, all_in_one_maxmem, family)
        all_in_one_analysis.run()
    if profiling:
        profiler.disable()
        print_profiler_stats(profiler)

def print_profiler_stats(profiler):
    stats = pstats.Stats(profiler)
    NUM_LINES = 20

    logger.debug("cProfiler info:")
    stats.sort_stats('tottime').print_stats(NUM_LINES)

    logger.debug("percentage breakdown:")
    entries = [ (key,data[2]) for key,data in stats.stats.items()]
    entries = sorted(entries, key=lambda x : x[1], reverse=True)
    entries = entries[:NUM_LINES]
    for key,data in entries:
        module,line,method = key
        if module == "~":
            callee = method
        else:
            callee = f"{module}:{line}({method})"
        percentage = round(data / stats.total_tt * 100,1)
        percentage = str(percentage).ljust(4)
        print(f"{percentage} %  {callee}")

def main():
    setup_logger()
    paynt_run()


if __name__ == "__main__":
    main()
