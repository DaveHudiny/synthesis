import paynt.parser.sketch
import os
import random

def load_sketch(project_path):
    project_path = os.path.abspath(project_path)
    sketch_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")    
    quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    return quotient

def random_action_and_obs(quotient, belief):
    ''' for testing purposes only: pick a random action (label) and a random next observation '''
    state = random.choices(list(belief.keys()),weights=list(belief.values()))[0]
    obs = quotient.pomdp.observations[state]
    action_label = random.choice(quotient.action_labels_at_observation[obs])
    action = quotient.action_labels_at_observation[obs].index(action_label)
    choice = quotient.pomdp.get_choice_index(state,action)
    action_states = []
    action_probs = []
    for entry in quotient.pomdp.transition_matrix.get_row(choice):
        action_states.append(entry.column)
        action_probs.append(entry.value())
    next_state = random.choices(action_states, weights=action_probs)[0]
    next_obs = quotient.pomdp.observations[next_state]
    return action_label,next_obs

def main():
    # project_path = "models/pomdp/large/geo-2-8"
    # project_path = "models/pomdp/large/maze-10" # peculiar beliefs...
    project_path = "models/pomdp/large/network-5-10-8"
    # project_path = "models/pomdp/large/rocks-4-20"
    quotient = load_sketch(project_path)

    # belief is a dictionary of state-probability pairs
    belief = {quotient.pomdp.initial_states[0]:1}
    for _ in range(10):
        action_label,next_obs = random_action_and_obs(quotient,belief)
        belief = quotient.next_belief(belief, action_label, next_obs)
        print(f"{action_label} + {next_obs} = {belief}")

main()
