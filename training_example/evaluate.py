"""
General Introduction:
    * the objective of this script is that students could see evaluation process of their models against other RL models
    * students could evaluate their best trained models with this script
    * the evaluation is done againts uploaded default best trained models with sac with default parameters
    * students should give path to their best models in LOAD_CUSTOM_MODEL
    * opponent vehicle number could be changed in OPPONENT_NUM and their initial racing positions in AGENT_LOCATIONS
    * by changing CONTROL_OTHER_AGENTS boolean students could evaluate their models against default RL trained model or IDM (autopilot) vehicles
    * its important to modify load_checkpoint() function if your model's network structure is not default Soft-Actor-Critic Network
    * evaluation in each step is done until NUM_EVAL_STEPS iteration number is reached, however this could be changed
    * in the competition, student will race their models against each others' RL models
"""

import torch
import simstar
import numpy as np
from time import time
from simstarEnv import SimstarEnv
from collections import namedtuple
from sac_agent import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# user's own best model could be loaded from saved_models folder
# TODO: right now default model is loaded, however users should evaluate their own models
LOAD_CUSTOM_MODEL = "default_model/default_model.dat"

# default model is trained with given sac agent code and training is breaked at 320K steps
LOAD_DEFAULT_MODEL = "default_model/default_model.dat"

NUM_EVAL_EPISODE = 2
NUM_EVAL_STEPS = 4000

ADD_OPPONENTS = True
OPPONENT_NUM = 3

# True: controls opponent vehicles with loaded default model weights
# False: opponent vehicles will be controled with IDM (Intelligent Driver Model)
CONTROL_OTHER_AGENTS = True

# initial locations of the opponents is defined pose data in the simulator
AGENT_LOCATIONS = [
    simstar.PoseData(1402.143066, -880.167114, yaw=-3.08),
    simstar.PoseData(1269.950562, -888.293701, yaw=-3.08),
    simstar.PoseData(1046.40271, -887.010681, yaw=-3.10),
]
if CONTROL_OTHER_AGENTS:
    AGENT_INIT_SPEEDS = [0, 0, 0, 0, 0]
else:
    AGENT_INIT_SPEEDS = [45, 80, 55, 100, 40]


def evaluate(port=8080):
    env = SimstarEnv(
        track=simstar.Environments.Racing,
        create_track=False,
        port=port,
        add_opponents=ADD_OPPONENTS,
        num_opponents=OPPONENT_NUM,
        opponent_pose_data=AGENT_LOCATIONS,
        speed_up=2,
        synronized_mode=True,
    )

    # update agent init configs
    env.agent_speeds = AGENT_INIT_SPEEDS

    # total length of chosen observation states
    insize = 4 + env.track_sensor_size + env.opponent_sensor_size
    outsize = env.action_space.shape[0]

    hyperparams = {
        "lrvalue": 0.0005,
        "lrpolicy": 0.0001,
        "gamma": 0.97,
        "episodes": 15000,
        "buffersize": 250000,
        "tau": 0.001,
        "batchsize": 64,
        "alpha": 0.2,
        "maxlength": 10000,
        "hidden": 256,
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)

    # load actor network from checkpoint
    agent = Model(env=env, params=hyprm, n_insize=insize, n_outsize=outsize)
    load_checkpoint(agent,LOAD_CUSTOM_MODEL)

    if CONTROL_OTHER_AGENTS:
        opponent_agent = Model(
            env=env, params=hyprm, n_insize=insize, n_outsize=outsize
        )
        load_checkpoint(opponent_agent,LOAD_DEFAULT_MODEL)

    total_reward = 0
    start_time = time()

    obs = env.reset()
    state = np.hstack(
        (obs.angle, obs.track, obs.trackPos, obs.speedX, obs.speedY, obs.opponents)
    )

    agent_observations = env.get_agent_observations()
    if CONTROL_OTHER_AGENTS:
        env.change_opponent_control_to_api()

    agent_actions = []

    epsisode_reward = 0

    while True:
        action = np.array(agent.select_action(state=state))

        if CONTROL_OTHER_AGENTS:
            # set other agent actions
            env.set_agent_actions(agent_actions)

        obs, reward, done, summary = env.step(action)
        
        # get other agent observation
        agent_observations = env.get_agent_observations()
        
        progress = env.get_progress_on_road()
        next_state = np.hstack(
            (
                obs.angle,
                obs.speedX,
                obs.speedY,
                obs.opponents,
                obs.track,
                obs.trackPos,
            )
        )

        if CONTROL_OTHER_AGENTS:
            agent_actions = []
            for agent_obs in agent_observations:
                agent_state = np.hstack(
                    (
                        agent_obs.angle,
                        agent_obs.speedX,
                        agent_obs.speedY,
                        agent_obs.opponents,
                        agent_obs.track,
                        agent_obs.trackPos,
                    )
                )
                agent_action = np.array(opponent_agent.select_action(state=agent_state))
                agent_actions.append(agent_action)

            

        epsisode_reward += reward

        if done:
            pass

        if progress > 1.0:
            current_time = time()
            elapsed_time = current_time - start_time
            print(f"One lap is done, total time {elapsed_time}")
            break

        state = next_state


def load_checkpoint(agent,path):
    try:
        checkpoint = torch.load(path)
        print("keys are: ", checkpoint.keys())

        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
        agent.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
        agent.eval()
        if "epsisode_reward" in checkpoint:
            reward = float(checkpoint["epsisode_reward"])
    
    except FileNotFoundError:
        raise FileNotFoundError("custom model weights are not found")


if __name__ == "__main__":
    evaluate()
