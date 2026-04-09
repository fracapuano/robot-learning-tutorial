import multiprocessing as mp
import signal
from typing import Callable
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.robots.so100_follower import SO100FollowerConfig
from lerobot.teleoperators.so100_leader import SO100LeaderConfig


run_learner: Callable = ...  # use/modify the functions defined earlier
run_actor: Callable = ...

"""Main function - coordinates actor and learner processes."""

device = "mps"  # or "cuda" or "cpu"
output_directory = Path("outputs/robot_learning_tutorial/hil_serl")
output_directory.mkdir(parents=True, exist_ok=True)

# find ports using lerobot-find-port
follower_port = ...
leader_port = ...

# the robot ids are used the load the right calibration files
follower_id = ...
leader_id = ...

# A pretrained model (to be used in-distribution!)
reward_classifier_id = "lerobot/reward_classifier_hil_serl_example"
reward_classifier = Classifier.from_pretrained(reward_classifier_id)

reward_classifier.to(device)
reward_classifier.eval()

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

# Robot and environment configuration
robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id)
teleop_cfg = SO100LeaderConfig(port=leader_port, id=leader_id)
processor_cfg = HILSerlProcessorConfig(control_mode="leader")

env_cfg = HILSerlRobotEnvConfig(robot=robot_cfg, teleop=teleop_cfg, processor=processor_cfg)

# Create robot environment
env, teleop_device = make_robot_env(env_cfg)

obs_features = hw_to_dataset_features(env.robot.observation_features, "observation")
action_features = hw_to_dataset_features(env.robot.action_features, "action")

# Create SAC policy for action selection
policy_cfg = SACConfig(
    device=device,
    input_features=obs_features,
    output_features=action_features,
)

policy_actor = SACPolicy(policy_cfg)
policy_learner = SACPolicy(policy_cfg)

demonstrations_repo_id = "lerobot/example_hil_serl_dataset"
offline_dataset = LeRobotDataset(repo_id=demonstrations_repo_id)

# Online buffer: initialized from scratch
online_replay_buffer = ReplayBuffer(device=device, state_keys=list(obs_features.keys()))
# Offline buffer: Created from dataset (pre-populated it with demonstrations)
offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
    lerobot_dataset=offline_dataset, device=device, state_keys=list(obs_features.keys())
)

# Create communication channels between learner and actor processes
transitions_queue = mp.Queue(maxsize=10)
parameters_queue = mp.Queue(maxsize=2)
shutdown_event = mp.Event()


# Signal handler for graceful shutdown
def signal_handler(sig):
    print(f"\nSignal {sig} received, shutting down...")
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Create processes
learner_process = mp.Process(
    target=run_learner,
    args=(
        transitions_queue,
        parameters_queue,
        shutdown_event,
        policy_learner,
        online_replay_buffer,
        offline_replay_buffer,
    ),
    kwargs={"device": device},  # can run on accelerated hardware for training
)

actor_process = mp.Process(
    target=run_actor,
    args=(
        transitions_queue,
        parameters_queue,
        shutdown_event,
        policy_actor,
        reward_classifier,
        env_cfg,
        output_directory,
    ),
    kwargs={"device": "cpu"},  # actor is frozen, can run on CPU or accelerate for inference
)

learner_process.start()
actor_process.start()

try:
    # Wait for actor to finish (it controls the episode loop)
    actor_process.join()
    shutdown_event.set()
    learner_process.join(timeout=10)

except KeyboardInterrupt:
    print("Main process interrupted")
    shutdown_event.set()
    actor_process.join(timeout=5)
    learner_process.join(timeout=10)

finally:
    if learner_process.is_alive():
        learner_process.terminate()
    if actor_process.is_alive():
        actor_process.terminate()
