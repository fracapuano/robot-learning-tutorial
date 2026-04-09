import multiprocessing as mp
from queue import Empty

import torch
from pathlib import Path

from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators.utils import TeleopEvents

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

def make_policy_obs(obs, device: torch.device = "cpu"):
    return {
        "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device),
        **{
            f"observation.image.{k}": 
                torch.from_numpy(obs["pixels"][k]).float().unsqueeze(0).to(device)
            for k in obs["pixels"]
        },
    }

def run_actor(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    policy_actor: SACPolicy,
    reward_classifier: Classifier,
    env_cfg: HILSerlRobotEnvConfig,
    device: torch.device = "mps",
    output_directory: Path | None = None
):
    """The actor process - interacts with environment and collects data.
    The policy is frozen and only the parameters are updated, popping the most recent ones 
    from a queue."""
    policy_actor.eval()
    policy_actor.to(device)

    reward_classifier.eval()
    reward_classifier.to(device)

    # Create robot environment inside the actor process
    env, teleop_device = make_robot_env(env_cfg)

    try:
        for episode in range(MAX_EPISODES):
            if shutdown_event.is_set():
                break

            obs, _info = env.reset()
            episode_reward = 0.0
            step = 0
            episode_transitions = []

            print(f"[ACTOR] Starting episode {episode + 1}")

            while step < MAX_STEPS_PER_EPISODE and not shutdown_event.is_set():
                try:
                    new_params = parameters_queue.get_nowait()
                    policy_actor.load_state_dict(new_params)
                    print("[ACTOR] Updated policy parameters from learner")
                except Empty:  # No new updated parameters available from learner, waiting
                    pass

                # Get action from policy
                policy_obs = make_policy_obs(obs, device=device)
                # predicts a single action, not a chunk of actions!
                action_tensor = policy_actor.select_action(policy_obs)
                action = action_tensor.squeeze(0).cpu().numpy()

                # Step environment
                next_obs, _env_reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated

                # Predict reward
                policy_next_obs = make_policy_obs(next_obs, device=device)
                reward = reward_classifier.predict_reward(policy_next_obs)

                if reward >= 1.0:  # success detected! halt episode
                    if not done:
                        terminated = True
                        done = True

                # In HIL-SERL, human interventions come from the teleop device
                is_intervention = False
                if hasattr(teleop_device, "get_teleop_events"):
                    # Real intervention detection from teleop device
                    teleop_events = teleop_device.get_teleop_events()
                    is_intervention = teleop_events.get(TeleopEvents.IS_INTERVENTION, False)

                # Store transition with intervention metadata
                transition = {
                    "state": policy_obs,
                    "action": action,
                    "reward": float(reward) if hasattr(reward, "item") else reward,
                    "next_state": policy_next_obs,
                    "done": done,
                    "truncated": truncated,
                    "complementary_info": {
                        "is_intervention": is_intervention,
                    },
                }

                episode_transitions.append(transition)

                episode_reward += reward
                step += 1

                obs = next_obs

                if done:
                    break

            # Send episode transitions to learner
            transitions_queue.put_nowait(episode_transitions)

    except KeyboardInterrupt:
        print("[ACTOR] Interrupted by user")
    finally:
        # Clean up
        if hasattr(env, "robot") and env.robot.is_connected:
            env.robot.disconnect()
        if teleop_device and hasattr(teleop_device, "disconnect"):
            teleop_device.disconnect()
        if output_directory is not None:
            policy_actor.save_pretrained(output_directory)
            print(f"[ACTOR] Latest actor policy saved at: {output_directory}")
        
        print("[ACTOR] Actor process finished")