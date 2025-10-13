import multiprocessing as mp
from queue import Empty, Full

import torch
import torch.optim as optim

from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer

LOG_EVERY = 10
SEND_EVERY = 10

def run_learner(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    policy_learner: SACPolicy,
    online_buffer: ReplayBuffer,
    offline_buffer: ReplayBuffer,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: torch.device = "mps",
):
    """The learner process - trains SAC policy on transitions streamed from the actor, 
    updating parameters for the actor to adopt."""
    policy_learner.train()
    policy_learner.to(device)

    # Create Adam optimizer from scratch - simple and clean
    optimizer = optim.Adam(policy_learner.parameters(), lr=lr)

    print(f"[LEARNER] Online buffer capacity: {online_buffer.capacity}")
    print(f"[LEARNER] Offline buffer capacity: {offline_buffer.capacity}")

    training_step = 0

    while not shutdown_event.is_set():
        # retrieve incoming transitions from the actor process
        try:
            transitions = transitions_queue.get(timeout=0.1)
            for transition in transitions:
                # HIL-SERL: Add ALL transitions to online buffer
                online_buffer.add(**transition)

                # HIL-SERL: Add ONLY human intervention transitions to offline buffer
                is_intervention = \
                    transition.get("complementary_info", {}).get("is_intervention", False)
                if is_intervention:
                    offline_buffer.add(**transition)
                    print(
                        f"[LEARNER] Human intervention detected!"
                        f"Added to offline buffer (now {len(offline_buffer)} transitions)"
                    )

        except Empty:
            pass  # No transitions available, continue

        # Train if we have enough data
        if len(online_buffer) >= policy_learner.config.online_step_before_learning:
            # Sample from online buffer (autonomous + human data)
            online_batch = online_buffer.sample(batch_size // 2)

            # Sample from offline buffer (human demonstrations only)
            offline_batch = offline_buffer.sample(batch_size // 2)

            # Combine batches - this is the key HIL-SERL mechanism!
            batch = {}
            for key in online_batch.keys():
                if key in offline_batch:
                    batch[key] = torch.cat([online_batch[key], offline_batch[key]], dim=0)
                else:
                    batch[key] = online_batch[key]

            loss, _ = policy_learner.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_step += 1

            if training_step % LOG_EVERY == 0:
                print(
                    f"[LEARNER] Training step {training_step}, Loss: {loss.item():.4f}, "
                    f"Buffers: Online={len(online_buffer)}, Offline={len(offline_buffer)}"
                )

            # Send updated parameters to actor every 10 training steps
            if training_step % SEND_EVERY == 0:
                try:
                    state_dict = {k: v.cpu() for k, v in policy_learner.state_dict().items()}
                    parameters_queue.put_nowait(state_dict)
                    print("[LEARNER] Sent updated parameters to actor")
                except Full:
                    # Missing write due to queue not being consumed (should happen rarely)
                    pass

    print("[LEARNER] Learner process finished")
