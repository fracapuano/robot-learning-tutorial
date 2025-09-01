# Robot Learning: A Tutorial

Google "robot learning tutorial", and you will spend just as much time skimming through sources as actually learning about robot learning.
This tutorial solves this: a unified entry point to the field of robot learning, presenting the conceptual underpinnings of popular approaches in the field, as well as presenting practical examples of how to use SOTA algorithms in `lerobot`, an open-source library for full-stack robotics.

# TODO

```markdown
## 1. Introduction
- [x] 1.1 Motivation
- [x] 1.2 Structure of the Report

## 2. Classical Robotics
- [x] 2.1 Different kinds of motion
- [x] 2.2 Example: (Planar) Manipulation
    - [x] 2.3.1 Adding Feedback Loops
- [x] 2.4 Limitations of Dynamics-based Robotics

## 3. Robot Learning
- [ ] 3.1 Reinforcement Learning (RL) for Robotics
    - [ ] 3.1.1 A (Concise) Introduction to RL
- [ ] 3.2 Model-Free RL for Real-world Robotics
    - [ ] 3.2.1 RL in lerobot: sample efficient, data-driven, and real-world
    - [ ] 3.2.2 Code Example: HIL-SERL in lerobot
- [ ] 3.3 Limitations of RL in Real-World Robotics: Simulators and Reward Design
- [ ] 3.4 Behavioral Cloning (BC) for Robotics
    - [ ] 4.1.1 Leveraging Real-World Demonstrations
    - [ ] 4.1.2 Reward-Free Training and Betting on Data

## 4. Single-Task Policy Architectures
- [ ] 4.2 Action Chunking with Transformers (ACT)
    - [ ] 4.2.1 Model Architecture and Training Objectives
    - [ ] 4.2.2 Code Example: Use ACT in lerobot
- [ ] 4.3 Diffusion-Based Policy Models
    - [ ] 4.3.1 Generative Modeling for Action Sequences
    - [ ] 4.3.2 Code Example: Use Diffusion Policy in lerobot

## 5. Multi-task Policies: Vision-Language-Action (VLA) Models in Robotics
- [ ] 5.1 Multi-task Policies: Vision-Language-Action (VLA) Models in Robotics
    - [ ] 5.1.1 Overview of Major Architectures: Pi0, SmolVLA
    - [ ] 5.1.2 Practical Implementation: Using VLA in lerobot

## 6. (Emerging) Directions in Robot Learning
- [ ] 6.1 VLAs Post-Training
    - [ ] 6.1.1 From Imitation to Refinement
    - [ ] 6.1.2 EXPO

## 7. Conclusions
```

If time permits (vs current TOC):

- [ ] 3.3 Model-based RL for Robotics
    - [ ] 3.3.1 TD-MPC
    - [ ] 3.3.2 Code Example: Use TD-MPC in lerobot
- [ ] 3.5 Popular benchmarks in Robot Learning

- 4.3 Vector-Quantized Behavior Transformer (VQ-BeT)
    - [ ] 4.3.1 Model Architecture and Training Objectives
    - [ ] 4.3.2 Code Example: Use VQ-BeT in lerobot

- [ ] 6.1 Using World Models for Robotics
    - [ ] 6.1.1 In the architecture: V-JEPA and V-JEPA2
    - [ ] 6.1.2 In the simulation: GENIE
