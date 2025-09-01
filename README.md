TODO:

## 1. Introduction

- [] 1.1 Motivation: The Interdisciplinary Nature of Robotics in the Machine Learning Era
- [] 1.2 Scope and Contributions of this work
- [] 1.3 Structure of the Report

## 2. Classical Robotics

- [] 2.1 Artificial motion
- [] 2.2 Different kinds of motion: locomotion, manipulation, and whole-body control
- [] 2.3 Case study: (Planar) Manipulation
    - [] 2.3.1 Overcoming estimation error via feedback loops (controller gains)
- [] 2.4 Limitations: modeling is fragile, not scalable, and does not leverage growing data

## 3. Robot Learning

- [] 3.1 Reinforcement Learning (RL) for Robotic Control
    - [] 3.1.1 Problem Formulation and Control Objectives
    - [] 3.1.2 Policy Optimization Methods in Robotics
    - [] 3.1.3 Practical Implementation: Training RL Policies with LeRobot
    - [] 3.1.4 Limitations of RL in Real-World Robotics: Simulators and Reward Design
- [] 3.2 Imitation Learning (IL) for Robotics
    - [] 3.2.1 Leveraging Real-World Demonstrations
    - [] 3.2.2 Reward-Free Training and Betting on Data
    - [] 3.2.3 A taxonomy of IL approaches

## 4. Single-Task Policy Architectures

- [] 4.1 Action Chunking with Transformers
    - [] 4.1.1 Model Architecture and Training Objectives
    - [] 4.1.2 Practical Implementation with lerobot
- [] 4.2 Diffusion-Based Policy Models
    - [] 4.2.1 Generative Modeling for Action Sequences
    - [] 4.2.2 Practical Implementation with lerobot

## 5. Multi-task Policies: Vision-Language-Action (VLA) Models in Robotics

- [] 5.1 Overview of Major Architectures: RT-1, RT-2, OpenVLA, Pi0, SmolVLA, SmolVLA++
- [] 5.3 Practical Implementation: Using VLAs with lerobot

## 6. Emerging Directions in Robot Learning

- [] 6.1 World Models for Robotics
    - [] 6.1.1 V-JEPA and V-JEPA2
    - [] 6.1.2 GENIE
- [] 6.2 VLAs Post-Training
    - [] 6.2.1 EXPO
    - [] 6.2.2 From Imitation to Refinement

## 7. Conclusions
