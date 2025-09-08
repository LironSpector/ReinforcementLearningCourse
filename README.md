# ReinforcementLearningCourse — RL & Deep Learning Projects (Python)

This repository is a collection of hands‑on projects completed while following the **“Practical AI with Python and Reinforcement Learning”** Udemy course. It includes using **OpenAI Gym** with **DQN**, a custom **Snake** environment + agent, and several **TensorFlow/Keras** projects.


## Getting Started
```bash
pip install tensorflow keras keras-rl2 gym pillow pygame numpy matplotlib
# Some projects also use: scikit-learn, pandas
# If a folder has a dedicated requirements.txt, prefer that.
```
Some projects include pre‑trained weights available to load to evaluate without training.


## Example Projects
- **Snake Game with AI (Gym + Pygame + DQN)**  
  Custom OpenAI Gym environment for Snake (rendered with **pygame**) plus a DQN agent. Supports image observations and comes with example trained weights for evaluation/play.

- **DQN on Atari/Gym (e.g., Pong/Breakout game with image processing)**  
  Builds a **Keras‑RL** DQN with convolutional layers.

- **CartPole / Acrobot (Classic Control)**  
  DQNs for classic Gym tasks to demonstrate network design, exploration, and more.

- **TensorFlow: CIFAR‑10 Image Classification** *(TensorFlow/Keras)*  
  CNN on CIFAR‑10 with training/evaluation, showing data normalization, categorical labels, and more. (Other TF examples include MNIST and a medical‑images classifier).
