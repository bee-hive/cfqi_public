# Compositional Fitted Q-iteration
This repository contains the implementation of Compositional Fitted Q-iteration (CFQI) from the paper Compositional Q-learning for 
electrolyte repletion with imbalanced patient sub-populations, to be presented at ML4H 2023. Compositional Fitted Q-iteration is a 
variant on Fitted Q-iteration that enables better Q-function approximation when a dataset has several sub-populations with heterogeneous treatment effect

### Abstract
Reinforcement learning (RL) is an effective framework for solving sequential decision-making tasks. However, applying RL methods 
in medical care settings is challenging due to heterogeneity in treatment response among patients. Some patients can be treated with 
standard protocols whereas others, such as those with chronic diseases, need personalized treatment planning. Traditional RL methods 
often fail to account for this diversity, because they assume that transition dynamics are shared across all users. We introduce Compositional 
Fitted Q-iteration (CFQI), which uses a compositional task structure to represent diverse treatment responses in medical care settings. 
A compositional task consists of several variations of the same task, each progressing in difficulty; solving simpler variants of the task 
can enable efficient solving of harder variants. CFQI uses a compositional $Q$-value function with separate modules for each task variant, 
allowing it to take advantage of shared knowledge while learning distinct policies for each variant. We validate CFQI's performance using 
a Cartpole environment and use it to recommend electrolyte repletion for patients with and without renal disease. Our results demonstrate 
that CFQI is robust even in the presence of class imbalance, enabling effective information usage across patient groups. CFQI exhibits 
great promise for clinical applications in scenarios characterized by known compositional structures.
