import tqdm
import numpy as np
import torch.optim as optim
import itertools
from environments import CartPoleRegulatorEnv
from utils import NFQAgent
from utils import CompositionalNFQNetwork, MGNFQNetwork
from utils import make_reproducible

'''
Runs Joint FQI, with the union of the background and foreground datasets
'''
def run_fqi(force_left):
    seed =1234
    performance_fg = []
    performance_bg = []
    for inst in range(10):
        train_env_bg = CartPoleRegulatorEnv(group=0,mode="train",force_left=0)
        train_env_fg = CartPoleRegulatorEnv(group=1,mode="train",force_left=force_left)
        eval_env_bg = CartPoleRegulatorEnv(group=0,  mode='eval', force_left=0)
        eval_env_fg = CartPoleRegulatorEnv(group=1,  mode='eval', force_left=force_left)
        make_reproducible(seed, use_numpy=True, use_torch=True)
        train_env_bg.seed(seed)
        eval_env_bg.seed(seed)
        train_env_fg.seed(seed)
        eval_env_fg.seed(seed)
        nfq_net = CompositionalNFQNetwork(
            state_dim=train_env_bg.state_dim, is_compositional=False, big=True)
        optimizer = optim.Adam(itertools.chain(nfq_net.layers_fqi.parameters(),nfq_net.layers_last_fg.parameters()),
            lr=5e-2)

        nfq_agent = NFQAgent(nfq_net, optimizer)
        init_experience = 200
        bg_rollouts = []
        fg_rollouts = []
        for _ in range(init_experience):
            rollout_bg, _, _ = train_env_bg.generate_rollout(None, render=False, group=0)
            rollout_fg,  _, _ = train_env_fg.generate_rollout(None, render=False, group=1)
            bg_rollouts.extend(rollout_bg)
            fg_rollouts.extend(rollout_fg)
        all_rollouts = bg_rollouts + fg_rollouts
        bg_success_queue = [0] * 3
        fg_success_queue = [0] * 3
        for kk, ep in enumerate(tqdm.tqdm(range(1000))): # number of epochs
            state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
                    all_rollouts
                )
            loss = nfq_agent.train((state_action_b, target_q_values, groups))
            _,eval_success_fg,_ = nfq_agent.evaluate(eval_env_fg, render=False)
            _,eval_success_bg,_ = nfq_agent.evaluate(eval_env_bg, render=False)

            bg_success_queue = bg_success_queue[1:]
            bg_success_queue.append(1 if eval_success_bg else 0)

            fg_success_queue = fg_success_queue[1:]
            fg_success_queue.append(1 if eval_success_fg else 0)

            if sum(bg_success_queue) == 3 and sum(fg_success_queue) == 3:
                print("Done training")
                break

        evaluations = 5
        for kk, it in enumerate(tqdm.tqdm(range(evaluations))):
            eval_episode_length_bg,_,_ = nfq_agent.evaluate(eval_env_bg, False)
            performance_bg.append(eval_episode_length_bg)
            eval_episode_length_fg,_,_ = nfq_agent.evaluate(eval_env_fg, False)
            performance_fg.append(eval_episode_length_fg)
        print("BG stayed up for steps: ", np.mean(performance_bg))
        print("FG stayed up for steps: ", np.mean(performance_fg))
    return performance_bg, performance_fg

# Separate FQI policies
def separate_fqi(force_left, group):
    performance = []
    seed = 1234
    for inst in range(10):
        if group == "bg":
            train_env = CartPoleRegulatorEnv(group=0,mode="train",force_left=0)
            eval_env = CartPoleRegulatorEnv(group=0, mode='eval', force_left=0)
            g = 0
        else:
            train_env = CartPoleRegulatorEnv(group=1,mode="train",force_left=force_left)
            eval_env = CartPoleRegulatorEnv(group=1, mode='eval', force_left=force_left)
            g = 1
        make_reproducible(seed, use_numpy=True, use_torch=True)
        train_env.seed(seed)
        eval_env.seed(seed)
        nfq_net = CompositionalNFQNetwork(
            state_dim=train_env.state_dim, is_compositional=False, big=True)

        optimizer = optim.Adam(itertools.chain(nfq_net.layers_fqi.parameters(),nfq_net.layers_last_fg.parameters()),
            lr=5e-2)
        nfq_agent = NFQAgent(nfq_net, optimizer)
        init_experience = 200
        train_rollouts = []
        for _ in range(init_experience):
            rollout, _, _ = train_env.generate_rollout(None, render=False, group=g)
            train_rollouts.extend(rollout)
        success_queue = [0] * 3
        for kk, ep in enumerate(tqdm.tqdm(range(1000))): # number of epochs
            state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(train_rollouts)
            loss = nfq_agent.train((state_action_b, target_q_values, groups))
            _,eval_success,_ = nfq_agent.evaluate(eval_env, render=False)
            success_queue = success_queue[1:]
            success_queue.append(1 if eval_success else 0)
            if sum(success_queue) == 3:
                print("Done training")
                break

        evaluations = 5
        for kk, it in enumerate(tqdm.tqdm(range(evaluations))):
            eval_episode_length,_,_ = nfq_agent.evaluate(eval_env, False)
            performance.append(eval_episode_length)
        print("Stayed up for steps: ", np.mean(performance))
    return performance

# CFQI
def run_cfqi(force_left):
    seed =1234
    performance_fg = []
    performance_bg = []
    for inst in range(10):
        train_env_bg = CartPoleRegulatorEnv(group=0,mode="train",force_left=0)
        train_env_fg = CartPoleRegulatorEnv(group=1,mode="train",force_left=force_left)
        eval_env_bg = CartPoleRegulatorEnv(group=0, mode='eval', force_left=0)
        eval_env_fg = CartPoleRegulatorEnv(group=1, mode='eval', force_left=force_left)
        make_reproducible(seed, use_numpy=True, use_torch=True)
        train_env_bg.seed(seed)
        eval_env_bg.seed(seed)
        train_env_fg.seed(seed)
        eval_env_fg.seed(seed)
        nfq_net = CompositionalNFQNetwork(
            state_dim=train_env_bg.state_dim, is_compositional=True, big=True)
        optimizer = optim.Adam(itertools.chain(nfq_net.layers_shared.parameters(),nfq_net.layers_last_shared.parameters()),
            lr=1e-2)

        nfq_agent = NFQAgent(nfq_net, optimizer)
        init_experience = 200
        bg_rollouts = []
        fg_rollouts = []
        for _ in range(init_experience):
            rollout_bg, _, _ = train_env_bg.generate_rollout(None, render=False, group=0)
            rollout_fg,  _, _ = train_env_fg.generate_rollout(None, render=False, group=1)
            bg_rollouts.extend(rollout_bg)
            fg_rollouts.extend(rollout_fg)
        all_rollouts = bg_rollouts + fg_rollouts
        bg_success_queue = [0] * 3
        fg_success_queue = [0] * 3
        for kk, ep in enumerate(tqdm.tqdm(range(1000))): # number of epochs
            if nfq_net.freeze_shared:
                state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
                    fg_rollouts
                )
            else:
                state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
                    bg_rollouts
            )
            loss = nfq_agent.train((state_action_b, target_q_values, groups))
            _,eval_success_fg,_ = nfq_agent.evaluate(eval_env_fg, render=False)
            _,eval_success_bg,_ = nfq_agent.evaluate(eval_env_bg, render=False)

            bg_success_queue = bg_success_queue[1:]
            bg_success_queue.append(1 if eval_success_bg else 0)

            fg_success_queue = fg_success_queue[1:]
            fg_success_queue.append(1 if eval_success_fg else 0)
            if sum(bg_success_queue) == 3 and nfq_net.freeze_shared == False:
                nfq_net.freeze_shared = True
                # Freeze shared layers
                for param in nfq_net.layers_shared.parameters():
                    param.requires_grad = False
                for param in nfq_net.layers_last_shared.parameters():
                    param.requires_grad = False
                # Unfreeze foreground1 layers
                for param in nfq_net.layers_fg.parameters():
                    param.requires_grad = True
                for param in nfq_net.layers_last_fg.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(itertools.chain(nfq_net.layers_fg.parameters(),nfq_net.layers_last_fg.parameters(),),
                    lr=5e-2,
                )
                nfq_agent._optimizer = optimizer
            if sum(fg_success_queue) == 3:
                print("Done training")
                break

        evaluations = 5
        for kk, it in enumerate(tqdm.tqdm(range(evaluations))):
            eval_episode_length_bg,_,_ = nfq_agent.evaluate(eval_env_bg, False)
            performance_bg.append(eval_episode_length_bg)
            eval_episode_length_fg,_,_ = nfq_agent.evaluate(eval_env_fg, False)
            performance_fg.append(eval_episode_length_fg)
        print("BG stayed up for steps: ", np.mean(performance_bg))
        print("FG stayed up for steps: ", np.mean(performance_fg))
    return performance_bg, performance_fg

# Multi-group CFQI
def run_mgnfqi(verbose=False):
    is_contrastive=True
    train_env_bg = CartPoleRegulatorEnv(group=0,  mode="train", force_left=0)
    train_env_fg1 = CartPoleRegulatorEnv(group=1,mode="train", force_left=1)
    train_env_fg2 = CartPoleRegulatorEnv(group=2, mode="train", force_left=5)
    train_env_fg3 = CartPoleRegulatorEnv(group=3, mode="train", force_left=8)
    eval_env_bg = CartPoleRegulatorEnv(group=0, mode='eval', force_left=0)
    eval_env_fg1 = CartPoleRegulatorEnv(group=1, mode='eval', force_left=1)
    eval_env_fg2 = CartPoleRegulatorEnv(group=2, mode='eval', force_left=5)
    eval_env_fg3 = CartPoleRegulatorEnv(group=3, mode='eval', force_left=8)
    nfq_net = MGNFQNetwork(
        state_dim=train_env_bg.state_dim, is_compositional=is_contrastive, big=True)
    optimizer = optim.Adam(itertools.chain(nfq_net.layers_shared.parameters(), nfq_net.layers_last_shared.parameters()),
                           lr=5e-2)

    nfq_agent = NFQAgent(nfq_net, optimizer)
    init_experience = 200
    bg_rollouts = []
    fg1_rollouts = []
    fg2_rollouts = []
    fg3_rollouts = []
    for _ in range(init_experience):
        rollout_bg, _, _ = train_env_bg.generate_rollout(None, render=False, group=0)
        rollout_fg1, _, _ = train_env_fg1.generate_rollout(None, render=False, group=1)
        rollout_fg2, _, _ = train_env_fg2.generate_rollout(None, render=False, group=2)
        rollout_fg3, _, _ = train_env_fg3.generate_rollout(None, render=False, group=3)
        bg_rollouts.extend(rollout_bg)
        fg1_rollouts.extend(rollout_fg1)
        fg2_rollouts.extend(rollout_fg2)
        fg3_rollouts.extend(rollout_fg3)

    all_rollouts = bg_rollouts + fg1_rollouts + fg2_rollouts + fg3_rollouts

    bg_success_queue = [0] * 3
    fg1_success_queue = [0] * 3
    fg2_success_queue = [0] * 3
    fg3_success_queue = [0] * 3
    for kk, ep in enumerate(tqdm.tqdm(range(3000))):  # number of epochs
        state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
            all_rollouts
        )
        loss = nfq_agent.train((state_action_b, target_q_values, groups))
        _, eval_success_fg1, _ = nfq_agent.evaluate(eval_env_fg1, render=False)
        _, eval_success_fg2, _ = nfq_agent.evaluate(eval_env_fg2, render=False)
        _, eval_success_fg3, _ = nfq_agent.evaluate(eval_env_fg3, render=False)
        _, eval_success_bg, _ = nfq_agent.evaluate(eval_env_bg, render=False)

        bg_success_queue = bg_success_queue[1:]
        bg_success_queue.append(1 if eval_success_bg else 0)

        fg1_success_queue = fg1_success_queue[1:]
        fg1_success_queue.append(1 if eval_success_fg1 else 0)

        fg2_success_queue = fg2_success_queue[1:]
        fg2_success_queue.append(1 if eval_success_fg2 else 0)

        fg3_success_queue = fg3_success_queue[1:]
        fg3_success_queue.append(1 if eval_success_fg3 else 0)
        if is_contrastive:
            if sum(bg_success_queue) == 3 and nfq_net.freeze_shared == False:
                if verbose: print("Freezing shared")
                nfq_net.freeze_shared = True
                # Freeze shared layers
                for param in nfq_net.layers_shared.parameters():
                    param.requires_grad = False
                for param in nfq_net.layers_last_shared.parameters():
                    param.requires_grad = False
                # Unfreeze foreground1 layers
                for param in nfq_net.layers_fg1.parameters():
                    param.requires_grad = True
                for param in nfq_net.layers_last_fg1.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(
                    itertools.chain(
                        nfq_net.layers_fg1.parameters(),
                        nfq_net.layers_last_fg1.parameters(),
                    ),
                    lr=1e-2,
                )
                nfq_agent._optimizer = optimizer
            if sum(fg1_success_queue) == 3 and nfq_net.freeze_shared and nfq_net.freeze_fg1 == False:
                if verbose: print("Freezing Fg1")
                nfq_net.freeze_fg1 = True
                # Freeze Fg1
                for param in nfq_net.layers_fg1.parameters():
                    param.requires_grad = False
                for param in nfq_net.layers_last_fg1.parameters():
                    param.requires_grad = False
                # Unfreeze Fg2
                for param in nfq_net.layers_fg2.parameters():
                    param.requires_grad = True
                for param in nfq_net.layers_last_fg2.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(
                    itertools.chain(
                        nfq_net.layers_fg2.parameters(),
                        nfq_net.layers_last_fg2.parameters(),
                    ),
                    lr=1e-2,
                )
            if sum(fg2_success_queue) == 3 and nfq_net.freeze_fg1 and nfq_net.freeze_fg2 == False:
                if verbose: print("Freezing fg2")
                nfq_net.freeze_fg2 = True
                # Freeze Fg2
                for param in nfq_net.layers_fg2.parameters():
                    param.requires_grad = False
                for param in nfq_net.layers_last_fg2.parameters():
                    param.requires_grad = False

                # Unfreeze Fg3
                for param in nfq_net.layers_fg3.parameters():
                    param.requires_grad = True
                for param in nfq_net.layers_last_fg3.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(
                    itertools.chain(
                        nfq_net.layers_fg3.parameters(),
                        nfq_net.layers_last_fg3.parameters(),
                    ),
                    lr=1e-2,
                )
            if (nfq_net.freeze_fg2 and nfq_net.freeze_fg1 and nfq_net.freeze_shared) and sum(fg3_success_queue) == 3:
                if verbose: print("Done training")
                break
        else:
            if (sum(bg_success_queue) == 3 and sum(fg1_success_queue) == 3 and sum(fg2_success_queue) == 3 and sum(
                    fg3_success_queue) == 3):
                break

    performance_fg1 = []
    performance_fg2 = []
    performance_fg3 = []
    performance_bg = []
    evaluations = 20
    for kk, it in enumerate(tqdm.tqdm(range(evaluations))):
        eval_episode_length_bg, _, _ = nfq_agent.evaluate(eval_env_bg, False)
        performance_bg.append(eval_episode_length_bg)
        eval_episode_length_fg, _, _ = nfq_agent.evaluate(eval_env_fg1, False)
        performance_fg1.append(eval_episode_length_fg)
        eval_episode_length_fg, _, _ = nfq_agent.evaluate(eval_env_fg2, False)
        performance_fg2.append(eval_episode_length_fg)
        eval_episode_length_fg, _, _ = nfq_agent.evaluate(eval_env_fg3, False)
        performance_fg3.append(eval_episode_length_fg)
    if verbose: print("BG stayed up for steps: ", performance_bg)
    if verbose: print("FG1 stayed up for steps: ", performance_fg1)
    if verbose: print("FG2 stayed up for steps: ", performance_fg2)
    if verbose: print("FG3 stayed up for steps: ", performance_fg3)
    return performance_bg, performance_fg1, performance_fg2, performance_fg3