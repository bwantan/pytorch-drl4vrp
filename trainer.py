"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Actor, Encoder, Critic
import utilities
from utilities import Logger
import time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)

def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    save_dir = args.save_dir

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)


    best_params = None
    best_reward = np.inf
    start_time = time.time()
    for epoch in range(20):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start
        printer.print_out("Training started...")
        for batch_idx, batch in enumerate(train_loader):
            static, dynamic, depot_loc = batch      # return batch of customers (static), load and demand (dynamic), depot locations

            static = static.to(device)
            dynamic = dynamic.to(device)
            depot_loc = depot_loc.to(device) if len(depot_loc) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, depot_loc)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])
                printer.print_out('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validate(valid_loader, actor, reward_fn, render_fn,
                              valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        printer.print_out('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
              np.mean(times)))

    printer.print_out('Total time is {}'.format( \
        time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

# setup training for TSP problem
def train_tsp(args):

    # Goals from paper:
    # TSP20, 3.97
    # TSP50, 6.08
    # TSP100, 8.44

    from tasks import tsp
    from tasks.tsp import TSPDataset

    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 1 # dummy for compatibility

    train_data = TSPDataset(args.num_nodes, args.train_size, args.seed)
    valid_data = TSPDataset(args.num_nodes, args.valid_size, args.seed + 1)

    update_fn = None

    actor = Actor(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    update_fn,
                    tsp.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = Critic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tsp.reward
    kwargs['render_fn'] = tsp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = TSPDataset(args.num_nodes, args.train_size, args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, tsp.reward, tsp.render, test_dir, num_plot=5)

    print('Average tour length: ', out)

# setup training for VRP problem
def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    from tasks import vrp
    from tasks.vrp import VehicleRoutingDataset, Environment

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20,
                 20: 30,
                 50: 40,
                 100: 50}
    MAX_DEMAND = 9      # max demand from the customer
    STATIC_SIZE = 2     # (x, y) coordinates
    DYNAMIC_SIZE = 2    # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]    # dynamic set max load from input parameters

    # Train/Validation data format
    # Training data: 100000 of customer (x,y) coordinates and associated (load, demand) values
    train_data = VehicleRoutingDataset(args.train_size, # default 100000 problems
                                       args.num_nodes,  # default 10 customer
                                       max_load,        # set fom LOAD_DICT
                                       MAX_DEMAND,      # 9
                                       args.seed)       # 123456

    # Validation data: 1000 of customer (x,y) coordinates and associated (load, demand) values
    valid_data = VehicleRoutingDataset(args.valid_size, #1000
                                       args.num_nodes,  #10 customer
                                       max_load,        #20
                                       MAX_DEMAND,      #9
                                       args.seed + 1)

    # Build critic model
    critic = Critic(STATIC_SIZE,  # 2 (x,y)
                    DYNAMIC_SIZE,  # 2 (load, demand)s
                    args.hidden_size).to(device)

    # Build actor model
    actor = Actor(STATIC_SIZE,                # 2 (x,y)
                    DYNAMIC_SIZE,               # 2 (load, demand)
                    args.hidden_size,           # default 128
                    train_data.update_dynamic,  #
                    train_data.update_mask,
                    args.num_layers,            #1
                    args.dropout).to(device)    #0.1
    env = Environment()

    kwargs = vars(args)   # create kwarg object and assign additional properties
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = env.reward
    kwargs['render_fn'] = env.render

    # load back the weights that have been trained
    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    # if not testing, start training
    if not args.test:
        train(actor, critic, **kwargs)

    # Testing data: 1000 of customer (x,y) coordinates and associated (load, demand) values
    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    printer.print_out('Average tour length: ', out)

if __name__ == '__main__':
    # process input arguments
    # e.g. python3 train.py --task=vrp --nodes=10 --gpu=0
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='tsp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=1000000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--stdout_print', default=True, type=utilities.str2bool, help='print control')
    args = parser.parse_args()

    # allow gpu to be specified from command line for testing
    if args.gpu >= 0:
        chosen_device = 'cuda:%d' % (args.gpu)
        globals()['device'] = torch.device(chosen_device)
    print(device)
    args.device = device

    # create log folder and log file
    try:
        now = '%s' % datetime.datetime.now().time()
        now = now.replace(':', '_')
        args.save_dir = os.path.join(args.task, '%d' % args.num_nodes, now)
        os.makedirs(args.save_dir)
    except:
        pass
    out_file = open(os.path.join(args.save_dir, 'results.txt'), 'w+')
    printer = Logger(out_file, args.stdout_print)


    # print the run args
    for key, value in sorted((vars(args)).items()):
        printer.print_out("{}: {}".format(key, value))

    if args.task == 'tsp':
        train_tsp(args)
    elif args.task == 'vrp':
        train_vrp(args)
    else:
        raise ValueError('Task <%s> not understood'%args.task)