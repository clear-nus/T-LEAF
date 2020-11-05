import sys
import os
sys.path.append(os.getcwd())


import time
from collections import deque


import numpy as np
import torch

import pickle as pk
import random

from src.Imitation_Cooking.algorithm.a2c_ppo_acktr import algo, utils
from src.Imitation_Cooking.algorithm.a2c_ppo_acktr.algo import gail
from src.Imitation_Cooking.algorithm.a2c_ppo_acktr.arguments import get_args
from src.Imitation_Cooking.algorithm.a2c_ppo_acktr.envs import make_vec_envs
from src.Imitation_Cooking.algorithm.a2c_ppo_acktr.model import Policy
from src.Imitation_Cooking.algorithm.a2c_ppo_acktr.storage import RolloutStorage
from src.Imitation_Cooking.algorithm.a2c_ppo_acktr.evaluation import evaluate
from src.Imitation_Cooking.algorithm.a2c_ppo_acktr.algo.ppo import PPO
from src.Imitation_Cooking.algorithm.embedder_loss_util_imitation import get_embedder_loss
from src.Synthetic.models.models import Edge_Embedder, Node_only_random_agg_min_clip
from src.Synthetic.models.Util import read_pk_file





def pad_hist(my_list, num):
    return my_list + list(np.zeros((num-len(my_list),len(my_list[0]))))

if __name__ == "__main__":
    os.environ["LRU_CACHE_CAPACITY"] = "1"
    print ('LRU_CACHE_CAPACITY', os.environ["LRU_CACHE_CAPACITY"])




    args = get_args()
    torch.set_num_threads(1)
    device = torch.device("cuda:"+str(args.device_id) if args.cuda else "cpu")

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    expert_traj_save_threshold = 5
    set_num_updates = None
    np.random.seed(args.seed)

    num_expert_trajs = 100
    expert_trajs = {
        'states': [],
        'actions': [],
        'rewards': [],
        'lengths': []
    }

    use_ob_rms = False
    save_model = True
    use_clip_action = False

    pixel_ob = True

    if pixel_ob:
        args.lr = 1e-4
    else:
        args.lr = 3e-5

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    ####### embedder loss utils #################################
    action_feature = read_pk_file(args.rule_dir+"/actions_features.pk")
    ingreds_feature = read_pk_file(args.rule_dir+"/ingreds_features.pk")
    affordable_rules = read_pk_file(args.rule_dir+"/affordable.pk")
    order_rules = read_pk_file(args.rule_dir+"/ordering.pk")
    actions_name = read_pk_file(args.rule_dir+"/actions_name.pk")
    ingreds_name = read_pk_file(args.rule_dir+"/ingreds_name.pk")
    word_embedding = {}
    for key in action_feature.keys():
        word_embedding[actions_name[key]] = torch.tensor(action_feature[key], dtype=torch.float32)
    for key in ingreds_feature.keys():
        word_embedding[ingreds_name[key]] = torch.tensor(ingreds_feature[key], dtype=torch.float32)


    WORD_DIMESION = 100
    NUM_FORMULAES = 10
    INGREDS_SAMPLE_SIZE = 2
    ORDER_RULE_SAMPLE_SIZE = 2
    FEASIBLE_RULE_SAMPLE_SIZE = 5
    node_feature = np.load(args.embedder_dataset_root + "/node.npy", allow_pickle=True)
    op_feature = np.load(args.embedder_dataset_root + "/op.npy", allow_pickle=True)
    feature = {}
    one_feature = torch.ones([200], dtype=torch.float32)
    zero_feature = torch.zeros([200], dtype=torch.float32)
    feature["1"] = one_feature
    feature["0"] = zero_feature
    feature["AND"] = op_feature[0]
    feature["OR"] = op_feature[1]
    feature[""] = zero_feature
    all = {}
    all["actions"] = actions_name
    all["ingredients"] = ingreds_name
    info_embedder = (all, feature, node_feature, word_embedding, order_rules, affordable_rules, actions_name, ingreds_name, WORD_DIMESION, INGREDS_SAMPLE_SIZE, ORDER_RULE_SAMPLE_SIZE, FEASIBLE_RULE_SAMPLE_SIZE, args.embedder_dataset_root)

    edge_embedder = torch.load(args.edge_embedder_dir, map_location=device)
    meta_embedder = Node_only_random_agg_min_clip(device).to(device)
    meta_embedder.load_state_dict(torch.load(args.meta_embedder_dir, map_location=device))
    meta_embedder = meta_embedder.float()
    ###############################################



    render_threshold = 10000000
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, ob = use_ob_rms, no_reward = args.no_reward, checker = args.checker_loss)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        # file_name = os.path.join(
        #     args.gail_experts_dir, "trajs_{}.pt".format(
        #         args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            args.gail_experts_dir, num_trajectories=50, subsample_frequency=1)
        print (len(expert_dataset))
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)


    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)




    episode_r = 0
    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if set_num_updates is not None:
        num_updates = set_num_updates
    reward_save = {'mean':[],'median':[], 'min': [], 'max': [], 'dist_entropy': [], 'value_loss':[], 'action_loss': []}
    for j in range(num_updates):
        print('update no.', j)
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        if args.embedder_loss:
            logic_loss = {}
            logic_seq_temp = [[],[]]
        if args.checker_loss:
            checker_reward = {}
            checker_indicator = True
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, dist = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                action = action.squeeze(1)

            # Obser reward and next obs
            if use_clip_action:
                action = clip_action(action)
            obs, reward, done, infos = envs.step(action)
            if j > render_threshold:
                envs.render()
            episode_r += reward




            # for info in infos:
            #     episode_rewards.append(info['acc_r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

            if args.embedder_loss:
                logic_seq_temp[0].append(int(action[0]))
                logic_seq_temp[1].append(int(obs[0][action[1]*2]))
                if step % args.embedder_freq == 0 or done[0]:
                    with torch.no_grad():
                        logic_loss[step] = get_embedder_loss(device, logic_seq_temp[0], logic_seq_temp[1], meta_embedder, edge_embedder, info_embedder)
                    logic_seq_temp = [[],[]]
                    # print ('logic_loss[step]', logic_loss[step])
            if args.checker_loss:
                if infos[0]['checker_reward'] < 0:
                    checker_indicator = False
                if step % args.checker_freq == 0 or done[0]:
                    checker_reward[step] = 0 if checker_indicator else -1
                    checker_indicator = True


            if done[0]:
                for info in infos:
                    episode_rewards.append(info['acc_r'])
                envs.reset()
                if args.save_expert:
                    if infos[0]['acc_r'] > expert_traj_save_threshold and len(expert_trajs['lengths']) < num_expert_trajs and infos[0]['history']['lengths'] > 1:
                        print ('new expert traj saved', len(expert_trajs['lengths']), infos[0]['history']['lengths'], infos[0]['acc_r'])
                        expert_trajs['lengths'].append(infos[0]['history']['lengths'])
                        max_length = 100
                        expert_trajs['states'].append(pad_hist(infos[0]['history']['states'], max_length))
                        expert_trajs['actions'].append(pad_hist(infos[0]['history']['actions'], max_length))
                        expert_trajs['rewards'].append(pad_hist(infos[0]['history']['rewards'],max_length))

                    # break
                    if len(expert_trajs['lengths']) == num_expert_trajs:
                        expert_trajs['lengths'] = torch.tensor(expert_trajs['lengths'])
                        expert_trajs['states'] = torch.tensor(expert_trajs['states'])
                        expert_trajs['actions'] = torch.tensor(expert_trajs['actions'])
                        expert_trajs['rewards'] = torch.tensor(expert_trajs['rewards'])
                        torch.save(expert_trajs,args.gail_experts_dir)
                        print ('expert dataset saved')
            elif step == args.num_steps-1:
                # for info in infos:
                #     episode_rewards.append(info['acc_r'])
                envs.reset()


        # print ('logic_loss_dic', logic_loss)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                if args.embedder_only or args.no_reward or args.checker_only:
                    rollouts.rewards[step] = 0
                else:
                    rollouts.rewards[step] = discr.predict_reward(
                        rollouts.obs[step], rollouts.actions[step], args.gamma,
                        rollouts.masks[step])

            if args.embedder_loss:
                for step in logic_loss:
                    # print ('print scale', rollouts.rewards[step], logic_loss[step])
                    rollouts.rewards[step] -= args.tradeoff_logic*logic_loss[step]
            if args.checker_loss:
                for step in checker_reward:
                    rollouts.rewards[step] += checker_reward[step]

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "" and save_model:
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_" + '_PIX'+str(pixel_ob)+".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            reward_save['mean'].append(np.mean(episode_rewards))
            reward_save['median'].append(np.median(episode_rewards))
            reward_save['min'].append(np.min(episode_rewards))
            reward_save['max'].append(np.max(episode_rewards))
            reward_save['dist_entropy'].append(dist_entropy)
            reward_save['value_loss'].append(value_loss)
            reward_save['action_loss'].append(action_loss)
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
            # print('debug var', dist.mean, dist.scale)
        else:
            reward_save['mean'].append(0)
            reward_save['median'].append(0)
            reward_save['min'].append(0)
            reward_save['max'].append(0)
            reward_save['dist_entropy'].append(0)
            reward_save['value_loss'].append(0)
            reward_save['action_loss'].append(0)
        gail_name = '_gail' if args.gail else ''
        logic_loss_name = '_logic' if args.embedder_loss else ''
        checker_name = '_checker' if args.checker_loss else ''
        logic_only_name = '_ol' if args.embedder_only else ''
        checker_only_name = '_oc' if args.checker_only else ''
        no_reward_name = '_noreward' if args.no_reward else ''
        pk.dump(reward_save,open(args.log_dir+'/reward_save'+gail_name+logic_loss_name+checker_name+logic_only_name+checker_only_name+no_reward_name+'_'+str(args.seed),'wb'))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

