import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
# from env_point_mass_maze import PointMazeEnv, ob_filt
# from empw_cal_sample import sample_dissonance
from collections.abc import Mapping

double = True
use_cuda = True
if double:
    torch.set_default_tensor_type(torch.DoubleTensor)
    dtype = torch.double
device = torch.device("cuda:1" if use_cuda else "cpu")

def min_kl_div(mu1, sigma1, mu2, sigma2):
    if not torch.is_tensor:
        mu1 = torch.tensor(mu1).to(device)
    if not torch.is_tensor(mu2):
        mu2 = torch.tensor(mu1).to(device)
    p = torch.distributions.MultivariateNormal(mu1, torch.diag(sigma1[0]))

    if mu2.dim() == 1:
        q = torch.distributions.MultivariateNormal(mu2.unsqueeze(0), torch.diag(sigma2))
    elif mu2.dim() == 2:
        q = torch.distributions.MultivariateNormal(mu2, torch.diag(sigma2[0]))
    kl1 = torch.distributions.kl_divergence(p, q)
    kl2 = torch.distributions.kl_divergence(q, p)
    return torch.min(kl1, kl2)


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, direction, empw_model, qvar_model, transition_model, use_raw, emp_K = 5, action_dim=2, pixel_ob = False):
    use_alter = True
    n_episod = 0

    running_acc1 = [0., 0.]
    running_acc2 = [0., 0.]

    running_lowb_acc = [0., 0.]

    ######## create raw env ######
    if use_raw:
        env_raw = PointMazeEnv(direction=direction)
        if use_alter:
            if direction == 0:
                direction_alter = 1
            else:
                direction_alter = 0
            env_alter_raw = PointMazeEnv(direction=direction_alter)
    ########################################

    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, direction=direction, pixel_ob = pixel_ob)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    eval_envs.reset()
    obs, reward, done, infos = eval_envs.step(torch.zeros(action_dim))
    if isinstance(infos, Mapping):
        pos = infos['pos']
    else:
        pos = infos[0]['pos']
    if use_raw:
        obs_raw = env_raw.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    if use_alter:
        if direction == 0:
            direction_alter = 1
        else:
            direction_alter = 0
        eval_envs_alter = make_vec_envs(env_name, seed + num_processes, num_processes,
                                        None, eval_log_dir, device, True, direction=direction_alter, pixel_ob = pixel_ob)
        vec_norm_alter = utils.get_vec_normalize(eval_envs_alter)
        if vec_norm_alter is not None:
            vec_norm_alter.eval()
            vec_norm_alter.ob_rms = ob_rms

        eval_episode_rewards_alter = []

        eval_envs_alter.reset()
        obs_alter, reward_alter, done_alter, pos_alter = eval_envs_alter.step(torch.zeros(action_dim))
        if use_raw:
            obs_alter_raw = env_alter_raw.reset()
        eval_recurrent_hidden_states_alter = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks_alter = torch.zeros(num_processes, 1, device=device)

    n_step = 0
    reward_all = 0
    reward_all_alter = 0
    if use_raw:
        reward_all_raw = 0
        reward_all_raw_alter = 0
    while n_episod < 10:
        n_step += 1
        with torch.no_grad():
            if double:
                obs = obs.double()
                # pos = pos.double()
                obs_alter = obs_alter.double()
                # pos_alter = pos_alter.double()
                if use_raw:
                    obs_raw = torch.tensor(obs_raw)
                    obs_alter_raw = torch.tensor(obs_alter_raw)
            _, action, _, eval_recurrent_hidden_states, dist = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)
            if use_raw:
                action_empw, dist_emp = empw_model.policy.get_action(obs_raw)
            else:
                if emp_K > 1:
                    dummy_action = torch.zeros(action_dim).unsqueeze(0)
                    o_input = torch.cat((obs,dummy_action), 1)
                else:
                    o_input = obs
                if empw_model.cnn:
                    o_input = empw_model.cnn(o_input)[1]
                action_empw, dist_emp = empw_model.policy.get_action(o_input)

            if use_alter:
                _, action_alter, _, eval_recurrent_hidden_states_alter, dist_alter = actor_critic.act(
                    obs_alter,
                    eval_recurrent_hidden_states_alter,
                    eval_masks_alter,
                    deterministic=True)
                if use_raw:
                    action_empw_alter, dist_emp_alter = empw_model.policy.get_action(obs_alter_raw)
                else:
                    if emp_K > 1:
                        dummy_action = torch.zeros(action_dim).unsqueeze(0)
                        o_input = torch.cat((obs_alter, dummy_action),1)
                    else:
                        o_input = obs_alter
                    if empw_model.cnn:
                        o_input = empw_model.cnn(o_input)[1]
                    action_empw_alter, dist_emp_alter = empw_model.policy.get_action(o_input)
        # dist.loc, dist.scale
        # dist_entropy = dist.entropy().mean()

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)
        if isinstance(infos, Mapping):
            pos = infos['pos']
        else:
            pos = infos[0]['pos']
        reward_all += reward
        if use_raw:
            obs_raw, reward_raw, done_raw, _ = env_raw.step(action)
            reward_all_raw += reward_raw

        if double:
            obs = obs.double()
            # pos = pos.double()
            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.double,
                device=device)
        else:
            eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)


        if use_alter:
            # Obser reward and next obs
            obs_alter, reward, done_alter, pos_alter = eval_envs_alter.step(action_alter)
            reward_all_alter += reward
            if use_raw:
                obs_alter_raw, reward_raw, _, _ = env_alter_raw.step(action_alter)
                reward_all_raw_alter += reward_raw

            if double:
                obs_alter = obs_alter.double()
                # pos_alter = pos_alter.double
                eval_masks_alter = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done_alter],
                    dtype=torch.float32,
                    device=device)
            else:
                eval_masks_alter = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done_alter],
                dtype=torch.double,
                device=device)

        if done or n_step >= 200:
            # print (reward_all, reward_all_raw)
            n_episod += 1
            n_step = 0
            eval_envs.reset()
            obs, reward, done, infos = eval_envs.step(torch.zeros(action_dim))
            if isinstance(infos, Mapping):
                pos = infos['pos']
            else:
                pos = infos[0]['pos']
            if double:
                obs = obs.double()
                # pos = pos.double()
            if use_raw:
                obs_raw = env_raw.reset()
            eval_episode_rewards.append(reward_all)
            reward_all = 0
            if use_alter:
                eval_envs_alter.reset()
                obs_alter, reward_alter, done_alter, pos_alter = eval_envs_alter.step(torch.zeros(action_dim))
                if double:
                    obs_alter = obs_alter.double()
                    # pos_alter = pos_alter.double()
                if use_raw:
                    obs_alter_raw = env_alter_raw.reset()
                eval_episode_rewards_alter.append(reward_all_alter)
                reward_all_alter = 0


        ############### test the calculated empw - KL lowerbound ##########
        print ('\n ***** Test Lower Bound: Empw - KL *****')
        min_kl = min_kl_div(dist.loc, dist.scale, dist_emp['mean'],dist_emp['log_std'].exp())
        min_kl_alter = min_kl_div(dist_alter.loc, dist_alter.scale, dist_emp_alter['mean'], dist_emp_alter['log_std'].exp())

        if use_raw:
            empw = empw_model.eval(torch.tensor(obs_raw))-min_kl
            empw_alter = empw_model.eval(torch.tensor(obs_alter_raw))-min_kl_alter
        else:
            empw =  empw_model.eval(obs) - min_kl
            empw_alter = empw_model.eval(obs_alter) - min_kl_alter

        lowb = empw-min_kl
        lowb_alter = empw_alter - min_kl_alter

        if lowb <= lowb_alter:
            running_lowb_acc[0] += 1
        running_lowb_acc[1] += 1
        print ('action dist std', dist.scale, dist_alter.scale)
        print ('action dist entropy', dist.entropy().mean(), dist_alter.entropy().mean())
        print('omega dist mean', dist_emp['mean'], dist_emp_alter['mean'])
        print ('omega dist std', dist_emp['log_std'].exp(), dist_emp_alter['log_std'].exp())
        print('empw', empw, empw_alter)
        print('kl', min_kl, min_kl_alter)
        print ('lowb: empw-KL', lowb, lowb_alter, running_lowb_acc[0]/running_lowb_acc[1])
        #####################################################


        ###### test empw calculated using sampling ##########################
        print('\n ***** Sampling dissonance *****')
        n_steps = 1
        dissonance1, dissonance2 = sample_dissonance(current_pos = pos, current_state=obs, actor_critic=actor_critic, ob_rms=ob_rms,
                                       env_direction=direction, qvar_model=qvar_model, transition_model=transition_model, num_steps = n_steps, emp_K = emp_K, pixel_ob = pixel_ob)
        dissonance_alter1, dissonance_alter2 = sample_dissonance(current_pos = pos, current_state=obs_alter, actor_critic=actor_critic, ob_rms=ob_rms,
                                       env_direction=direction_alter, qvar_model=qvar_model, transition_model=transition_model, num_steps = n_steps, emp_K = emp_K, pixel_ob = pixel_ob)
        if dissonance1 <= dissonance_alter1:
            running_acc1[0] += 1.0
        running_acc1[1] += 1.0

        if dissonance2 <= dissonance_alter2:
            running_acc2[0] += 1.0
        running_acc2[1] += 1.0

        print('dissonance 1', dissonance1, dissonance_alter1, running_acc1[0]/running_acc1[1])
        print('dissonance 2', dissonance2, dissonance_alter2, running_acc2[0]/running_acc2[1])
        ######################################################

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    if use_alter:
        eval_envs_alter.close()

        print("Alternative Env | Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards_alter), np.mean(eval_episode_rewards_alter)))