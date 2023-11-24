import os
import csv
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime

from MultiAgents.MultiAgent import IMARL


class TrainAgent(object):
    def __init__(self, agent, env, test_env, dn_ffw, ep_infos, max_reward=10, rw_func="loss"):
        self.agent = agent
        self.env = env
        self.test_env = test_env
        self.dn_ffw = dn_ffw
        self.ep_infos = ep_infos
        self.max_rw = max_reward
        self.rw_func = rw_func
        self.last_number_updates = 0
        self.last_stats = self.last_scores = self.last_steps = None

    # following competition evaluation script
    def compute_episode_score(self, chronic_id, agent_step, agent_reward, ffw=None):
        min_losses_ratio = 0.7
        ep_marginal_cost = self.env.gen_cost_per_MW.max()
        if ffw is None:
            ep_do_nothing_reward = self.ep_infos[chronic_id]["donothing_reward"]
            ep_do_nothing_nodisc_reward = self.ep_infos[chronic_id]["donothing_nodisc_reward"]
            ep_dn_played = self.ep_infos[chronic_id]["dn_played"]
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])
        else:
            start_idx = 0 if ffw == 0 else ffw * 288 - 2
            end_idx = start_idx + 864
            ep_dn_played, ep_do_nothing_reward, ep_do_nothing_nodisc_reward = self.dn_ffw[(chronic_id, ffw)]
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])[start_idx:end_idx]
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])[start_idx:end_idx]

        # Add cost of non delivered loads for blackout steps
        blackout_loads = ep_loads[agent_step:]
        if len(blackout_loads) > 0:
            blackout_reward = np.sum(blackout_loads) * ep_marginal_cost
            agent_reward += blackout_reward

        # Compute ranges
        worst_reward = np.sum(ep_loads) * ep_marginal_cost
        best_reward = np.sum(ep_losses) * min_losses_ratio
        zero_reward = ep_do_nothing_reward
        zero_blackout = ep_loads[ep_dn_played:]
        zero_reward += np.sum(zero_blackout) * ep_marginal_cost
        nodisc_reward = ep_do_nothing_nodisc_reward

        # Linear interp episode reward to codalab score
        if zero_reward != nodisc_reward:
            # DoNothing agent doesnt complete the scenario
            reward_range = [best_reward, nodisc_reward, zero_reward, worst_reward]
            score_range = [100.0, 80.0, 0.0, -100.0]
        else:
            # DoNothing agent can complete the scenario
            reward_range = [best_reward, zero_reward, worst_reward]
            score_range = [100.0, 0.0, -100.0]

        ep_score = np.interp(agent_reward, reward_range, score_range)
        return ep_score

    def interaction(self, obs, prev_act, start_step):
        self.agent.save_start_transition()
        # order = None if self.agent.order is None else self.agent.order.clone()
        reward, train_reward, step = 0, 0, 0
        while True:
            # prev_act is executed at first anyway
            if prev_act:
                act = prev_act
                prev_act = None
            else:
                act = self.agent.act(obs, None, None)
                if self.agent.save:
                    # pass this act to the next step.
                    prev_act = act
                    break
            # just step if action is okay or failed to find other action
            obs, rew, done, info = self.env.step(act)
            reward += rew
            new_reward = info["rewards"][self.rw_func]
            train_reward += new_reward
            step += 1
            if start_step + step == 864:
                done = True
            if done:
                break
        train_reward = np.clip(train_reward, -2, self.max_rw)
        die = bool(done and info["exception"])
        transition = (train_reward, die)
        self.agent.save_transition(*transition)
        infos = (step + start_step, prev_act, info, 1)
        return obs, reward, done, infos

    # compute weight for chronic sampling
    def chronic_priority(self, cid, ffw, step):
        m = 864
        scale = 2.0
        diff_coef = 0.05
        d = self.dn_ffw[(cid, ffw)][0]  # how many steps was DN agent able to survive for this chron_id and ffw moment
        progress = 1 - np.sqrt(step / m)  # steps survived current agent current agent
        difficulty = 1 - np.sqrt(d / m)
        score = (progress + diff_coef * difficulty) * scale
        return score

    def train(
        self,
        seed,
        nb_frame,
        test_step,
        train_chronics,
        valid_chronics,
        output_dir,
        model_path,
        max_ffw,
        best_score=-100,
        verbose=True,
    ):
        if verbose:
            print(
                " ****************************** \n"
                " ***  START TRAINING AGENT  *** \n"
                " ****************************** \n"
            )

        # initialize training chronic sampling weights
        train_chronics_ffw = [(cid, fw) for cid in train_chronics for fw in range(max_ffw)]
        total_chronic_num = len(train_chronics_ffw)
        chronic_records = [0] * total_chronic_num
        chronic_step_records = [0] * total_chronic_num

        # for each chronic fw upto where DNagent performed worst for evaluation
        if max_ffw == 5:
            valid_chron_ffw = range(max_ffw)
            with open(os.path.join(output_dir, "score.csv"), "a", newline="") as cf:
                csv.writer(cf).writerow(["env_interactions"] + [f"score_chron17_{i}" for i in range(5)])
        else:
            valid_chron_ffw = {
                i: int(np.argmin([self.dn_ffw[(i, fw)][0] for fw in range(max_ffw)])) for i in valid_chronics
            }
            with open(os.path.join(output_dir, "score.csv"), "a", newline="") as cf:
                csv.writer(cf).writerow(
                    ["env_interactions"] + [f"score_chron{chron}_{ffw}" for chron, ffw in valid_chron_ffw.items()]
                )

        for i in range(total_chronic_num):
            cid, fw = train_chronics_ffw[i]
            chronic_records[i] = self.chronic_priority(cid, fw, 1)

        update_pbar = nb_frame / 100

        # evaluate initial start of the agent
        best_score, prune = self.eval_train(valid_chron_ffw, output_dir, model_path, best_score, verbose)

        tic = datetime.now()
        if verbose:
            pbar = tqdm(
                desc="Progress training agent for %g timesteps" % nb_frame, total=nb_frame, position=0, leave=True
            )
        else:
            pbar = None
        # training loop
        while self.agent.agent_step < nb_frame:
            # sample training chronic
            dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(chronic_records))
            record_idx = dist.sample().item()
            chronic_id, ffw = train_chronics_ffw[record_idx]
            self.env.set_id(
                train_chronics.index(chronic_id)
            )  # NOTE: this will take the previous chronic since with env.reset() you will get the next
            obs = self.env.reset()
            if ffw > 0:
                self.env.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self.env.step(self.env.action_space())
            done = False
            alive_frame = 0
            total_reward = 0
            train_reward = 0

            # update = False
            self.agent.reset(obs)
            prev_act = self.agent.act(obs, None, None)
            while not done:
                obs, reward, done, info = self.interaction(obs, prev_act, alive_frame)
                alive_frame, prev_act = info[:2]
                interacted = info[-1]
                # total_reward += reward
                # train_reward += info[0][3]

                if self.agent.check_start_update():
                    # toc = datetime.now()
                    # print("Duration of interaction: ", toc - tic)
                    # # update = True
                    # # start updating agent when there is enough memory
                    # tic = datetime.now()
                    self.agent.update()
                    # toc = datetime.now()
                    # print(f"\n Duration of agent {self.agent.update_step} update: {toc - tic} "
                    #       f"\n steps taken is now {self.agent.agent_step}")

                if (self.agent.agent_step % test_step == 0) & interacted:
                    # start evaluation agent after test_step number of updates
                    cache = self.agent.cache_stat()
                    best_score, prune = self.eval_train(valid_chron_ffw, output_dir, model_path, best_score, verbose)
                    self.agent.load_cache_stat(cache)

                    tic = datetime.now()

                if self.agent.agent_step > nb_frame:
                    break
                if prune == 1:
                    break
            if prune == 1:
                break
            # update chronic sampling weight
            chronic_records[record_idx] = self.chronic_priority(chronic_id, ffw, alive_frame)
            chronic_step_records[record_idx] = alive_frame
            if verbose and ((self.agent.agent_step - pbar.n) > update_pbar):
                pbar.n = self.agent.agent_step
                pbar.refresh()

        if verbose:
            self.agent.save_model(model_path, "last")
            pbar.close()
            print(f"\n__________________________________\n     Training is done:\n----------------------------------")
            print(f"** Best score agent: {best_score:9.4f} **")
            print(f"The agent did {self.agent.update_step} updates in total")
            if isinstance(self.agent, IMARL):
                self.agent.print_updates_per_agent()

        return best_score

    def eval_train(self, valid_chron_ffw, output_dir, model_path, best_score, verbose=True):
        # start evaluation agent after test_step number of updates
        eval_iter = self.agent.agent_step  # // test_step
        if (self.agent.update_step > self.last_number_updates) or (self.agent.update_step == 0):
            self.last_number_updates = self.agent.update_step
            result, self.last_stats, self.last_scores, self.last_steps = self.test(valid_chron_ffw, verbose)
            if (best_score - 0.5) < self.last_stats["score"]:  # update also when score is almost as good as best
                if verbose:
                    print(f"Found score higher or similar to best, save agent!")
                best_score = max(best_score, self.last_stats["score"])
                self.agent.save_model(model_path, "best")
                np.save(os.path.join(output_dir, f"best_score.npy"), best_score)

        if verbose:
            print(f"[{eval_iter:4d}] Valid: score {self.last_stats['score']} | step {self.last_stats['step']}")

        # log and save model
        with open(os.path.join(output_dir, "score.csv"), "a", newline="") as cf:
            csv.writer(cf).writerow([self.agent.agent_step] + self.last_scores)
        with open(os.path.join(output_dir, "step.csv"), "a", newline="") as cf:
            csv.writer(cf).writerow(self.last_steps)

        return best_score, self.last_stats["score"]

    def test(self, chron_ffw, verbose=True):
        result = {}
        steps, scores = [], []
        if verbose:
            print("\n")
        for i in chron_ffw:
            self.test_env.seed(59)
            obs = self.test_env.reset()
            cur_chron = int(self.test_env.chronics_handler.get_name())
            if len(chron_ffw) == 5:
                ffw = i
            else:
                ffw = chron_ffw[cur_chron]
            dn_step = self.dn_ffw[(cur_chron, ffw)][0]

            self.agent.reset(obs)

            if ffw > 0:
                self.test_env.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self.test_env.step(self.test_env.action_space())

            total_reward = 0
            alive_frame = 0
            done = False
            result[(cur_chron, ffw)] = {}

            while not done:
                act = self.agent.act(obs, 0, 0)
                obs, reward, done, info = self.test_env.step(act)
                total_reward += reward
                alive_frame += 1
                if alive_frame == 864:
                    done = True

            l2rpn_score = float(self.compute_episode_score(cur_chron, alive_frame, total_reward, ffw))
            if verbose:
                print(
                    f"[Test Ch{cur_chron:4d}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) Score: {l2rpn_score:9.4f} "
                )
            scores.append(l2rpn_score)
            steps.append(alive_frame)

            result[(cur_chron, ffw)]["real_reward"] = total_reward
            result[(cur_chron, ffw)]["reward"] = l2rpn_score
            result[(cur_chron, ffw)]["step"] = alive_frame

        val_step = val_score = val_rew = 0
        for key in result:
            val_step += result[key]["step"]
            val_score += result[key]["reward"]
            val_rew += result[key]["real_reward"]
        stats = {
            "step": val_step / len(chron_ffw),
            "score": val_score / len(chron_ffw),
            "reward": val_rew / len(chron_ffw),
            # 'alpha': self.agent.log_alpha.exp().item() # wont work for MA_SACD need other way to track alpha
        }
        if len(chron_ffw) != 5:
            # correct order of scores since the chronics picked are one later.
            scores.insert(0, scores.pop())
        return result, stats, scores, steps

    def evaluate(self, chronics, max_ffw, path, sample=False):
        result = {}
        steps, scores = [], []
        if sample:
            path = os.path.join(path, "sample/")
            os.makedirs(path, exist_ok=True)

        if max_ffw == 5:
            chronics = chronics * 5
        for idx, chron_id in enumerate(chronics):
            if max_ffw == 5:
                ffw = idx
            else:
                ffw = int(
                    np.argmin(
                        [
                            self.dn_ffw[(chron_id, fw)][0]
                            for fw in range(max_ffw)
                            if (chron_id, fw) in self.dn_ffw and self.dn_ffw[(chron_id, fw)][0] >= 10
                        ]
                    )
                )

            dn_step = self.dn_ffw[(chron_id, ffw)][0]
            self.test_env.seed(59)
            self.test_env.set_id(chron_id)
            obs = self.test_env.reset()
            self.agent.reset(obs)

            if ffw > 0:
                self.test_env.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self.test_env.step(self.test_env.action_space())

            total_reward = 0
            alive_frame = 0
            done = False
            # df_topo = pd.DataFrame(columns=['ts', 'safe', 'rho', 'topo', 'topo_goal',
            #                                 'action_val', 'low_actions', 'curr_load', 'future_load',
            #                                 'curr_gen', 'future_gen'])
            # topo_changes = [np.array([])] * len(self.agent.converter.sub_mask)

            if 0:
                topo_changes = np.array([obs.topo_vect[self.agent.action_converter.sub_mask]])
            else:
                topo_changes = np.array([obs.topo_vect])

            # goals = None
            # inputs = None
            safe = np.array([1])

            result[(chron_id, ffw)] = {}
            while not done:
                act = self.agent.act(obs, None, 0) if sample else self.agent.act(obs, 0, 0)
                # if act != self.agent.action_space({}):
                if 0:
                    bus_goal = np.where(self.agent.goal > self.agent.bus_thres, 2, 1)
                    prev_topo = obs.topo_vect[self.agent.action_converter.sub_mask]
                else:
                    bus_goal = act.set_bus
                    prev_topo = obs.topo_vect
                prev_step = alive_frame
                # if type(self.agent) == SAC:
                #     df_topo.loc[len(df_topo)] = [prev_step, self.agent.is_safe(obs), obs.rho.round(2), prev_topo,
                #                                  bus_goal, np.array(self.agent.goal.squeeze()), self.agent.low_actions,
                #                                  obs.load_p, obs.get_forecasted_inj()[2],
                #                                  obs.gen_p.round(2), obs.get_forecasted_inj()[0]]

                topo_changes = np.vstack((topo_changes, prev_topo))
                safe = np.append(safe, self.agent.is_safe(obs))
                # if not self.agent.is_safe(obs):
                #     if goals is None:
                #         goals = np.array(self.agent.goal)
                #         inputs = np.array(self.agent.get_current_state())
                #     else:
                #         goals = np.vstack((goals, self.agent.goal))
                #         inputs = np.vstack((inputs, self.agent.get_current_state()))
                obs, reward, done, info = self.test_env.step(act)
                total_reward += reward
                alive_frame += 1
                if alive_frame == 864:
                    done = True

            l2rpn_score = float(self.compute_episode_score(chron_id, alive_frame, total_reward, ffw))
            # if type(self.agent) == SAC:
            #     df_topo.to_csv(os.path.join(path, f"Ch{chron_id}_{ffw}_topoAnalytics.csv"), index=False)

            np.save(os.path.join(path, f"Ch{chron_id}_{ffw}_topo.npy"), topo_changes)
            np.save(os.path.join(path, f"Ch{chron_id}_{ffw}_safe.npy"), safe)

            # np.save(os.path.join(path, f"Ch{chron_id}_{ffw}_goalactions.npy"), goals)
            # np.save(os.path.join(path, f"Ch{chron_id}_{ffw}_inputs.npy"), inputs)

            print(f"[Test Ch{chron_id:4d}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) Score: {l2rpn_score:9.4f}")
            scores.append(l2rpn_score)
            steps.append(alive_frame)

            result[(chron_id, ffw)]["real_reward"] = total_reward
            result[(chron_id, ffw)]["reward"] = l2rpn_score
            result[(chron_id, ffw)]["step"] = alive_frame

        val_step = val_score = val_rew = 0
        for key in result:
            val_step += result[key]["step"]
            val_score += result[key]["reward"]
            val_rew += result[key]["real_reward"]

        stats = {
            "step": val_step / len(chronics),
            "score": val_score / len(chronics),
            "reward": val_rew / len(chronics),
        }
        return stats, scores, steps


class Train(TrainAgent):
    def interaction(self, obs, prev_act, start_step):
        act = prev_act
        reward, done, step, info = 0, 0, 0, None
        interacted = False
        while (not self.agent.save) & (start_step < 864) & (not done):
            # action is do nothing OR reconnect line
            obs, rew, done, info = self.env.step(act)
            start_step += 1
            # reward += rew
            act = self.agent.act(obs, None, None)

        if self.agent.save & (start_step < 864):
            interacted = True
            # agent.act() has generated a new action (goal)
            self.agent.save_start_transition()
            train_reward = 0
            discount = 1
            skip = False
            if act == self.env.action_space():
                if isinstance(self.agent, IMARL):
                    # agent for this sub decided to keep topology config as is.
                    skip = True
                else:
                    # agent does nothing while action should be taken.
                    train_reward -= 1
            while not done:
                if skip:
                    # train_reward += 0
                    info = {"exception": False}
                else:
                    obs, rew, done, info = self.env.step(act)
                    # reward += rew
                    new_reward = info["rewards"][self.rw_func]
                    train_reward += discount * new_reward
                    step += 1
                    discount *= self.agent.gamma
                if start_step + step == 864:
                    if not (self.agent.save and done):
                        # environment is still safe
                        train_reward += discount * 1
                    done = True
                if done:
                    break
                act = self.agent.act(obs, None, None)
                if self.agent.save or (train_reward > self.max_rw):
                    # a new actions has been generated OR the env has been safe for a long time
                    # pass this act to the next step.
                    prev_act = act
                    break
                elif skip:
                    skip = False
            train_reward = np.clip(train_reward, -2, self.max_rw)
            die = bool(done and info["exception"])
            self.agent.save_transition(train_reward, die, n_step=step)
            infos = (step + start_step, prev_act, info, interacted)
        else:
            done = True
            infos = (start_step, prev_act, info, interacted)
        return obs, reward, done, infos


class ParamTuningTrain(Train):
    """
    Agent for parameter tuning
    """

    def __init__(self, agent, env, test_env, dn_ffw, ep_infos, max_reward=10, trial=None, seed=0):
        super().__init__(agent, env, test_env, dn_ffw, ep_infos, max_reward)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.last_mean_reward = -100
        self.seed = seed

    def eval_train(self, valid_chron_ffw, output_dir, model_path, best_score, verbose=True):
        best_score, self.last_mean_reward = super().eval_train(
            valid_chron_ffw, output_dir, model_path, best_score, verbose
        )
        self.eval_idx += 1
        # Send report to Optuna
        if self.seed:
            old_value = self.trial._cached_frozen_trial.intermediate_values[self.eval_idx]
            new_value = (old_value * self.seed + self.last_mean_reward) / (self.seed + 1)
            self.trial.storage.set_trial_intermediate_value(self.trial._trial_id, self.eval_idx, new_value)
            self.trial._cached_frozen_trial.intermediate_values[self.eval_idx] = new_value
        else:
            self.trial.report(self.last_mean_reward, self.eval_idx)
        # Prune trial if need
        if self.trial.should_prune():
            self.is_pruned = True
        return best_score, self.is_pruned


# class TrainAgentIMARL(TrainAgent):
#     def __init__(self, agent, env, test_env, device,dn_json_path, dn_ffw, ep_infos, max_reward=10):
#         super().__init__(agent, env, test_env, device,dn_json_path, dn_ffw, ep_infos, max_reward=10)
#         if not isinstance(agent, IMARL):
#             raise TypeError("Agent should be a independent multi-agent (IMARL).")
#         self.tot_rw_agent = {sub: 0.0 for sub in self.agent.agents}
#         self.disc_agent = {sub: 0.0 for sub in self.agent.agents}
#         self.n_step = {sub: 0 for sub in self.agent.agents}
#
#     def interaction(self, obs, prev_act, start_step):
#         act = prev_act
#         reward, done, step, info = 0, 0, 0, None
#         while (not self.agent.save) & (start_step < 864) & (not done):
#             # action is do nothing OR reconnect line
#             obs, rew, done, info = self.env.step(act)
#             start_step += 1
#             # reward += rew
#             act = self.agent.act(obs, None, None)
#
#         if self.agent.save & (start_step < 864):
#             current_sub, _ = self.agent.goal
#             if self.tot_rw_agent[current_sub] > 0:
#                 self.agent.save_transition(self.tot_rw_agent[current_sub], done=0, n_step= self.n_step[current_sub])
#                 self.tot_rw_agent[current_sub] = 0
#             # agent.act() has generated a new action (goal)
#             self.agent.save_start_transition()
#             train_reward = 0
#             self.disc_agent[current_sub] = 1
#             while not done:
#                 obs, rew, done, info = self.env.step(act)
#                 # reward += rew
#                 new_reward = info['rewards']['loss']
#                 self.tot_rw_agent = {old_rw: self.disc_agent[sub] * new_reward if old_rw > 0 else old_rw for sub, old_rw in
#                                      self.tot_rw_agent.items()}
#                 train_reward += self.disc_agent[current_sub] * new_reward
#                 self.disc_agent[current_sub] *= self.agent.gamma
#                 step += 1
#                 if (start_step + step == 864):
#                     if not (self.agent.save and done):
#                         # environment is still safe
#                         train_reward += discount * 1
#                     done = True
#                 if done:
#                     break
#                 act = self.agent.act(obs, None, None)
#                 if self.agent.save or (train_reward > self.max_rw):
#                     # a new actions has been generated OR the env has been safe for a long time
#                     # pass this act to the next step.
#                     prev_act = act
#                     break
#             train_reward = np.clip(train_reward, -2, self.max_rw)
#             die = bool(done and info['exception'])
#             if done:
#                 self.agent.save_transition(train_reward, die, n_step=step)
#
#             infos = (step + start_step, prev_act, info)
#         else:
#             done = True
#             infos = (start_step, prev_act, info)
#         return obs, reward, done, infos
