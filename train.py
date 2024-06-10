import os
import csv
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
from util import (write_depths_to_csv, write_action_counts_to_csv, write_steps_ol_to_csv, write_topologies_to_csv, 
                  write_is_safe_to_csv, write_ra_action_counts_to_csv, write_substation_configs_to_csv, 
                  write_unique_topos_total_to_csv, compute_per_chronic_measures, compute_across_chronic_measures)

from MultiAgents.MultiAgent import IMARL, ReArIMARL


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

        self.begin = {sub: self.agent.action_converter.sub_to_topo_begin[sub] for sub in self.agent.action_converter.masked_sorted_sub}
        self.end = {sub: self.agent.action_converter.sub_to_topo_end[sub] for sub in self.agent.action_converter.masked_sorted_sub}
        self.substation_id_to_index = {sub: idx for idx, sub in enumerate(self.agent.action_converter.masked_sorted_sub)}
        self.eval_train_count = -1
        self.count_all = np.zeros((len(self.agent.action_converter.masked_sorted_sub), len(self.agent.action_converter.masked_sorted_sub)), int)
        self.count_last = np.zeros((len(self.agent.action_converter.masked_sorted_sub), len(self.agent.action_converter.masked_sorted_sub)), int)
        self.previous_sub = None
        # TO DO: implement the "per evaluation period" RA transition matrix.
        if type(self.agent) == ReArIMARL:
            self.count_ra = np.zeros((len(self.agent.action_converter.ra_idx),len(self.agent.action_converter.ra_idx)), int)
        self.actions = 0
        self.reconnects = 0
        self.donothings = 0
        self.last_actions = 0
        self.last_reconnects = 0
        self.last_donothings = 0

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
            # FIX WHEN THE TIME COMES (DSAC, TRANSITION MATRIX, CHECK THE OTHER INTERACTION METHOD)
            if act != self.agent.action_space() and act != self.agent.reconnect_line(obs):
                print("hey")
                for sub in self.agent.action_converter.masked_sorted_sub:
                    if any(act.set_bus[self.begin[sub]:self.end[sub]] != 0):
                        acted_sub = sub
                        break
                if self.previous_sub is not None:
                    self.count_all[self.previous_sub, acted_sub] += 1
                    self.count_last[self.previous_sub, acted_sub] += 1
                self.previous_sub = acted_sub
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
        # print(f"Total number of training chronic: {total_chronic_num}")
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
        
        # create directory to store the training measures
        train_dir = os.path.join(output_dir, "train_measures")
        os.makedirs(train_dir, exist_ok=True)

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
            self.previous_sub = None
            self.counter = 0
            # update = False
            self.agent.reset(obs)
            prev_act = self.agent.act(obs, None, None)
            if prev_act != self.agent.action_space() and prev_act != self.agent.reconnect_line(obs):
                self.actions += 1
                for sub in self.agent.action_converter.masked_sorted_sub:
                    if any(prev_act.set_bus[self.begin[sub]:self.end[sub]] != 0):
                        self.previous_sub = sub
                        break
                # print("first, action:", self.actions)
            elif prev_act == self.agent.reconnect_line(obs):
                self.reconnects += 1
                # print("first reconnect line action:", self.actions2)
            elif prev_act == self.agent.action_space():
                self.donothings += 1
                # print("first, do nothing action:", self.actions3)

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

                    # Print the transition matrix for the last evaluation period
                    print(f"Middle-Agent transition matrix for the last {test_step} episodes:\n{self.count_last}")
                    # Reset the transition matrix for the next evaluation period
                    self.count_last.fill(0)

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
            if isinstance(self.agent, IMARL) or isinstance(self.agent, ReArIMARL):
                self.agent.print_updates_per_agent()
            # Print the full transition matrix
            print(f"Middle-Agent transition matrix:\n{self.count_all}") 

        return best_score

    def eval_train(self, valid_chron_ffw, output_dir, model_path, best_score, verbose=True):
        # start evaluation agent after test_step number of updates
        self.eval_train_count += 1
        eval_iter = self.agent.agent_step  # // test_step
        fill_outputs = False
        if (self.agent.update_step > self.last_number_updates) or (self.agent.update_step == 0):
            self.last_number_updates = self.agent.update_step
            fill_outputs = True
            if type(self.agent) == ReArIMARL:
                result, self.last_stats, self.last_scores, self.last_steps, topologies, unique_topos, unique_topos_total, substation_configs, is_safe, steps_overloaded, sub_depth, elem_depth, action_counts, ra_action_counts = self.test(valid_chron_ffw, verbose)
            else:
                result, self.last_stats, self.last_scores, self.last_steps, topologies, unique_topos, unique_topos_total, substation_configs, is_safe, steps_overloaded, sub_depth, elem_depth, action_counts = self.test(valid_chron_ffw, verbose)
            
            if (best_score - 0.5) < self.last_stats["score"]:  # update also when score is almost as good as best
                if verbose:
                    print(f"Found score higher or similar to best, save agent!")
                best_score = max(best_score, self.last_stats["score"])
                self.agent.save_model(model_path, "best")
                np.save(os.path.join(output_dir, f"best_score.npy"), best_score)

        if verbose:
            print(f"[{eval_iter:4d}] Valid: score {self.last_stats['score']} | step {self.last_stats['step']}")

        # New directory for the training measurements
        train_dir = os.path.join(output_dir, "train_measures")

        # log and save model
        with open(os.path.join(output_dir, "score.csv"), "a", newline="") as cf:
            csv.writer(cf).writerow([self.agent.agent_step] + self.last_scores)
        with open(os.path.join(output_dir, "step.csv"), "a", newline="") as cf:
            csv.writer(cf).writerow(self.last_steps)

        # Measures are computed only if the agent is updated (it will usually be the case, but not for short fast runs)
        if fill_outputs:
            with open(os.path.join(train_dir, "unique_topologies_chron.csv"), "a", newline="") as cf:
                csv.writer(cf).writerow(unique_topos)

            # Write other measures to CSV files using utility functions
            # only a selection of these will be tracked in the csv files
            if self.eval_train_count in range(10) or self.eval_train_count in [15, 25] or self.eval_train_count % 10 == 0:
                write_topologies_to_csv(topologies, os.path.join(train_dir, f"raw_topologies_{self.agent.agent_step}.csv"))
                write_unique_topos_total_to_csv(unique_topos_total, os.path.join(train_dir, f"unique_topologies_total_{self.agent.agent_step}.csv"))
                write_substation_configs_to_csv(substation_configs, os.path.join(train_dir, f"unique_substation_configurations_{self.agent.agent_step}.csv"))
                write_is_safe_to_csv(is_safe, os.path.join(train_dir, f"is_safe_{self.agent.agent_step}.csv"))
                write_steps_ol_to_csv(steps_overloaded, os.path.join(train_dir, f"steps_overloaded_{self.agent.agent_step}.csv"))
                write_depths_to_csv(sub_depth, os.path.join(train_dir, f"sub_depths_{self.agent.agent_step}.csv"))
                write_depths_to_csv(elem_depth, os.path.join(train_dir, f"elem_depths_{self.agent.agent_step}.csv"))
                write_action_counts_to_csv(os.path.join(train_dir, f"action_counts_{self.agent.agent_step}.csv"), action_counts)

                if type(self.agent) == ReArIMARL:
                    write_ra_action_counts_to_csv(os.path.join(train_dir, f"ra_action_counts_{self.agent.agent_step}.csv"), ra_action_counts)
                
                # Compute and write summary measures
                #compute_per_chronic_measures(train_dir, valid_chron_ffw, self.agent.agent_step)
                #compute_across_chronic_measures(train_dir, valid_chron_ffw, self.agent.agent_step)
            
        return best_score, self.last_stats["score"]

    def test(self, chron_ffw, verbose=True):
        result, sub_depth, elem_depth, topologies, is_safe, unique_topos_total, substation_configs = {}, {}, {}, {}, {}, {}, {}
        steps, scores, unique_topos = [], [], []
        action_counts = {chron_id: {sub: 0 for sub in self.agent.action_converter.masked_sorted_sub} for chron_id in chron_ffw}
        reconnections = 0
        if type(self.agent) == ReArIMARL:
                ra_action_counts = {chron_id: {ra: 0 for ra in self.agent.action_converter.ra_idx} for chron_id in chron_ffw}
        steps_overloaded = {chron_id: {0.9: 0, 0.95: 0, 0.96: 0, 0.98: 0, 0.99: 0, 1: 0} for chron_id in chron_ffw}
        
        def get_topo_id(topo_vect):
            subs_with_2 = [f"sub{sub}" for sub in self.agent.action_converter.masked_sorted_sub if np.any(topo_vect[self.begin[sub]:self.end[sub]] == 2)]
            return "_".join(subs_with_2)
        
        topo_counter = {}
        topo_id_counter = 1

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
            sub_depth[(cur_chron, 0)], elem_depth[(cur_chron, 0)] = 0, 0
            is_safe[(cur_chron, 0)] = self.agent.is_safe(obs)
            topologies[(cur_chron, 0)] = obs.topo_vect.copy()
            unique_ts = np.array([obs.topo_vect])

            while not done:
                for threshold in steps_overloaded[i]:
                    if max(obs.rho) >= threshold:
                        steps_overloaded[i][threshold] += 1
                
                act = self.agent.act(obs, 0, 0)

                if act != self.agent.action_space() and act != self.agent.reconnect_line(obs):
                    if type(self.agent) == ReArIMARL:
                        ra_action_counts[i][self.agent.action_converter.current_ra] += 1
                    # Increment action count for the corresponding substation
                    # print("action", act)
                    for sub in self.agent.action_converter.masked_sorted_sub:
                        if any(act.set_bus[self.begin[sub]:self.end[sub]] != 0):
                            action_counts[i][sub] += 1
                            acted_sub = sub
                            # print("sub", sub)
                            break  # Exit loop if action is found
                
                obs, reward, done, info = self.test_env.step(act)
                total_reward += reward
                alive_frame += 1

                if not np.any(np.all(unique_ts == obs.topo_vect, axis=1)) and np.all(obs.topo_vect != -1) and not done:
                    unique_ts = np.vstack((unique_ts, obs.topo_vect))
                    # print("new topo", prev_topo, "timestep", alive_frame)

                if not done:
                    topologies[(i, alive_frame)] = obs.topo_vect.copy()
                    is_safe[(i, alive_frame)] = self.agent.is_safe(obs)
                    
                    depth = 0
                    for sub in self.agent.action_converter.masked_sorted_sub:
                        #print(obs.topo_vect[begin:end], np.ones(end - begin, dtype=int))
                        sub_topo = obs.topo_vect[self.begin[sub]:self.end[sub]]
                        if 2 in sub_topo:
                            depth += 1
                            # print("sub depth", s_depth, "due to sub", sub, "timestep", alive_frame)

                    sub_depth[(i, alive_frame)] = depth
                    elem_depth[(i, alive_frame)] = np.sum((obs.topo_vect != 1) & (obs.topo_vect != -1))
                    
                    if act != self.agent.action_space() and act != self.agent.reconnect_line(obs):
                        topo_id = get_topo_id(obs.topo_vect)
                        if topo_id not in topo_counter:
                            topo_counter[topo_id] = 1
                        else:
                            topo_counter[topo_id] += 1

                        if str(obs.topo_vect.copy()) not in unique_topos_total:
                            unique_topos_total[str(obs.topo_vect.copy())] = {'count': 1, 'topo_id': f"{topo_id}_{topo_id_counter}"}
                            topo_id_counter += 1
                        else:
                            unique_topos_total[str(obs.topo_vect.copy())]['count'] += 1

                        sub_id = f"sub{acted_sub}"
                        sub_config = tuple(obs.topo_vect[self.begin[acted_sub]:self.end[acted_sub]])
                        if not np.array_equal(sub_config, np.ones(self.end[sub] - self.begin[sub], dtype=int)):
                            if sub_id not in substation_configs:
                                substation_configs[sub_id] = {}
                            if sub_config not in substation_configs[sub_id]:
                                substation_configs[sub_id][sub_config] = 1
                            else:
                                substation_configs[sub_id][sub_config] += 1

                if alive_frame == 864:
                    done = True

            l2rpn_score = float(self.compute_episode_score(cur_chron, alive_frame, total_reward, ffw))
            if verbose:
                print(
                    f"[Test Ch{cur_chron:4d}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) Score: {l2rpn_score:9.4f} "
                )
            scores.append(l2rpn_score)
            steps.append(alive_frame)
            unique_topos.append(len(unique_ts))
            # print("unique topologies", len(unique_ts))

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

        if type(self.agent) == ReArIMARL:
            return result, stats, scores, steps, topologies, unique_topos, unique_topos_total, substation_configs, is_safe, steps_overloaded, sub_depth, elem_depth, action_counts, ra_action_counts
        else:
            return result, stats, scores, steps, topologies, unique_topos, unique_topos_total, substation_configs, is_safe, steps_overloaded, sub_depth, elem_depth, action_counts

    def evaluate(self, chronics, max_ffw, path, sample=False):
        result, sub_depth, elem_depth, topologies, unique_topos_total, substation_configs, is_safe = {}, {}, {}, {}, {}, {}, {}
        steps, scores, unique_topos = [], [], []
        action_counts = {chron_id: {sub: 0 for sub in self.agent.action_converter.masked_sorted_sub} for chron_id in chronics}
        reconnections = 0
        if type(self.agent) == ReArIMARL:
            ra_action_counts = {chron_id: {ra: 0 for ra in self.agent.action_converter.ra_idx} for chron_id in chronics}
        steps_overloaded = {chron_id: {0.9: 0, 0.95: 0, 0.96: 0, 0.98: 0, 0.99: 0, 1: 0} for chron_id in chronics}

        def get_topo_id(topo_vect):
            subs_with_2 = [f"sub{sub}" for sub in self.agent.action_converter.masked_sorted_sub if np.any(topo_vect[self.begin[sub]:self.end[sub]] == 2)]
            return "_".join(subs_with_2)
        
        topo_counter = {}
        topo_id_counter = 1

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
            sub_depth[(chron_id, 0)], elem_depth[(chron_id, 0)] = 0, 0
            is_safe[(chron_id, 0)] = self.agent.is_safe(obs)
            topologies[(chron_id, 0)] = obs.topo_vect.copy()
            unique_ts = np.array([obs.topo_vect])
            
            while not done:
                for threshold in steps_overloaded[chron_id]:
                    if max(obs.rho) >= threshold:
                        steps_overloaded[chron_id][threshold] += 1

                act = self.agent.act(obs, None, 0) if sample else self.agent.act(obs, 0, 0)
                bus_goal = act.set_bus if act != self.agent.action_space() else obs.topo_vect
                prev_topo = obs.topo_vect
                prev_step = alive_frame
                
                if act != self.agent.action_space() and act != self.agent.reconnect_line(obs):
                    if type(self.agent) == ReArIMARL:
                        ra_action_counts[chron_id][self.agent.action_converter.current_ra] += 1
                    
                    for sub in self.agent.action_converter.masked_sorted_sub:
                        if any(act.set_bus[self.begin[sub]:self.end[sub]] != 0):
                            action_counts[chron_id][sub] += 1
                            acted_sub = sub
                            break
                    print("action at step", alive_frame, "affecting substation", acted_sub)
                elif act == self.agent.reconnect_line(obs):
                    reconnections += 1
                    print("reconnection at step", alive_frame)

                if not np.any(np.all(unique_ts == prev_topo, axis=1)) and np.all(obs.topo_vect != -1) and not done:
                    unique_ts = np.vstack((unique_ts, prev_topo))

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
                
                if not done:
                    topologies[(chron_id, alive_frame)] = obs.topo_vect.copy()
                    is_safe[(chron_id, alive_frame)] = self.agent.is_safe(obs)

                    depth = 0
                    for sub in self.agent.action_converter.masked_sorted_sub:
                        sub_topo = obs.topo_vect[self.begin[sub]:self.end[sub]]
                        if 2 in sub_topo:
                            depth += 1
                    sub_depth[(chron_id, alive_frame)] = depth
                    elem_depth[(chron_id, alive_frame)] = np.sum((obs.topo_vect != 1) & (obs.topo_vect != -1))

                    if act != self.agent.action_space() and act != self.agent.reconnect_line(obs):
                        topo_id = get_topo_id(obs.topo_vect)
                        if topo_id not in topo_counter:
                            topo_counter[topo_id] = 1
                        else:
                            topo_counter[topo_id] += 1

                        if str(obs.topo_vect.copy()) not in unique_topos_total:
                            unique_topos_total[str(obs.topo_vect.copy())] = {'count': 1, 'topo_id': f"{topo_id}_{topo_id_counter}"}
                            topo_id_counter += 1
                        else:
                            unique_topos_total[str(obs.topo_vect.copy())]['count'] += 1

                        sub_id = f"sub{acted_sub}"
                        sub_config = tuple(obs.topo_vect[self.begin[acted_sub]:self.end[acted_sub]])
                        if not np.array_equal(sub_config, np.ones(self.end[sub] - self.begin[sub], dtype=int)):
                            if sub_id not in substation_configs:
                                substation_configs[sub_id] = {}
                            if sub_config not in substation_configs[sub_id]:
                                substation_configs[sub_id][sub_config] = 1
                            else:
                                substation_configs[sub_id][sub_config] += 1

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
            unique_topos.append(len(unique_ts))
            #print("unique topologies:", len(unique_ts))
            #for i in unique_ts:
            #    print(i)

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
        if type(self.agent) == ReArIMARL:
            return stats, scores, steps, topologies, unique_topos, unique_topos_total, substation_configs, is_safe, steps_overloaded, sub_depth, elem_depth, action_counts, ra_action_counts
        else:
            return stats, scores, steps, topologies, unique_topos, unique_topos_total, substation_configs, is_safe, steps_overloaded, sub_depth, elem_depth, action_counts


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
            if act != self.agent.action_space() and act != self.agent.reconnect_line(obs):
                self.actions += 1
                for sub in self.agent.action_converter.masked_sorted_sub:
                    if any(act.set_bus[self.begin[sub]:self.end[sub]] != 0):
                        acted_sub = sub
                        break

                if self.previous_sub is not None and acted_sub is not None:
                    prev_index = self.substation_id_to_index[self.previous_sub]
                    act_index = self.substation_id_to_index[acted_sub]
                    self.count_all[prev_index, act_index] += 1
                    self.count_last[prev_index, act_index] += 1
                self.previous_sub = acted_sub

            elif act == self.agent.reconnect_line(obs):
                self.reconnects += 1
            elif act == self.agent.action_space():
                self.donothings += 1

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
                    self.previous_sub = None
                    break
                act = self.agent.act(obs, None, None)

                if act != self.agent.action_space() and act != self.agent.reconnect_line(obs):
                    self.actions += 1
                    for sub in self.agent.action_converter.masked_sorted_sub:
                        if any(act.set_bus[self.begin[sub]:self.end[sub]] != 0):
                            acted_sub = sub
                            break

                    if self.previous_sub is not None and acted_sub is not None:
                        prev_index = self.substation_id_to_index[self.previous_sub]
                        act_index = self.substation_id_to_index[acted_sub]
                        self.count_all[prev_index, act_index] += 1
                        self.count_last[prev_index, act_index] += 1
                    self.previous_sub = acted_sub

                elif act == self.agent.reconnect_line(obs):
                    self.reconnects += 1

                elif act == self.agent.action_space():
                    self.donothings += 1
                
                if self.agent.save or (train_reward > self.max_rw):
                    # a new actions has been generated OR the env has been safe for a long time
                    # pass this act to the next step.
                    prev_act = act
                    break
                elif skip:
                    skip = False

            # debugging info
            # print("printing from train.interaction", ", step: ", step)
            # print(act)

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
