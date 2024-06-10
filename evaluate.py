import os
import csv
import json
from argparse import ArgumentParser, Namespace
import torch

from test import make_envs, select_chronics, make_agent, log_params
from train import TrainAgent, Train
from util import (write_depths_to_csv, write_action_counts_to_csv, write_steps_ol_to_csv, write_topologies_to_csv, 
                write_is_safe_to_csv, write_ra_action_counts_to_csv, write_substation_configs_to_csv, write_unique_topos_total_to_csv,
                compute_per_chronic_measures, compute_across_chronic_measures)
import warnings
import glob

warnings.filterwarnings("ignore", message=".*: No more data to get, the last known data is returned.")

MAX_FFW = {"5": 5, "14": 26, "wcci": 26}

TRAINER = {"1": TrainAgent, "2": Train}


def cli():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="", help="the directory used for data")
    parser.add_argument(
        "-rd",
        "--res_dir",
        type=str,
        default="results_snellius/case_14",
        help="",
    )
    parser.add_argument("-mn", "--model_name", type=str, default="final_tr13_case14_ppo")
    parser.add_argument("-gpu", "--gpuid", type=int, default=0)

    parser.add_argument("-l", "--last", action="store_true")
    parser.add_argument(
        "-s", "--sample", help="sample action instead of always take mean / argmax", action="store_true"
    )

    args, _ = parser.parse_known_args()
    return args, parser


def args_agent(path, known_args):
    print(f"Gathering arguments from {path}... Note assuming all agents have same args!")
    with open(os.path.join(path, "param.json"), "rt") as f:
        agent_args = Namespace()
        agent_args.__dict__.update(json.load(f))
        agent_args.__dict__.update(known_args.__dict__)
    return agent_args


def read_loss_json(path, chronics):
    losses = {}
    loads = {}
    chronics = list(set(chronics))
    for i in chronics:
        json_path = os.path.join(path, "%s.json" % i)
        with open(json_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        losses[i] = res["losses"]
        loads[i] = res["sum_loads"]
    return losses, loads


if __name__ == "__main__":
    args, parser = cli()

    my_dir = args.dir if args.dir else "."

    # settings
    model_name = f"{args.model_name}"
    print("model name: ", model_name)

    my_dir = args.dir if args.dir else "."
    output_result_dir = os.path.join(my_dir, args.res_dir)
    # get arguments from the first agent param.json
    all_agents = glob.glob(f"{output_result_dir}//{model_name}_*")
    args = args_agent(all_agents[0], args)  # assuming all agents have the same params
    log_params(args, output_result_dir)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define which trainer function to use
    TRAINER = TrainAgent if args.agent == "sac" else Train
    # Define environments
    env, test_env, env_path = make_envs(args)
    # Select chronics end define dn_ffw
    _, _, test_chronics, dn_ffw, ep_infos = select_chronics(env_path, env, test_env, args.case, eval=True)

    mode = "last" if args.last else "best"

    # specify agent
    my_agent = make_agent(env, env_path, vars(args))

    trainer = TRAINER(my_agent, env, test_env, dn_ffw, ep_infos, max_reward=args.max_reward)

    for agent_dir in all_agents:
        print("Evaluating agent ", agent_dir)
        model_path = os.path.join(agent_dir, "model")
        my_agent.load_model(model_path, mode)

        if not os.path.exists(agent_dir):
            os.makedirs(agent_dir)
        
        agent_dir = os.path.join(agent_dir, "evaluation_measures")
        if not os.path.exists(agent_dir):
            os.makedirs(agent_dir)
        
        if args.agent == "raippo":
            stats, scores, steps, topologies, unique_topos, unique_topos_total, substation_configs, is_safe, steps_overloaded, sub_depths, elem_depths, action_counts, ra_action_counts = trainer.evaluate(test_chronics, MAX_FFW[args.case], agent_dir, args.sample)
        else:
            stats, scores, steps, topologies, unique_topos,  unique_topos_total, substation_configs, is_safe, steps_overloaded, sub_depths, elem_depths, action_counts = trainer.evaluate(test_chronics, MAX_FFW[args.case], agent_dir, args.sample)
        # mode, plot_topo=True)
        with open(os.path.join(agent_dir, "score.csv"), "a", newline="") as cf:
            csv.writer(cf).writerow(scores)
        with open(os.path.join(agent_dir, "step.csv"), "a", newline="") as cf:
            csv.writer(cf).writerow(steps)
        with open(os.path.join(agent_dir, "unique_topologies_chron.csv"), "a", newline="") as cf:    
            csv.writer(cf).writerow(unique_topos)

        # dictionaries are treated differently
        write_unique_topos_total_to_csv(unique_topos_total, os.path.join(agent_dir, "unique_topologies_total.csv"))
        write_substation_configs_to_csv(substation_configs, os.path.join(agent_dir, "unique_substation_configurations.csv"))
        write_topologies_to_csv(topologies, os.path.join(agent_dir, "raw_topologies.csv"))
        write_is_safe_to_csv(is_safe, os.path.join(agent_dir, "is_safe.csv"))
        write_steps_ol_to_csv(steps_overloaded, os.path.join(agent_dir, "steps_overloaded.csv"))
        write_depths_to_csv(sub_depths, os.path.join(agent_dir, "sub_depths.csv"))
        write_depths_to_csv(elem_depths, os.path.join(agent_dir, "elem_depths.csv"))
        write_action_counts_to_csv(os.path.join(agent_dir, "action_counts.csv"), action_counts)

        if args.agent == "raippo":
            write_ra_action_counts_to_csv(os.path.join(agent_dir, "ra_action_counts.csv"), ra_action_counts)

        # Compute and write summary measures
        #chronic_ids = test_chronics  # List of chronic IDs used in evaluation
        #compute_per_chronic_measures(agent_dir, chronic_ids)
        #compute_across_chronic_measures(agent_dir, chronic_ids)
