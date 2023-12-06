import os
import re
import json
import random
import shutil
from datetime import datetime
from argparse import ArgumentParser
import torch
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore

from custom_reward import *
from MultiAgents.MultiAgent import IMARL, DepMARL
from Agents.sac import SAC, SMAAC
from Agents.SACD import SacdGoal, SacdSimple, SacdEmb
from Agents.DQN import DQN, DQN2
from Agents.PPO import PPO
from train import TrainAgent, Train

import matplotlib.cbook
import warnings
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import PerfectForecastHandler, CSVHandler

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", message=".*: No more data to get, the last known data is returned.")


ENV_CASE = {
    "5": "rte_case5_example",
    "14": "l2rpn_case14_sandbox",
}

DATA_SPLIT = {
    "5": ([i for i in range(20) if i not in [17, 19]], [17], [19]),
    "14": (
        list(range(0, 40 * 26, 40)),
        list(range(1, 100 * 10 + 1, 100)),
        list(range(2, 100 * 10 + 2, 100)),
    ),
}


def get_max_ffw(case):
    MAX_FFW = {"5": 5, "14": 26}
    return MAX_FFW[case]


def get_agent(agent_name):
    AGENT = {
        "sac": SMAAC,
        "sac2": SAC,
        "sacd": SacdSimple,
        "sacd_emb": SacdEmb,
        "sacd2": SacdGoal,
        "isacd_base": IMARL,
        "isacd_emb": IMARL,
        "dsacd_emb": DepMARL,
        "dqn": DQN,
        "dqn2": DQN2,
        "ppo": PPO,
        "ippo": IMARL,
        "dppo": DepMARL,
    }
    return AGENT[agent_name]


def cli():
    parser = ArgumentParser()

    # Save / load location and name
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="change the directory used for storing environment + data",
        default="",
    )
    parser.add_argument("-n", "--name", type=str, default="untitled")
    parser.add_argument(
        "-la",
        "--load_agent",
        type=str,
        default="",
        help="load a learned agents parameters to continue training",
    )

    # General env parameters
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--case", type=str, default="5", choices=["14", "5"])
    parser.add_argument("-rw", "--reward", type=str, default="margin", choices=["loss", "margin"])
    parser.add_argument("-gpu", "--gpuid", type=int, default=0)
    parser.add_argument("-ml", "--memlen", type=int, default=50000)
    parser.add_argument(
        "-ns",
        "--nb_steps",
        type=int,
        default=100,
        help="total number of agent steps",
    )
    parser.add_argument(
        "-ev",
        "--eval_steps",
        type=int,
        default=50,
        help="the number of steps between each evaluation",
    )

    parser.add_argument(
        "-m",
        "--mask",
        type=int,
        default=3,
        help='this agent manages the substations containing topology elements over "mask"',
    )
    parser.add_argument("-mr", "--max_reward", type=int, default=10, help="max reward during training")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        default=["p_i", "r", "o", "d", "m"],
        help="Options for input information in the GNN.\n "
        ' "p_i" - active power input (gen/loads), \n'
        ' "p_l" - active power lines, \n'
        ' "r" - rho (line capacity rate), \n'
        ' "o" - overflow, \n'
        ' "d" - danger, \n'
        ' "m" - maintenance',
    )
    parser.add_argument("-fc", "--forecast", type=int, default=0, help="forecast as input for training")

    # Highest level agent parameter
    parser.add_argument(
        "-dg",
        "--danger",
        type=float,
        default=0.9,
        help="the powerline with rho over danger is regarded as hazardous",
    )

    # Middle level agent
    parser.add_argument(
        "-ma",
        "--middle_agent",
        type=str,
        default="capa",
        choices=["fixed", "random", "capa"],
    )

    # Choose the lowest level agent
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        default="sacd_emb",
        choices=[
            "sac",
            "sac2",
            "sacd",
            "sacd_emb",
            "sacd2",
            "isacd_base",
            "isacd_emb",
            "dsacd_emb",
            "dqn",
            "dqn2",
            "ppo",
            "ippo",
            "dppo",
        ],
    )

    # (Deep) learning  parameters
    parser.add_argument("-nn", "--network", type=str, default="lin")
    parser.add_argument(
        "-hn",
        "--head_number",
        type=int,
        default=8,
        help="the number of head for attention",
    )
    parser.add_argument(
        "-sd",
        "--state_dim",
        type=int,
        default=128,
        help="dimension of hidden state for GNN",
    )
    parser.add_argument("-nh", "--n_history", type=int, default=6, help="length of frame stack")
    parser.add_argument("-do", "--dropout", type=float, default=0.0)
    parser.add_argument("-nl", "--n_layers", type=int, default=3)

    parser.add_argument("-lr", "--lr", type=float, default=5e-3)
    parser.add_argument("-g", "--gamma", type=float, default=0.995)

    # batch sizes and updates
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument(
        "-u",
        "--update_start",
        type=int,
        default=2,
        help="updating the agent starts after *n* x batch_size",
    )

    # SMAAC specific
    parser.add_argument(
        "-r",
        "--rule",
        type=str,
        default="c",
        choices=["c", "d", "o", "f"],
        help="low-level rule (capa, desc, opti, fixed)",
    )
    parser.add_argument(
        "-thr",
        "--threshold",
        type=float,
        default=0.1,
        help="[-1, thr) -> bus 1 / [thr, 1] -> bus 2",
    )

    # SACD parameters
    parser.add_argument("-tu", "--target_update", type=int, default=1, help="period of target update")
    parser.add_argument("--tau", type=float, default=1e-3, help="the weight of soft target update")
    parser.add_argument(
        "-te",
        "--target_entropy_scale",
        type=float,
        default=0.98,
        help="coefficient for scaling the autotune entropy target",
    )

    # PPO parameters
    parser.add_argument("-ep", "--epsilon", type=float, default=0.2, help="clipping ratio for PPO")
    parser.add_argument("-en", "--entropy", type=float, default=0.01)
    parser.add_argument("-l", "--lambda", type=float, default=0.95, help="GAE parameter PPO")

    args = parser.parse_args()
    return args


def log_params(args, path):
    f = open(os.path.join(path, "param.txt"), "w")
    for key, val in args.__dict__.items():
        f.write(key + ": " + str(val) + "\n")
    f.close()
    with open(os.path.join(path, "param.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f)


def read_ffw_json(path, chronics, case):
    res = {}
    for i in chronics:
        for j in range(get_max_ffw(case)):
            with open(os.path.join(path, f"{i}_{j}.json"), "r", encoding="utf-8") as f:
                a = json.load(f)
                res[(i, j)] = (
                    a["dn_played"],
                    a["donothing_reward"],
                    a["donothing_nodisc_reward"],
                )
            if i >= 2880:
                break
    return res


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def def_env(args):
    return grid2op.make(
        ENV_CASE[args.case],
        test=True if args.case == "5" else False,
        reward_class=L2RPNSandBoxScore,
        backend=LightSimBackend(),
        # chronics_class=MultifolderWithCache,
        other_rewards={"loss": LossReward, "margin": MarginReward},
        data_feeding_kwargs={
            "gridvalueClass": FromHandlers,
            "gen_p_handler": CSVHandler("prod_p"),
            "load_p_handler": CSVHandler("load_p"),
            "gen_v_handler": CSVHandler("prod_v"),
            "load_q_handler": CSVHandler("load_q"),
            # "h_forecast": [5 * args.forecast],
            "h_forecast": [(i + 1) * 5 for i in range(args.forecast)],
            "gen_p_for_handler": PerfectForecastHandler("prod_p_forecasted"),
            "load_p_for_handler": PerfectForecastHandler("load_p_forecasted"),
            "load_q_for_handler": PerfectForecastHandler("load_q_forecasted"),
        },
    )


def make_envs(args):
    env_name = ENV_CASE[args.case]
    my_dir = args.dir if args.dir else "."
    DATA_DIR = os.path.join(my_dir, "data")
    env_path = os.path.join(DATA_DIR, env_name)
    env = def_env(args)
    test_env = def_env(args)

    if not args.forecast:
        env.deactivate_forecast()
        test_env.deactivate_forecast()
    test_env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
    test_env.parameters.NB_TIMESTEP_RECONNECTION = env.parameters.NB_TIMESTEP_RECONNECTION = 12
    test_env.parameters.NB_TIMESTEP_COOLDOWN_LINE = env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
    test_env.parameters.NB_TIMESTEP_COOLDOWN_SUB = env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
    test_env.parameters.HARD_OVERFLOW_THRESHOLD = env.parameters.HARD_OVERFLOW_THRESHOLD = 200.0
    # test_env.seed(59)
    # print(env.parameters.__dict__)
    return env, test_env, env_path


def select_chronics(env_path, env, test_env, case, eval=False):
    # select chronics end define dn_ffw
    dn_json_path = os.path.join(env_path, "json")
    train_chronics, valid_chronics, test_chronics = DATA_SPLIT[case]
    ep_infos = {}
    if eval:
        chronics = test_chronics
    else:
        chronics = train_chronics + valid_chronics
    dn_ffw = read_ffw_json(dn_json_path, chronics, case)

    if os.path.exists(dn_json_path):
        for i in list(set(chronics)):
            with open(os.path.join(dn_json_path, f"{i}.json"), "r", encoding="utf-8") as f:
                ep_infos[i] = json.load(f)

    train_chron_str = "|".join(["%02d" % chr if case == "5" else "%04d" % chr for chr in train_chronics])
    env.chronics_handler.set_filter(lambda path: re.match(".*(%s)$" % train_chron_str, path) is not None)
    kept = env.chronics_handler.real_data.reset()
    valid_chron_str = "|".join(["%02d" % chr if case == "5" else "%04d" % chr for chr in valid_chronics])
    test_env.chronics_handler.set_filter(lambda path: re.match(".*(%s)$" % valid_chron_str, path) is not None)
    kept = test_env.chronics_handler.real_data.reset()

    return train_chronics, valid_chronics, test_chronics, dn_ffw, ep_infos


def load_agent(args, my_dir, my_agent):
    model_input = f"{args.load_agent}"
    load_model_dir = os.path.join(my_dir, model_input)
    with open(os.path.join(load_model_dir, "param.json"), "rt") as f:
        params = json.load(f)
        if params["agent"] != args.agent:
            raise TypeError(f"Model type agent loaded is {params['agent']} but current agent is {args.agent}")
        if params["case"] != args.case:
            raise Exception(
                f"Model of agent loaded is for" f" case {params['case']}, but current agent is for case {args.case}"
            )
        if params["forecast"] != args.forecast:
            raise Exception(
                f"Model of agent loaded contains"
                f" {params['forecast']} steps of forecast, but current agent has {args.forecast} steps"
            )

    # copy current score.csv and step.csv
    shutil.copy(os.path.join(load_model_dir, "score.csv"), output_result_dir)
    shutil.copy(os.path.join(load_model_dir, "step.csv"), output_result_dir)
    # load agent model
    load_model_path = os.path.join(load_model_dir, "model")
    print(f"\n >>> Loading agent from {load_model_dir}...")
    my_agent.load_model(load_model_path, "last")
    best_score = np.load(os.path.join(load_model_dir, "best_score.npy"))
    return best_score


def make_dirs(model_name, result_dir):
    output_result_dir = os.path.join(result_dir, model_name)
    model_path = os.path.join(output_result_dir, "model")
    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)
        os.makedirs(model_path)
    return output_result_dir, model_path


def make_agent(env, env_path, kwargs):
    AGENT = get_agent(kwargs["agent"])
    my_agent = AGENT(env, **kwargs)
    mean = torch.load(os.path.join(env_path, "mean.pt"))
    std = torch.load(os.path.join(env_path, "std.pt"))
    my_agent.load_mean_std(mean.unsqueeze(0), std.unsqueeze(0))
    return my_agent


if __name__ == "__main__":
    args = cli()
    print(args)
    seed_everything(args.seed)

    # settings
    model_name = f"{args.name}_{args.agent}_{args.seed}"
    print("model name: ", model_name)
    TRAINER = TrainAgent if args.agent == "sac" else Train

    my_dir = args.dir if args.dir else "."
    output_result_dir, model_path = make_dirs(model_name, os.path.join(my_dir, "result"))
    log_params(args, output_result_dir)

    if torch.cuda.is_available():
        print(">> >> using cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define environments
    env, test_env, env_path = make_envs(args)
    # env.seed(9)

    # Select chronics end define dn_ffw
    train_chronics, valid_chronics, _, dn_ffw, ep_infos = select_chronics(env_path, env, test_env, args.case)
    # specify agent
    my_agent = make_agent(env, env_path, vars(args))

    best_score = -100
    if args.load_agent:
        best_score = load_agent(args, my_dir, my_agent)

    trainer = TRAINER(
        my_agent,
        env,
        test_env,
        dn_ffw,
        ep_infos,
        max_reward=args.max_reward,
        rw_func=args.reward,
    )

    tic = datetime.now()
    trainer.train(
        args.seed,
        args.nb_steps,
        args.eval_steps,
        train_chronics,
        valid_chronics,
        output_result_dir,
        model_path,
        get_max_ffw(args.case),
        best_score=best_score,
    )
    toc = datetime.now()
    print("Duration of training the agent: ", toc - tic)
