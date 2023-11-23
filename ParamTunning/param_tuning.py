import time
from datetime import datetime
import torch
import os
import pickle as pkl
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.exceptions import ExperimentalWarning

from test import (
    make_envs,
    select_chronics,
    make_dirs,
    get_max_ffw,
    make_agent,
    seed_everything,
)
from train import ParamTuningTrain
from ParamTunning.sample_params import sample_params
from util import plot_trials

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)


def cli():
    parser = ArgumentParser()

    # Save location and name
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="..",
        help="change the directory used for storing environment + data",
    )
    parser.add_argument("-n", "--name", type=str, default="untitled")
    # Load a study:
    parser.add_argument("-l", "--load_study", type=str, default="")

    # General env parameters
    parser.add_argument("-s", "--seeds", type=int, default=2, help="number of seeds")
    parser.add_argument("-c", "--case", type=str, default="5", choices=["14", "wcci", "5"])
    parser.add_argument("-gpu", "--gpuid", type=int, default=0)
    parser.add_argument("-ml", "--memlen", type=int, default=50000)
    parser.add_argument(
        "-ns",
        "--nb_steps",
        type=int,
        default=200,
        help="the total number of agent steps",
    )

    parser.add_argument(
        "-ev",
        "--eval_steps",
        type=int,
        default=25,
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
        default="ppo",
        choices=[
            "sac",
            "sac2",
            "sacd",
            "sacd2",
            "masacd",
            "dsacd",
            "dqn",
            "dqn2",
            "ppo",
            "mappo",
            "dppo",
        ],
    )

    # (Deep) learning  parameters # fixed for now.
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
    parser.add_argument("-nl", "--n_layers", type=int, default=1)

    # Parameter tuning arguments
    parser.add_argument(
        "-tr",
        "--n_trials",
        type=int,
        default=4,
        help="Maximum number of trials for param tuning",
    )
    parser.add_argument("-nj", "--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    parser.add_argument(
        "-st",
        "--n_startup_trials",
        type=int,
        default=3,
        help="Stop random sampling after N_STARTUP_TRIALS",
    )
    parser.add_argument(
        "-ws",
        "--n_warmup_steps",
        type=int,
        default=3,
        help="Number of steps before evaluations during the training. -> "
        "Pruning is disabled until the trial exceeds the given number of step.",
    )
    parser.add_argument("-to", "--time_out", type=int, default=5, help="number of hours before time out")

    args = parser.parse_args()
    return args


def trial_one_seed(
    env,
    env_path,
    kwargs,
    test_env,
    dn_ffw,
    ep_infos,
    seed,
    train_chronics,
    valid_chronics,
    output_result_dir,
    model_path,
    trial,
    best_score,
):
    case = kwargs.get("case")
    nb_steps = kwargs.get("nb_steps")
    eval_steps = kwargs.get("eval_steps")
    seed_everything(seed)
    agent = make_agent(env, env_path, kwargs)
    trainer = ParamTuningTrain(agent, env, test_env, dn_ffw, ep_infos, trial=trial, seed=seed)
    output_result_dir = os.path.join(output_result_dir, f"seed_{seed}")
    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)
    # Train the model
    best_score = trainer.train(
        seed,
        nb_steps,
        eval_steps,
        train_chronics,
        valid_chronics,
        output_result_dir,
        model_path,
        get_max_ffw(case),
        best_score=best_score,
        verbose=False,
    )

    if trainer.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return trainer.last_mean_reward, best_score


def objective(trial: optuna.Trial) -> float:
    # default params
    args = cli()
    agent_name = args.agent
    # add optuna trial params:
    kwargs = vars(args)
    kwargs.update(sample_params(trial, agent_name))

    if trial.number < args.n_jobs:
        time.sleep(trial.number)
    print(f"\n ******  START TRIAL {trial.number} at {time.asctime()} ****** ")
    tic = datetime.now()

    # make directories
    my_dir = args.dir if args.dir else "."
    output_result_dir = os.path.join(os.path.join(my_dir, "result"), f"{args.name}_{args.agent}_tuning")
    trial_path = f"trial_{trial.number}"
    output_result_dir, model_path = make_dirs(trial_path, output_result_dir)

    # Define environments
    env, test_env, env_path = make_envs(args)
    # TO DO: -> When we want to use seed in future envs (with maintenance and opponnent) fix the seed for every test.
    # Select chronics end define dn_ffw
    train_chronics, valid_chronics, _, dn_ffw, ep_infos = select_chronics(env_path, env, test_env, args.case)

    seeds = range(args.seeds)
    nan_encountered = False
    last_mean_reward = np.zeros(args.seeds)
    best_score = -100
    # multi_envs = [env for _ in range(NUM_CORE)]
    try:
        for seed in seeds:
            last_mean_reward[seed], best_score = trial_one_seed(
                env,
                env_path,
                kwargs,
                test_env,
                dn_ffw,
                ep_infos,
                seed,
                train_chronics,
                valid_chronics,
                output_result_dir,
                model_path,
                trial,
                best_score,
            )

        # with multiprocessing.Pool(processes=len(seeds)) as pool:
        #     last_mean_reward = pool.starmap(test_one_seed, [
        #         (my_env, env_path, kwargs, test_env, dn_ffw, ep_infos, seed,
        #          train_chronics, valid_chronics, output_result_dir, model_path, trial)
        #         for seed, my_env in zip(seeds, multi_envs)])
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        env.close()
        test_env.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    toc = datetime.now()
    print(f"    Duration of trial {trial.number}: {toc - tic} ")

    return last_mean_reward.mean()


def report_trials(study, output_dir):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{output_dir}/report.csv")

    with open(f"{output_dir}/study.pkl", "wb+") as f:
        pkl.dump(study, f)


def save_optuna_visualizations(output_dir):
    from optuna.visualization.matplotlib import (
        plot_param_importances,
        plot_parallel_coordinate,
        plot_timeline,
        plot_intermediate_values,
        plot_contour,
    )

    try:
        images_dir = os.path.join(output_dir, "images")
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
        # fig = plot_optimization_history(study)
        # plt.savefig(os.path.join(images_dir, "plot_optimization_history.png"))
        fig = plot_param_importances(study)
        plt.savefig(os.path.join(images_dir, "plot_param_importances.png"))
        fig = plot_parallel_coordinate(study)
        f = plt.gcf()
        f.set_size_inches(15, 7)
        plt.savefig(os.path.join(images_dir, "plot_parallel_coordinate.png"))
        fig = plot_timeline(study)
        plt.savefig(os.path.join(images_dir, "plot_timeline.png"))
        fig = plot_intermediate_values(study)
        f = plt.gcf()
        f.set_size_inches(15, 7)
        plt.savefig(os.path.join(images_dir, "plot_intermediate_values.png"))
        fig = plot_contour(study, params=["lr", "gamma"])
        plt.savefig(os.path.join(images_dir, "plot_contour.png"))
        plt.clf()
        plt.cla()
    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)


if __name__ == "__main__":
    args = cli()
    print(args)
    my_dir = args.dir if args.dir else "."
    output_result_dir = os.path.join(os.path.join(my_dir, "result"), f"{args.name}_{args.agent}_tuning")
    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)

    if torch.cuda.is_available():
        print(">> >> using cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.n_jobs)

    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=args.n_startup_trials)
    # Define pruner
    pruner = MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps)
    # pruner = optuna.pruners.PatientPruner(pruner, patience=3)

    # Create the study and start the hyperparameter optimization
    if args.load_study:
        path_study = os.path.join(os.path.join(my_dir, "result"), args.load_study)
        file_name = os.path.join(path_study, "study.pkl")
        study = joblib.load(file_name)

        storage = optuna.storages.RDBStorage(url="sqlite:///example.db")
        try:
            study_id = storage.create_new_study(
                [optuna.study.StudyDirection.MAXIMIZE],
                study_name=f"tuning_{args.agent}",
            )
        except optuna.exceptions.DuplicatedStudyError:
            study_id = storage.get_study_id_from_name(f"tuning_{args.agent}")
            storage.delete_study(study_id)
            study_id = storage.create_new_study(
                [optuna.study.StudyDirection.MAXIMIZE],
                study_name=f"tuning_{args.agent}",
            )

        for trial in study.get_trials():
            storage.create_new_trial(study_id=study_id, template_trial=trial)

        study = optuna.load_study(
            sampler=sampler,
            pruner=pruner,
            study_name=f"tuning_{args.agent}",
            storage=storage,
        )
    else:
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            study_name=f"tuning_{args.agent}",
        )

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.time_out * 3600,
        )
    except KeyboardInterrupt:
        pass

    report_trials(study, output_result_dir)
    save_optuna_visualizations(output_result_dir)
    plot_trials(
        output_result_dir,
        len(study.trials),
        args.seeds,
        args.nb_steps,
        name=f"{args.name}_{args.agent}",
    )
    print(f"\n-----  Saved results study in: {output_result_dir} -----")
