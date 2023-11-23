import os
import random
import numpy as np
import csv
import json
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors


colors = [i for i in mcolors.TABLEAU_COLORS.keys()]


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(list(self.memory), n)


def show_results(res_dir, case, n, name=""):
    """
    This function plots the test results that are saved during the training of the agent. It shows the result
    for each seed in [0-n].
    Next to a plot the output will give a pandas dataframe with a summary of the results with the following columns:
    Maximum steps,
        The maximum number of steps this agent was able to achieve in an episode (max 864)
    Agent updates (steps),
        After how many updates of the agent (training iterations) was the agent able to achieve this maximum number
        of steps.
    Best mean,
        Best mean score that the agent got during the training.
    Agent updates (mean)
        After how many updates of the agent (training iterations) was the agent able to achieve this best mean score.

    Parameters
    ----------

    res_dir: ``str`` or path
        Name of the directory where the results of a trained agents are saved.
    case: ``str``
        Name of the agents e.g. "case_5_ppo"
    n: ``int``
        number of seeds for which this agent is trained. The results will be plotted per seed: e.g.
        case_5_ppo_0 ... case_5_ppo_4
    name: ``str``
        Optional. This will be the last part of the title of the plot, which is of the form
        "Training *agent* - *case* *name*"

    Example
    ----------
    case = 'final_case5_sacd'
    seeds = 5
    summary, res_sacd = show_results(res_dir, case, seeds, name='')

    """
    df = pd.DataFrame()
    if type(n) is int:
        seeds = range(n)
    else:
        seeds = n
        n = len(n)

    best_steps = np.zeros(n)
    best_mean = np.zeros(n)
    first = np.zeros(n)
    it = np.zeros(n)

    script_path = os.path.join(res_dir, f"{case}_{seeds[0]}")
    with open(os.path.join(script_path, "param.json"), "r") as f:
        params = json.load(f)

    params_to_print = ["name", "batch_size", "update_start", "gamma", "lr"]
    [print(f"{key}:\t {params.get(key)}") for key in params_to_print]

    name = f"{params['agent'].upper()} - case {params['case']} {name}"
    for i, seed in enumerate(seeds):
        steps = pd.read_csv(f"{res_dir}/{case}_{seed}/step.csv", names=range(5))
        score = pd.read_csv(f"{res_dir}/{case}_{seed}/score.csv")
        sem = score.iloc[:, 1:].sem(axis=1)
        mean = score.iloc[:, 1:].mean(axis=1)
        plt.fill_between(
            score["env_interactions"], mean - sem, mean + sem, interpolate=True, color=colors[seed], alpha=0.1
        )
        plt.plot(score["env_interactions"], mean, label=f"seed {seed}", color=colors[seed])
        df["env_interactions"] = score["env_interactions"]
        df[f"mean s{i}"] = mean
        best_steps[i] = steps.mean(axis=1).max()
        first[i] = steps.mean(axis=1).argmax()
        best_mean[i] = mean.max()
        it[i] = mean.argmax()

    df["env_interactions"] = score["env_interactions"]
    summary = pd.DataFrame(
        {"Max steps": best_steps, "Agent updates (steps)": first, "Best mean": best_mean, "Agent updates (mean)": it}
    )
    summary.index.name = "seed"

    plt.title(label=f"Training {name}", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=24)
    plt.ylim(ymin=-100, ymax=105)
    plt.xlim(xmin=0, xmax=params["nb_steps"])
    plt.xlabel("environment interactions", fontsize=16)
    plt.ylabel("mean score", fontsize=16)
    plt.legend(fontsize=18, loc="lower right")
    f = plt.gcf()
    f.set_size_inches(15, 7)
    return summary, df


def compare_results(cases, list_data_frames, style="default"):
    """
    With this function you can plot the results of multiple different agents.

    Parameters
    ----------

    cases: ``list``
        A list of strings that will represent the labels in the plot.
    list_data_frames: ``list``
        A list of data frames that should correspond to the labels in cases. Note that we need
        size(cases) == size(list_data_frames)
    style: ``str`` or ``list``
        If you use
        import scienceplots
        you can create nice and clear scientific plots.

    Example
    __________
    figure = compare_results(cases, [res_dsacd, res_fix_dsacd, res_rand_dsacd], style=['science', 'grid', 'no-latex'])

    """
    plt.style.use(style)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, all_scores in enumerate(list_data_frames):
        sem = all_scores.iloc[:, 1:].sem(axis=1)
        mean = all_scores.iloc[:, 1:].mean(axis=1)
        plt.fill_between(
            all_scores["env_interactions"], mean - sem, mean + sem, interpolate=True, color=colors[idx], alpha=0.1
        )
        plt.plot(all_scores["env_interactions"], mean, label=cases[idx], color=colors[idx])

    plt.title(label="Training Scores", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(xmin=0, xmax=10_000)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
    plt.ylim(ymin=-105, ymax=105)
    plt.xlabel("environment interactions", fontsize=16)
    plt.ylabel("mean score", fontsize=16)
    plt.legend(fontsize=18, loc="lower right")
    f = plt.gcf()
    f.set_size_inches(15, 7)
    return f


def get_trial_res(trial_dir, n_seeds):
    df = pd.DataFrame()
    for seed in range(n_seeds):
        try:
            score = pd.read_csv(f"{trial_dir}/seed_{seed}/score.csv")
            mean = score.iloc[:, 1:].mean(axis=1)
            df["env_interactions"] = score["env_interactions"]
            df[f"mean s{seed}"] = mean
        except:
            print(f"{trial_dir} does not have any values for seed {seed}, this trial probably got pruned")
    return df


def plot_trials(output_dir, n_trials, n_seeds, max_steps, name=""):
    """
    This function is used to plot the results of the parameter tunning. For usage see also param_tunning.py

    Parameters
    ----------
    output_dir: ``str``
        Directory where the trial results are stored.
    n_trials: ``int`` or ``list`` or ``range``
        trials to plot. If integer all trials in [0 ... n_trials] will be plotted.
    n_seeds: ``int``
        number of seeds that have been tested for each trial.
    max_steps: ``int``
        Maximum number of environment interactions (can be less if the process was stopped in the middle.)

    Example
    __________
    For usage see also param_tunning.py

    dir_name = "final_param_case5_dppo_tuning"
    trials = range(16,21)
    n_seeds = 3
    max_steps = 5000
    plot_trials(dir_name, trials, n_seeds, max_steps)


    or specifically select trials with better results:

    dir_name = "final_param_case14_ppo_tuning"
    param_res = get_param_res(dir_name)
    n_seeds = 3
    max_steps = 50_000
    trials = list(param_res.loc[(param_res['value'] > 60)]['number'])
    plot_trials(dir_name, trials, n_seeds, max_steps)

    """
    plt.style.use("default")
    fig = plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if isinstance(n_trials, int):
        trials = range(n_trials)
    elif isinstance(n_trials, list) or isinstance(n_trials, range):
        trials = n_trials
    else:
        raise TypeError("n_trials should be integer, list or range!")
    for t in range(trials):
        trial_path = os.path.join(output_dir, f"trial_{t}")
        if os.path.exists(trial_path):
            mean_scores = get_trial_res(trial_path, n_seeds)
            sem = mean_scores.iloc[:, 1:].sem(axis=1)
            mean = mean_scores.iloc[:, 1:].mean(axis=1)
            plt.fill_between(
                mean_scores["env_interactions"],
                mean - sem,
                mean + sem,
                interpolate=True,
                color=colors[t % len(colors)],
                alpha=0.1,
            )
            plt.plot(mean_scores["env_interactions"], mean, label=f"trial {t}", color=colors[t % len(colors)])
    plt.title(label=f"Mean scores - {name} ", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(xmin=0, xmax=max_steps)
    plt.ylim(ymin=-100, ymax=105)
    plt.xlabel("environment interactions", fontsize=16)
    plt.ylabel("mean score", fontsize=16)
    plt.legend(fontsize=18, loc="lower right")
    # fig = plt.gcf()
    fig.set_size_inches(15, 7)
    fig.savefig(os.path.join(output_dir, "images/mean_scores"))
    plt.clf()
    plt.cla()


# plot functionalities
def plot_action_gantt(folder):
    files = [file for file in os.listdir(folder) if file.endswith("_topo.npy") & file.startswith("Ch")]
    log_folder = os.path.join(folder, "eval_actions_gantts")
    os.makedirs(log_folder, exist_ok=True)
    if files:
        for file in files:
            topo = np.load(os.path.join(folder, file))
            test_case = "_".join(file.split("_", 2)[:2])

            fig = plt.figure(figsize=(20, topo.shape[1] / 2))
            ax = fig.add_subplot(1, 1, 1)
            plt.title(test_case, fontsize=20, fontweight="bold")
            colors = ["limegreen", "tab:red", "silver"]
            curr = 1
            for j in range(topo.shape[1]):
                start_pos = 0
                end_pos = 0
                for i in range(topo.shape[0]):
                    if topo[:, j][i] != curr:
                        plt.hlines(
                            y=j,
                            xmin=start_pos,
                            xmax=end_pos,
                            color=colors[curr - 1] if curr > 0 else colors[curr],
                            lw=20,
                            label="bus " + str(curr),
                        )
                        curr = topo[:, j][i]
                        start_pos = end_pos
                    end_pos += 1
                plt.hlines(
                    y=j,
                    xmin=start_pos,
                    xmax=end_pos,
                    color=colors[curr - 1] if curr > 0 else colors[curr],
                    lw=20,
                    label="bus " + str(curr),
                )

            ax.set_ylabel("element", fontsize=18)
            ax.set_xlabel("time step", fontsize=18)
            plt.xticks(np.arange(864, step=100), fontsize=16)
            plt.yticks(np.arange(topo.shape[1]), fontsize=16)
            handles, labels = ax.get_legend_handles_labels()
            indexes = [labels.index(x) for x in set(labels)]
            ax.legend(
                [handle for i, handle in enumerate(handles) if i in indexes],
                [label for i, label in enumerate(labels) if i in indexes],
                fontsize=22,
                loc="lower right",
            )
            plt.savefig(os.path.join(log_folder, f"actions_%s.png" % test_case))
            plt.show()
    else:
        print(f"The folder %s does not contain any Ch*.npy files." % folder)


def plot_safe_gantt_np(folder, plot_title="Env check is safe "):
    files = [file for file in os.listdir(folder) if file.endswith("safe.npy") & file.startswith("Ch")]
    log_folder = os.path.join(folder, "eval_safe")
    os.makedirs(log_folder, exist_ok=True)
    if files:
        fig = plt.figure(figsize=(20, len(files) / 2))
        fig.suptitle(plot_title)
        ax = fig.add_subplot(1, 1, 1)
        colors = ["tab:red", "limegreen"]
        y_ticks = []
        for i, file in enumerate(files):
            is_safe = np.load(os.path.join(folder, file))
            y_ticks.append("_".join(file.split("_", 2)[:2]))
            # is_safe = df["safe"]
            curr = True
            start_pos = 0
            end_pos = 0
            for safe in is_safe:
                if safe != curr:
                    plt.hlines(y=i, xmin=start_pos, xmax=end_pos, color=colors[curr], lw=20, label=str(curr))
                    curr = safe
                    start_pos = end_pos
                end_pos += 1
            plt.hlines(y=i, xmin=start_pos, xmax=end_pos, color=colors[curr], lw=20, label=str(curr))

        ax.set_ylabel("Epochs")
        ax.set_xlabel("Timesteps")
        plt.xticks(np.arange(864, step=100))
        plt.yticks(np.arange(len(y_ticks)), y_ticks)
        handles, labels = ax.get_legend_handles_labels()
        indexes = [labels.index(x) for x in set(labels)]
        ax.legend(
            [handle for i, handle in enumerate(handles) if i in indexes],
            [label for i, label in enumerate(labels) if i in indexes],
            loc="best",
        )
        plt.savefig(os.path.join(log_folder, f"safe.png"))
        plt.show()
    else:
        print(f"The folder %s does not contain any Ch*topoAnalytics.csv files." % folder)


def plot_safe_gantt(folder):
    files = [file for file in os.listdir(folder) if file.endswith("topoAnalytics.csv") & file.startswith("Ch")]
    log_folder = os.path.join(folder, "eval_safe")
    os.makedirs(log_folder, exist_ok=True)
    if files:
        fig = plt.figure(figsize=(20, len(files) / 2))
        fig.suptitle("Env check is safe ")
        ax = fig.add_subplot(1, 1, 1)
        colors = ["tab:red", "limegreen"]
        y_ticks = []
        for i, file in enumerate(files):
            df = pd.read_csv(os.path.join(folder, file))
            y_ticks.append("_".join(file.split("_", 2)[:2]))
            is_safe = df["safe"]
            curr = True
            start_pos = 0
            end_pos = 0
            for safe in is_safe:
                if safe != curr:
                    plt.hlines(y=i, xmin=start_pos, xmax=end_pos, color=colors[curr], lw=20, label=str(curr))
                    curr = safe
                    start_pos = end_pos
                end_pos += 1
            plt.hlines(y=i, xmin=start_pos, xmax=end_pos, color=colors[curr], lw=20, label=str(curr))

        ax.set_ylabel("Epochs")
        ax.set_xlabel("Timesteps")
        plt.xticks(np.arange(864, step=100))
        plt.yticks(np.arange(len(y_ticks)), y_ticks)
        handles, labels = ax.get_legend_handles_labels()
        indexes = [labels.index(x) for x in set(labels)]
        ax.legend(
            [handle for i, handle in enumerate(handles) if i in indexes],
            [label for i, label in enumerate(labels) if i in indexes],
            loc="best",
        )
        plt.savefig(os.path.join(log_folder, f"safe.png"))
        plt.show()
    else:
        print(f"The folder %s does not contain any Ch*topoAnalytics.csv files." % folder)
