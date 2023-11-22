from typing import Any, Dict
import optuna


def sample_params(trial: optuna.Trial, agent_name) -> Dict[str, Any]:
    if agent_name == 'ppo':
        sampled_hyperparams = sample_ppo_params(trial)
    elif agent_name == 'dppo':
        sampled_hyperparams = sample_dppo_params(trial)
    elif agent_name == 'dsacd':
        sampled_hyperparams = sample_dsacd_params(trial)
    elif 'sac' in agent_name:
        sampled_hyperparams = sample_sac_params(trial)
    else:
        raise Exception(f"No sample parameters is implemented for agent of type {agent_name}.")
    return sampled_hyperparams


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
        used as reference: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128])
    update_start = trial.suggest_int("update_start", low=1, high=8, step=1)
    # discount factor between 0.9 and 0.999
    gamma = 1.0 - trial.suggest_float("gamma", 0.001, 0.1, log=True)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    entropy = trial.suggest_float("entropy", 1e-9, 0.1, log=True)
    epsilon = trial.suggest_float("epsilon", 0.1, 0.3)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)

    # Display true values
    trial.set_user_attr("gamma_", gamma)

    return {
        "update_start": update_start,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "entropy": entropy,
        "epsilon": epsilon,
        "gae_lambda": gae_lambda,
    }


def sample_dppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DPPO hyperparams.
        Used as a referenc: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    update_start = trial.suggest_int("update_start", low=1, high=6, step=1)
    # discount factor between 0.9 and 0.999
    gamma = 1.0 - trial.suggest_float("gamma", 0.001, 0.01, log=True)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    entropy = trial.suggest_float("entropy", 1e-6, 1e-3, log=True)
    epsilon = trial.suggest_float("epsilon", 0.05, 0.15)
    gae_lambda = trial.suggest_float("gae_lambda", 0.825, 0.925)

    # Display true values
    trial.set_user_attr("gamma_", gamma)

    return {
        "update_start": update_start,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "entropy": entropy,
        "epsilon": epsilon,
        "gae_lambda": gae_lambda,
    }


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC(D) hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    update_start = trial.suggest_int("update_start", low=2, high=8, step=2)
    # discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.001, 0.1, log=True)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    target_update = trial.suggest_int("target_update", low=1, high=2, step=1)
    target_entropy_scale = trial.suggest_float("target_entropy_scale", 0.95, 0.99)
    tau = trial.suggest_float("tau", 1e-3, 1e-2)
    update_freq = trial.suggest_int("update_freq", low=1, high=5, step=1)

    # Display true values
    trial.set_user_attr("gamma_", gamma)

    return {
        "update_start": update_start,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "target_update": target_update,
        "target_entropy_scale": target_entropy_scale,
        "tau": tau,
        "update_freq": update_freq
    }


def sample_dsacd_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SACD hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    update_start = trial.suggest_int("update_start", low=1, high=4, step=1)
    # discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.001, 0.1, log=True)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    target_update = trial.suggest_int("target_update", low=1, high=2, step=1)
    target_entropy_scale = trial.suggest_float("target_entropy_scale", 0.95, 0.99)
    tau = trial.suggest_float("tau", 1e-3, 1e-2)
    update_freq = trial.suggest_int("update_freq", low=1, high=5, step=1)

    # Display true values
    trial.set_user_attr("gamma_", gamma)

    return {
        "update_start": update_start,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "target_update": target_update,
        "target_entropy_scale": target_entropy_scale,
        "tau": tau,
        "update_freq": update_freq
    }