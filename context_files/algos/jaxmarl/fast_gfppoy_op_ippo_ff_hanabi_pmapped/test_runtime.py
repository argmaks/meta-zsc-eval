import time
import os

def test_runtime(func, config, expected_runtime_seconds):

    start_time = time.time()
    func(config)
    end_time = time.time()
    runtime = end_time - start_time
    assert runtime < expected_runtime_seconds, f"Runtime {runtime} seconds is greater than expected {expected_runtime_seconds} seconds"
    print(f"Runtime {runtime} seconds is less than expected {expected_runtime_seconds} seconds")


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from decpomdp_symmetries_op_ippo_ff_hanabi import main

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = OmegaConf.load(config_path)

    algo_config = config['runner']['algo_config']
    expected_runtime_seconds = config['runner']['tests']['test_runtime']['expected_runtime_seconds']

    # test overrides
    algo_config["NUM_SEEDS"] = 2
    algo_config["SAVE_MODELS"] = True
    algo_config["TOTAL_TIMESTEPS"] = 1e5
    algo_config["WANDB_MODE"] = "disabled"
    algo_config["CHECKPOINT_TIMESTEPS"] = None

    test_runtime(main, algo_config, expected_runtime_seconds)