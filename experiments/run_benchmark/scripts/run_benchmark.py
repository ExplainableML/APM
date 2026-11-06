import hydra
from omegaconf import DictConfig

from llm_personalization.personalization_system.benchmark import run_benchmark


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
