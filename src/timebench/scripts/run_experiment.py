"""Hydra entry point for one visible scheduler stage."""

from pathlib import Path
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path='../conf', config_name='experiment')
def main(config):
    from timebench.pipeline.workflow import Workflow

    resolved = OmegaConf.to_container(config, resolve=True)
    if resolved['dataset_config'] is None:
        resolved['dataset_config'] = str(Path(__file__).parents[1] / 'config/datasets.yaml')
    Workflow(resolved).run(resolved['stage'])


if __name__ == '__main__':
    main()
