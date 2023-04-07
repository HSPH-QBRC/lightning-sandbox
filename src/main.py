import sys
import logging

import hydra

from utils import write_git_commit

logger = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path="../conf", config_name="config")
def main(cfg):

    # Before anything else, check that we are permitting execution to proceed
    # if there are uncommited changes.
    try:
        if not cfg.strict_git_clean:
            logger.warning('Running in non-strict mode. This permits'
                           ' executions with a dirty git working tree')
        write_git_commit(strict=cfg.strict_git_clean)
    except Exception as ex:
        sys.stderr.write(f'{ex}\n')
        sys.exit(1)


if __name__ == "__main__":
    main()
