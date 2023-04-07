from pathlib import Path
import shlex
import subprocess as sp

from hydra.core.hydra_config import HydraConfig


def run_shell_cmd(cmd):
    '''
    A thin wrapper around some subprocess boilerplate.

    `cmd` is a string (just as one might type into a terminal)
    '''
    p = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.STDOUT)
    stdout, stderr = p.communicate()
    return p.returncode, stdout, stderr


def get_git_commit():
    '''
    Gets the hash of the current git commit.
    '''
    git_hash_cmd = 'git show --format="%H" --no-patch'
    rc, stdout, stderr = run_shell_cmd(git_hash_cmd)
    if rc == 0:
        return stdout.decode('utf-8').strip()
    else:
        raise Exception('When requesting git commit, had a non-zero'
                        f' exit code.\nStdout:{stdout}.\nStderr:{stderr}')


def write_git_commit(strict=True, name='git_commit.txt'):
    '''
    Writes the git commit into a file in the hydra-created
    output directory.

    If `strict=True` (default), we check if the git working tree is clean and
    immediately fail if the working tree is not clean. This prevents
    executions/results using uncommitted changes unless specifically requested.
    '''
    if strict:
        git_diff_cmd = 'git diff --quiet'
        rc, stdout, stderr = run_shell_cmd(git_diff_cmd)
        if rc == 1:
            raise Exception('Working tree was not clean and strict mode was'
                            ' requested. Exiting.')

    commit_hash = get_git_commit()
    output_path = Path.cwd()/Path(HydraConfig.get().output_subdir)/Path(name)
    with output_path.open('w') as fout:
        fout.write(commit_hash)
