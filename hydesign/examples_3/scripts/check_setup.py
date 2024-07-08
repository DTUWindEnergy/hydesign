# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:21:46 2023

@author: mikf
"""
import os
import sys
import subprocess
import hydesign

def _run_git_cmd(cmd, git_repo_path=None):
    git_repo_path = git_repo_path or os.getcwd()
    if not os.path.isdir(os.path.join(git_repo_path, ".git")):
        raise Warning("'%s' does not appear to be a Git repository." % git_repo_path)
    try:
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True,
                                   cwd=os.path.abspath(git_repo_path))
        stdout,stderr = process.communicate()
        if process.returncode != 0:
            raise EnvironmentError("%s\n%s"%(stdout, stderr))
        return stdout.strip()

    except EnvironmentError as e:
        raise e
        raise Warning("unable to run git")

def get_git_version(git_repo_path=None):
    # cmd = ["git", "describe", "--always"]
    cmd = ['git', 'rev-parse', 'HEAD']
    return _run_git_cmd(cmd, git_repo_path)

hydesign_package_path = os.path.dirname(os.path.dirname(hydesign.__file__))
print('Hydesign package path: ', hydesign_package_path)
print('Python path: ', sys.executable)
print('Git commit SHA: ', get_git_version(hydesign_package_path))
output = subprocess.check_output('conda info', shell=True)
output = output.decode("utf-8")
print(output)
output = subprocess.check_output('conda list hydesign', shell=True)
output = output.decode("utf-8")
print(output)
output = subprocess.check_output('pip show hydesign', shell=True)
output = output.decode("utf-8")
print(output)
