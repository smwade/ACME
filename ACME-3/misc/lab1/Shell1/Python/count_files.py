import subprocess

shell_dir = subprocess.check_output("find ~ -type d -name 'Shell2'", shell=True).split('\n')
shell_dir.pop()
if len(shell_dir) != 1:
    raise Exception("Ensure there is exactly one instance of the Shell-Lab directory on your computer")
shell_dir = shell_dir[0]
subprocess.call("find {} -type f | wc -l".format(shell_dir), shell=True)
