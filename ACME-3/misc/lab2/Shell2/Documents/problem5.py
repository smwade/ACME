#!//anaconda/bin/python
import subprocess

class Shell(object):

    def __init__(self):
        pass

    def find_word(self, word, file_name, directory=None):
        if directory is None:
            directory = subprocess.check_output("pwd", shell=True).rstrip()
        results = subprocess.check_output("grep '{}' {}/{}".format(word, directory, file_name), shell=True)
        results.split('\n')
        results.pop()
        return results


    def find_file(self, file_name, directory=None):
        if directory is None:
            directory = subprocess.check_output("pwd", shell=True).rstip()
        
