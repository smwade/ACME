import subprocess

class Shell(object):

    def find_file(self, filename, directory=None):
        """
        Find files in a given directory

        Inputs:
            filename -- Name of file to search for
            dir      -- Directory in which to search for file
                        Defaults to current directory

        Returns:
            files -- a list of files matching 'filename'
        """
        if directory is None:
            directory = subprocess.check_output("pwd", shell=True)
        directory = directory.rstrip()
        try:
            results = subprocess.check_output('find {} -type f -name "{}"'.format(directory, filename), shell=True)
        except:
            return []
        results = results.split('\n')
        results.pop()
        return results

    def find_word(self, word, directory=None):
        """
        Find files containing a certain string in a given directory

        Inputs:
            word -- String to find in given directory
            directory  -- Directory in which to search for the string
                    Defaults to current directory

        Returns:
            files -- a list of files containing the string
        """
        if directory is None:
            directory = subprocess.check_output("pwd", shell=True)
        directory = directory.rstrip()
        try:
            results = subprocess.check_output('grep -s -l "{}" {}'.format(word, directory), shell=True)
        except:
            return []
            print "The file dosnt have it, or it is a bad file..."
        results = results.split('\n')
        results.pop()
        return results

    def find_n_largest_files(self, n_files, directory=None):
        """
        Recursively find largest file in a given directory

        Inputs:
            n_files -- how many files to return
            directory     -- Directory in which to search for the string
                        Defaults to current directory

        Returns:
            file -- Largest file in given directory
        """
        if directory is None:
            directory = subprocess.check_output("pwd", shell=True)
        directory = directory.rstrip()
        results = subprocess.check_output('ls -SR {}'.format(directory), shell=True)
        results = results.split('\n')
        #results = results.pop()
        print results 
        print "-"*30
        if n_files < 0:
            return []
        if n_files > len(results):
            return results
        return results[:n_files]

#if __name__ == '__main__':
#    s = Shell()
#    print s.find_n_largest_files(2)
