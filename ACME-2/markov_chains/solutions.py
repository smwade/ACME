"""Volume II Lab 8: Markov Chains
Sean Wade
MATH 321
10/23/2015
"""

import numpy as np
import scipy as sp

def random_markov(n):
    """Create and return a transition matrix for a random
    Markov chain with 'n' states as an nxn NumPy array.
    """
    A = np.random.dirichlet(np.ones(n),size=n)
    return A.T
    

def forecast(num_days):
    """Run a simulation for the weather over 'num_days' days, with
    "hot" as the starting state. Return a list containing the day-by-day
    results, not including the starting day.

    Example:
        >>> forecast(5)
        [1, 0, 0, 1, 0]
        # Or, if you prefer,
        ['cold', 'hot', 'hot', 'cold', 'hot']
    """
    tran_matrix = np.array([[0.7, 0.6], [0.3, 0.4]])
    cur_day = 0
    forecast = []
    for x in xrange(0, num_days):
        r = np.random.random()
        if r < tran_matrix[1][cur_day]:
            forecast.append(1)
            cur_day = 1
        else:
            forecast.append(0)
            cur_day = 0
    return forecast

def four_state_forecast(days=1):
    """Same as forecast(), but using the four-state transition matrix."""
    tran_matrix = np.array([[.5, .3, .1, .0], [.3, .3, .2, .3], [.2, .3, .4, .5], [.0, .1, .3, .2]])
    cur_day = 0
    forecast = []
    for x in xrange(0, days):
        probabilities = tran_matrix[:, cur_day]
        pos = np.random.multinomial(1, probabilities).argmax()
        cur_day = pos
        forecast.append(pos)
    return forecast


def analyze_simulation():
    """Analyze the results of the previous two problems. What percentage
    of days are in each state? Print your results to the terminal.
    """
    f = forecast(1000000)
    total_days = float(len(f))
    c = f.count(0)
    h = f.count(1)
    print "The percent of days: 2 State"
    print "===================================="
    print "Hot: %s" % (h/total_days * 100)
    print "Cold: %s\n" % (c/total_days * 100)

    
    results = four_state_forecast(100000)
    total_days = float(len(results))
    hot = results.count(0)
    mild = results.count(1)
    cold = results.count(2)
    freezing = results.count(3)
    print "The percent of days: 4 State"
    print "===================================="
    print "Hot: %s" % (hot/total_days * 100)
    print "Mild: %s" % (mild/total_days * 100)
    print "Cold: %s" % (cold/total_days * 100)
    print "Freezing: %s" % (freezing/total_days * 100)

analyze_simulation()

def convert(in_file):
    lines_list = []
    word_dict = {}
    words = ['$start']
    val_int = 1
    with open(in_file, 'r') as my_file:
        for line in my_file:
            lines_list.append(line)
            for word in line.split():
                if word.strip() not in word_dict:
                    word_dict[word] = val_int
                    val_int += 1
                    words.append(word)
    
    with open('output.txt', 'w') as my_file:
        for line in lines_list:
            for word in line.split():
                my_file.write(str(word_dict[word]) + " ")
            my_file.write('\n')

    words.append('$end')
    return words

def fun(in_file, num_of_words):
    M = np.zeros((num_of_words, num_of_words))
    start_num = 0
    end_num = num_of_words - 1
    prev = start_num
    now = start_num
    with open(in_file, 'r') as my_file:
        for line in my_file:
            prev = start_num
            now = start_num
            for num in line.split():
                now = int(num.strip())
                M[now][prev] += 1
                prev = now
            M[end_num][now] += 1

    M = M / np.sum(M, axis=0)
    return M


#l = convert('in.txt')
#print fun('output.txt', len(l))

def sentences(infile, outfile, num_sentences=1):
    """Generate random sentences using the word list generated in
    Problem 5 and the transition matrix generated in Problem 6.
    Write the results to the specified outfile.

    Parameters:
        infile (str): The path to a filen containing a training set.
        outfile (str): The file to write the random sentences to.
        num_sentences (int): The number of random sentences to write.

    Returns:
        None
    """
    with open(outfile, 'w') as my_file:
        my_file.write("MY POEM\n====================================\n")
    word_list = convert(infile)
    Markov = fun('output.txt', len(word_list))
    sentance = 1
    while sentance <= num_sentences:
        result = []
        cur = 0
        while cur != len(word_list) - 1:
            probabilities = Markov[:, cur]
            pos = np.random.multinomial(1, probabilities).argmax()
            cur = pos
            result.append(word_list[pos])
        with open(outfile, 'a') as my_file:
            my_file.write(' '.join(result[:-1]) + "\n")

        sentance += 1

