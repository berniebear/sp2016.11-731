
import argparse # optparse is deprecated


parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('input', type=str)
opts = parser.parse_args()


with open(opts.input) as lines:
    for line in lines:
        line = line.strip().split(',')
        h1_score = line[0]
        h2_score = line[1]
        print(-1 if h1_score > h2_score else # \begin{cases}
                (0 if h1_score == h2_score
                    else 1)) # \end{cases}
