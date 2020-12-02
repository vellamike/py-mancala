import argparse
parser = argparse.ArgumentParser(description='Mancala AI')
parser.add_argument('-b', '--board', default=None)
args = parser.parse_args()
print(type(args.board))
