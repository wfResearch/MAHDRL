"""Train the sole MAHDRL algorithm."""
from mahdrl.scenario import parse_scenario
from mahdrl.runtime import train

if __name__ == '__main__':
    train(parse_scenario('train'))
