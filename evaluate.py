"""Evaluate the sole MAHDRL algorithm."""
from mahdrl.scenario import parse_scenario
from mahdrl.runtime import evaluate

if __name__ == '__main__':
    evaluate(parse_scenario('eval'))
