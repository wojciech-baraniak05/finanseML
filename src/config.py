"""Configuration constants for the credit scorecard pipeline."""

TARGET_PD = 0.04
BASE_SCORE = 600
PDO = 20
TARGET_ODDS = (1 - TARGET_PD) / TARGET_PD

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

CORRELATION_THRESHOLD = 0.8
VIF_THRESHOLD = 10.0
N_BINS_DEFAULT = 10
MIN_IV_THRESHOLD = 0.02

WINSORIZATION_LOWER = 0.01
WINSORIZATION_UPPER = 0.99

RATING_BOUNDARIES = {
    'AAA': (0.0, 0.001),
    'AA': (0.001, 0.005),
    'A': (0.005, 0.01),
    'BBB': (0.01, 0.03),
    'BB': (0.03, 0.07),
    'B': (0.07, 0.15),
    'CCC': (0.15, 0.30),
    'CC': (0.30, 0.50),
    'C': (0.50, 0.80),
    'D': (0.80, 1.0)
}

CALIBRATION_N_BINS = 10

CV_FOLDS = 3
BAYESIAN_N_ITER = 15
