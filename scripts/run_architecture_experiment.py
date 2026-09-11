"""Run the chronological state-space architecture experiment.

The earlier marginal-spread comparison is preserved in
archive/original/ridge_architecture_experiment.py. Its future-trained variance
budget is not valid chronological evidence and cannot authorize adoption.
"""

from .run_state_space_experiment import main

if __name__ == "__main__":
    main()
