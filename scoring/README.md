# Local Submission Scoring

The files in this repo are supposed to help you score your agents behavior locally.

**WARNING**: This is not the actual submission scoring --> Results will differ from the scores you achieve here. But the scoring setup is very similar to this setup.

**Beta Stage**: The scoring function here is still under development, use with caution.

## Introduction
This repo contains a very basic setup to test your own agent/algorithm on the Flatland scoring setup.
The repo contains 3 important files:

- `generate_tests.py` Pre-generates the test files for faster testing
- `score_tests.py` Scores your agent on the generated test files
- `show_test.py` Shows samples of the generated test files
- `parameters.txt` Parameters for generating the test files --> These differ in the challenge submission scoring

To start the scoring of your agent you need to do the following

## Parameters used for Level generation

| Test Nr.  | X-Dim  | Y-Dim  | Nr. Agents  | Random Seed  |
|:---------:|:------:|:------:|:-----------:|:------------:|
| Test 0      | 10 | 10 | 1 | 3 |
| Test 1      | 10 | 10 | 3 | 3 |
| Test 2      | 10 | 10 | 5 | 3 |
| Test 3      | 50 | 10 | 10 | 3 |
| Test 4      | 20 | 50 | 10 | 3 |
| Test 5      | 20 | 20 | 15 | 3 |
| Test 6      | 50 | 50 | 10 | 3 |
| Test 7      | 50 | 50 | 40 | 3 |
| Test 8      | 100 | 100 | 10 | 3 |
| Test 9      | 100 | 100 | 50 | 3 |

These can be changed if you like to test your agents behavior on different tests.

## Generate the test files
To generate the set of test files you just have to run `python generate_tests.py`
This generates pickle files of the levels to test on and places them in the corresponding folders.

## Run Test
To run the tests you have to modify the `score_tests.py` file to load your agent and the necessary predictor and observation.
The following lines have to be replaced by you code:

```
# Load your agent
agent = YourAgent
agent.load(Your_Checkpoint)

# Load the necessary Observation Builder and Predictor
predictor = ShortestPathPredictorForRailEnv()
observation_builder = TreeObsForRailEnv(max_depth=tree_depth, predictor=predictor)
```

The agent and the observation builder as well as an observation wrapper can be passed to the test function like this

```
test_score, test_dones, test_time = run_test(current_parameters, agent, observation_builder=your_observation_builder,
                                             observation_wrapper=your_observation_wrapper,
                                             test_nr=test_nr, nr_trials_per_test=10)
```

In order to speed up the test time you can limit the number of trials per test (`nr_trials_per_test=10`). After you have made these changes to the file you can run `python score_tests.py` which will produce an output similiar to this:

```
Running Test_0 with (x_dim,y_dim) = (10,10) and 1 Agents.
Progress: |********************| 100.0% Complete 
Test_0 score was -0.380 with 100.00% environments solved. Test took 0.62 Seconds to complete.

Running Test_1 with (x_dim,y_dim) = (10,10) and 3 Agents.
Progress: |********************| 100.0% Complete 
Test_1 score was -1.540 with 80.00% environments solved. Test took 2.67 Seconds to complete.

Running Test_2 with (x_dim,y_dim) = (10,10) and 5 Agents.
Progress: |********************| 100.0% Complete 
Test_2 score was -2.460 with 80.00% environments solved. Test took 4.48 Seconds to complete.

Running Test_3 with (x_dim,y_dim) = (50,10) and 10 Agents.
Progress: |**__________________| 10.0% Complete
```

The score is computed by

```
score = sum(mean(all_rewards))/max_steps
```
which is the sum over all time steps and the mean over all agents of the rewards. We normalize it by the maximum number of allowed steps for a level size. The max number of allowed steps is

```
max_steps = mult_factor * (env.height+env.width)
```
Where the `mult_factor` is a multiplication factor to allow for more time if difficulty is to high.

The number of solved envs is just the percentage of episodes that terminated with all agents done.

How these two numbers are used to define your final score will be posted on the [flatland page](https://www.aicrowd.com/organizers/sbb/challenges/flatland-challenge)
