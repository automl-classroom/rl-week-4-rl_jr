import json

import numpy as np
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils



# Load data

seeds = [0, 27049, 63645, 52180, 33027, 13418]
avg_rewards = []
frame_list = []

for s in seeds:
    with open(f"rl_exercises/week_4/results_level_2/seed_{s}.json", 'r') as file:
        data = json.load(file)
        avg_rewards.append(data['mean_rewards'])
        frame_list.append(data['frames'])

min_len = min(len(x) for x in avg_rewards)
avg_rewards = np.array([np.array(r[:min_len]) for r in avg_rewards])
algorithms = ['DQN']
algo_reward_dict = {'DQN': avg_rewards}



# Plot IQMs, median, mean, and optimality gap

aggregate_func = lambda x: np.array([
  metrics.aggregate_median(x),
  metrics.aggregate_iqm(x),
  metrics.aggregate_mean(x),
  metrics.aggregate_optimality_gap(x)
])
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
  algo_reward_dict, 
  aggregate_func, 
  reps=50000
)
fig, axes = plot_utils.plot_interval_estimates(
  aggregate_scores, 
  aggregate_score_cis,
  metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
  algorithms=algorithms, 
  xlabel="Mean Reward",
  xlabel_y_coordinate=-1.5
)
fig.savefig("rl_exercises/week_4/plots/level2_other_metrics.png", bbox_inches='tight')



# Plot training curve

frames = np.arange(min_len)

iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
iqm_scores, iqm_cis = rly.get_interval_estimates(
  algo_reward_dict, 
  iqm, 
  reps=50000
)
axes = plot_utils.plot_sample_efficiency_curve(
    frame_list[0][:min_len], 
    iqm_scores, 
    iqm_cis, 
    algorithms=algorithms,
    xlabel='Number of Frames',
    ylabel='Mean Reward'
)
fig = axes.get_figure()
fig.legend(labels=algorithms)
fig.savefig("rl_exercises/week_4/plots/level2_training_curve.png", bbox_inches='tight')
