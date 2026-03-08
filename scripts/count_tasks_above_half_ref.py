import json
import argparse

def count_tasks_above_half_ref(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    task_results = data['task_results']
    count = 0
    total = 0
    for v in task_results.values():
        total += 1
        if v['mean_reward'] >= 0.6 * v['ref_score']:
            count += 1
    print(f"{count} out of {total} tasks have mean_reward >= 50% of ref_score.")
    return count, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default = "results/exp_halfcheetah_28_condition_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/task_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]_less_sample_step(5).json", help='Path to the result JSON file')
    args = parser.parse_args()
    count_tasks_above_half_ref(args.json_path)

if __name__ == '__main__':
    main()
