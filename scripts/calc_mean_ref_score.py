import json
import os
import fnmatch

def calc_mean_ref_score(json_path: str) -> float:
    with open(json_path, 'r') as f:
        data = json.load(f)
    task_results = data['task_results']
    ref_scores = [v['ref_score'] for v in task_results.values()]
    mean_ref_score = sum(ref_scores) / len(ref_scores) if ref_scores else float('nan')
    return mean_ref_score

def find_dadp_json_files(root_dir: str):
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, 'dadp.json'):
            matches.append(os.path.join(root, filename))
    return matches

def main():
    data_dir = '/home/qinghang/DomainAdaptiveDiffusionPolicy/data'
    json_files = find_dadp_json_files(data_dir)
    if not json_files:
        print(f"No dadp.json files found in {data_dir}")
        return
    for path in json_files:
        mean_ref = calc_mean_ref_score(path)
        print(f"{path}: mean ref_score = {mean_ref:.2f}")

if __name__ == '__main__':
    main()
