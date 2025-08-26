# run_parallel_evaluation.py

# Heuristic (CQL-H)
# python -m src.evaluation.run_parallel_evaluation --model_path models/CQL_H_debug_run_20250821_090715/step_500000_actor.pth --run_name CQL_H_final_quantitative_eval --num_scenarios 200

# Heuristic-Scenario (CQL-HS)
# python -m src.evaluation.run_parallel_evaluation --model_path models/CQL_HS_scenario_run_20250819_123138/step_500000_actor.pth --run_name CQL_HS_final_quantitative_eval --num_scenarios 200

# Ensemble-Scenario (CQL-ES)
# python -m src.evaluation.run_parallel_evaluation --model_path models/CQL_ES_scenario_run_20250820_122732/step_500000_actor.pth --run_name CQL_ES_final_quantitative_eval --num_scenarios 200

# Action-Rarity-Scenario (CQL-ARS)
# python -m src.evaluation.run_parallel_evaluation --model_path models/CQL_ARS_scenario_run_20250820_203347/step_500000_actor.pth --run_name CQL_ARS_final_quantitative_eval --num_scenarios 200

# Baseline with BC Auxiliary Loss (CQL-B-BC)
# python -m src.evaluation.run_parallel_evaluation --model_path models/CQL_B_BCLoss_run1_20250821_170935/step_500000_actor.pth --run_name CQL_B_BCLoss_final_quantitative_eval --num_scenarios 200

# Baseline without BC Auxiliary Loss (CQL-B-noBC)
# python -m src.evaluation.run_parallel_evaluation --model_path models/CQL_B_noloss_run1_20250822_010004/step_500000_actor.pth --run_name CQL_B_noloss_final_quantitative_eval --num_scenarios 200
  

import os
import sys
import subprocess
import multiprocessing
from glob import glob
import random
import json
from tqdm import tqdm
from typing import List
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# Add src to path to import config loader
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils.config import load_config

def run_single_evaluation(args_tuple):
    """
    Worker function that takes a tuple of arguments and executes the
    evaluate_cql.py script as a subprocess.
    """
    model_path, scenario_id, run_name, metrics_dir, save_video_str = args_tuple
    
        # Define the output path for the per-scenario JSON metrics
    metrics_file_path = os.path.join(metrics_dir, f"{scenario_id}.json")
    
    # Construct the full command
    cmd = [
        "python", "-m", "src.evaluation.evaluate_cql",
        "--model_path", model_path,
        "--scenario_id", scenario_id,
        "--run_name", run_name,
        "--metrics_file", metrics_file_path, # Tell the script where to save its result
        "--quiet"
    ]
    
    if save_video_str == 'true':
        cmd.append("--save_video")

    # Execute the command
    try:
        # Execute the command. The script will write its own JSON file.
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        
        # --- THIS IS THE CRITICAL FIX ---
        # After the script succeeds, the worker reads the JSON file it created
        # and returns the contents.
        if os.path.exists(metrics_file_path):
            with open(metrics_file_path, 'r') as f:
                # The file contains the full summary. We only need the per-scenario detail.
                result_data = json.load(f)['details_per_scenario'][0]
            return scenario_id, True, result_data
        else:
            return scenario_id, False, {"error": "Metrics file not found after successful run."}
        # --- END OF FIX ---
    except subprocess.CalledProcessError as e:
        # If the script fails, return the error message
        error_message = f"--- FAILED for scenario {scenario_id} ---\n"
        error_message += f"STDOUT:\n{e.stdout}\n"
        error_message += f"STDERR:\n{e.stderr}\n"
        return scenario_id, False, error_message
    except subprocess.TimeoutExpired:
        return scenario_id, False, f"--- TIMEOUT for scenario {scenario_id} ---"

def main():
    # --- 1. Argument Parsing ---
    import argparse
    parser = argparse.ArgumentParser(
        description="Run parallel evaluation for a trained CQL agent using Waymax.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model_path", type=str, required=True, 
                        help="Path to the trained actor.pth model checkpoint.")
    parser.add_argument("-r", "--run_name", type=str, required=True, 
                        help="A unique name for this evaluation batch run. Outputs will be saved here.")
    parser.add_argument("-n", "--num_scenarios", type=int, default=100, 
                        help="Number of random scenarios to evaluate from the validation set.")
    parser.add_argument("-v", "--save_videos", action='store_true', 
                        help="Save a video for each rollout. WARNING: This will be slow and consume significant disk space.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for selecting scenarios.")
    args = parser.parse_args()

    config = load_config()
    
    # --- 2. Setup Output Directories ---
    output_dir = os.path.join('outputs', args.run_name)
    metrics_dir = os.path.join(output_dir, 'metrics_per_scenario') # More descriptive name
    os.makedirs(metrics_dir, exist_ok=True)
    
    # --- 3. Select Scenarios ---
    raw_data_dir = os.path.join(config['data']['processed_npz_dir'], 'validation')
    all_scenarios = glob(os.path.join(raw_data_dir, '*.npz'))
    if not all_scenarios:
        print(f"❌ Error: No scenario .npz files found in {raw_data_dir}")
        return

    random.seed(args.seed)
    scenarios_to_run = random.sample(all_scenarios, min(args.num_scenarios, len(all_scenarios)))
    scenario_ids = [os.path.basename(p).split('.')[0] for p in scenarios_to_run]
    
    print(f"Prepared {len(scenario_ids)} scenarios for parallel evaluation.")
    
    # --- 4. Create Task List for Workers ---
    save_video_str = 'true' if args.save_videos else 'false'
    tasks = [(args.model_path, sid, args.run_name, metrics_dir, save_video_str) for sid in scenario_ids]
    
    # --- 5. Run in Parallel ---
    # Use half the cores to leave resources for the system and JAX/PyTorch overhead.
    num_workers = 10 
    print(f"Starting parallel evaluation with {num_workers} workers...")
    
    all_results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(run_single_evaluation, tasks), total=len(tasks)):
            all_results.append(result)

    # --- 6. Aggregate and Summarize ---
    print("\n" + "="*50)
    print("---      Parallel Evaluation Complete: Aggregating Results      ---")
    print("="*50)
    
    successes = [res for res in all_results if res[1]]
    failures = [res for res in all_results if not res[1]]
    
    print(f"  Total runs attempted: {len(all_results)}")
    print(f"  ✅ Successful: {len(successes)}")
    print(f"  ❌ Failed: {len(failures)}")
    
    if failures:
        print("\n--- Failure Details ---")
        for sid, _, error_dict in failures:
            print(f"Scenario {sid}: {error_dict}")
            if 'stderr' in error_dict:
                # Print the last few lines of the error log, which are often the most relevant
                error_lines = error_dict['stderr'].strip().split('\n')
                for line in error_lines[-5:]:
                    print(f"  -> {line}")
    
    # The metrics are already in memory, loaded by the workers.
    all_metrics_summary = [metrics_dict for _, _, metrics_dict in successes]
    
    if not all_metrics_summary:
        print("\nNo successful runs to aggregate. Final metrics file will not be generated.")
        return

    # --- 7. Final Metrics Calculation (copied from evaluate_cql.py) ---
    num_evaluated = len(all_metrics_summary)
    
    total_collisions = int(np.sum([m['collided'] for m in all_metrics_summary]))
    total_offroad = int(np.sum([m['offroad'] for m in all_metrics_summary]))
    total_red_light_violations = int(np.sum([m['red_light_violation'] for m in all_metrics_summary]))
    
    collision_rate = 100 * total_collisions / num_evaluated
    offroad_rate = 100 * total_offroad / num_evaluated
    red_light_violation_rate = 100 * total_red_light_violations / num_evaluated
    
    valid_ttc_runs = [m['min_ttc'] for m in all_metrics_summary if m['min_ttc'] < 999.0]
    avg_min_ttc = float(np.mean(valid_ttc_runs)) if valid_ttc_runs else 999.0

    successful_runs = [m for m in all_metrics_summary if not m['collided'] and not m['offroad']]
    num_successful = len(successful_runs)

    if num_successful > 0:
        avg_progression = float(np.mean([m['sdc_progression'] for m in successful_runs]))
        avg_final_dist_to_goal = float(np.mean([m['final_distance_to_goal'] for m in successful_runs]))
        avg_route_adherence = float(np.mean([m['route_adherence_error'] for m in successful_runs]))
        avg_max_jerk = float(np.mean([m['max_jerk'] for m in successful_runs]))
        avg_max_lat_accel = float(np.mean([m['max_lateral_acceleration'] for m in successful_runs]))
    else:
        avg_progression, avg_final_dist_to_goal, avg_route_adherence = 0.0, 999.0, 999.0
        avg_max_jerk, avg_max_lat_accel = 999.0, 999.0

    summary = {
        'run_name': args.run_name,
        'model_path': args.model_path,
        'num_scenarios_evaluated': num_evaluated,
        'metrics_summary': {
            'safety': {'collision_rate_percent': collision_rate, 'offroad_rate_percent': offroad_rate, 'average_minimum_ttc_seconds': avg_min_ttc, 'total_collisions': total_collisions, 'total_offroad': total_offroad},
            'goal_achievement': {'success_rate_percent': 100 * num_successful / num_evaluated, 'num_successful_runs': num_successful, 'average_progression_on_success_meters': avg_progression, 'average_final_distance_to_goal_meters': avg_final_dist_to_goal, 'average_route_adherence_error_meters': avg_route_adherence},
            'comfort': {'average_max_jerk_on_success': avg_max_jerk, 'average_max_lateral_acceleration_on_success': avg_max_lat_accel},
            'rule_compliance': {'red_light_violation_rate_percent': red_light_violation_rate, 'total_red_light_violations': total_red_light_violations}
        },
        'details_per_scenario': all_metrics_summary
    }
    
    # Print the final summary to the console
    print("\n--- FINAL AGGREGATE METRICS ---")
    print(json.dumps(summary['metrics_summary'], indent=4))
    
    # --- 8. Save Final Aggregated JSON ---
    final_summary_path = os.path.join(output_dir, "aggregated_metrics_summary.json")
    with open(final_summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n✅ Aggregated metrics for {num_evaluated} scenarios saved to {final_summary_path}")

if __name__ == '__main__':
    main()