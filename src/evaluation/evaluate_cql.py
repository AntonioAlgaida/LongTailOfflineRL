# src/evaluation/evaluate_cql.py

# Evaluates a trained CQL agent in a closed-loop Waymax simulation.
#
# Example usage:
# conda activate longtail-rl
# python -m src.evaluation.evaluate_cql \
#   --model_path models/CQL_H_run1_20250815_103000/final_actor.pth \
#   --scenario_id 369a13f0908705a7 \
#   --run_name CQL_H_final_eval

# H scenario:
# python -m src.evaluation.evaluate_cql --model_path models/CQL_H_debug_run_20250821_090715/step_500000_actor.pth --run_name CQL_H_final_eval -n 10 -v

# HS scenario:
# python -m src.evaluation.evaluate_cql --model_path models/CQL_HS_scenario_run_20250819_123138/step_500000_actor.pth --run_name CQL_HS_final_eval -n 10 -v 

# ES scenario:
# python -m src.evaluation.evaluate_cql --model_path models/CQL_ES_scenario_run_20250820_122732/step_500000_actor.pth --run_name CQL_ES_final_eval -n 10 -v

# ARS scenario:
# python -m src.evaluation.evaluate_cql --model_path models/CQL_ARS_scenario_run_20250820_203347/step_500000_actor.pth --run_name CQL_ARS_final_eval -n 10 -v

# Baseline scenario with BC loss:
# python -m src.evaluation.evaluate_cql --model_path models/CQL_B_BCLoss_run1_20250821_170935/step_500000_actor.pth --run_name CQL_B_BCLoss_final_eval -n 10 -v

# Baseline without BC loss:
# python -m src.evaluation.evaluate_cql --model_path models/CQL_B_noloss_run1_20250822_010004/step_500000_actor.pth --run_name CQL_B_noloss_final_eval -n 10 -v

import os
import sys
import torch
import numpy as np
import jax
from jax import numpy as jnp
from tqdm import tqdm
import argparse
import mediapy
from glob import glob
import random
import json
from typing import Dict, Tuple
import traceback

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.rl.networks import Actor, StateEncoder
from src.rl.feature_extractor import FeatureExtractor
from src.evaluation.utils import construct_state_from_npz
# from src.evaluation.custom_metrics import check_collisions, check_offroad
from src.evaluation.custom_visualization import custom_plot_simulator_state
from src.evaluation.custom_metrics import CustomMetricsCalculator

# Import Waymax components
from waymax import datatypes
# from waymax import visualization
from waymax import env as _env
from waymax import dynamics
from waymax import config as waymax_config

def extract_metrics_from_dict(metrics_dict: dict) -> dict:
    """Extracts and cleans the final metrics from Waymax."""
    is_collided = bool(jnp.max(metrics_dict['overlap'].value) > 0)
    is_offroad = bool(jnp.any(metrics_dict['offroad'].value))
    # SDC progression is the final value for the SDC (index 0)
    # sdc_progression = float(metrics_dict['sdc_progression'].value[0, -1])
    
    return {
        'collided': is_collided,
        'offroad': is_offroad,
        'sdc_progression': 0,
    }

def main():
    # --- 1. Argument Parsing and Configuration ---
    parser = argparse.ArgumentParser(description="Evaluate a trained CQL agent in Waymax.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained actor.pth model checkpoint.")
    parser.add_argument("-s", "--scenario_id", type=str, default=None, help="Specific scenario ID to evaluate. If None, a random one is chosen.")
    parser.add_argument("-r", "--run_name", type=str, required=True, help="A unique name for this evaluation run.")
    parser.add_argument("-n", "--num_scenarios", type=int, default=1, help="Number of scenarios to evaluate. Default is 1.")
    parser.add_argument("-v", "--save_video", action='store_true', help="Whether to save the rollout video.")
    parser.add_argument("-f", "--metrics_file", type=str, default=None, help="Path to save the final metrics JSON file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--quiet", action='store_true', help="Suppress tqdm progress bars for batch evaluation.")

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- 1. One-Time Setup ---
    print("--- Performing One-Time Setup ---")
    config = load_config()
    device = torch.device("cpu")
    
    print(f"Loading trained actor model from: {args.model_path}")
    actor = Actor(config).to(device)
    actor.load_state_dict(torch.load(args.model_path, map_location=device))
    actor.eval()
    
    features_cfg = config['features']
    num_goal_points = features_cfg.get('num_goal_points', 5)
    horizon_seconds = np.arange(1, num_goal_points + 1)
    horizon_steps = (horizon_seconds * 10).astype(int)

    feature_extractor = FeatureExtractor(config)
    # --- NEW: Load structured normalization stats here ---
    print("Loading structured normalization statistics...")
    stats_path = config['data']['feature_stats_path_v2']
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Feature statistics file not found at: {stats_path}")
    stats = np.load(stats_path)
    state_mean = {k.replace('_mean',''): torch.from_numpy(v).to(device).float() for k,v in stats.items() if '_mean' in k}
    state_std = {k.replace('_std',''): torch.from_numpy(v).to(device).float() for k,v in stats.items() if '_std' in k}
    
    # --- NEW: Define the normalization function locally ---
    def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Applies normalization to each tensor in the state dictionary."""
        norm_dict = {}
        for key, tensor in state_dict.items():
            if key.endswith('_mask'):
                norm_dict[key] = tensor
            else:
                norm_tensor = (tensor - state_mean[key]) / (state_std[key] + 1e-6)
                norm_dict[key] = torch.clamp(norm_tensor, -10.0, 10.0)
        return norm_dict


    # --- 2. Select Scenarios to Evaluate ---
    raw_data_dir = os.path.join(config['data']['processed_npz_dir'], 'validation')
    if args.scenario_id:
        scenario_paths = [os.path.join(raw_data_dir, f"{args.scenario_id}.npz")]
    else:
        all_scenarios = glob(os.path.join(raw_data_dir, '*.npz'))
        scenario_paths = random.sample(all_scenarios, min(args.num_scenarios, len(all_scenarios)))
    
    print(f"\nSelected {len(scenario_paths)} scenarios for evaluation.")

    # --- 3. Main Evaluation Loop ---
    all_metrics_summary = []
    dynamics_model = dynamics.InvertibleBicycleModel()
        
    for scenario_path in (tqdm(scenario_paths, desc="Evaluating Scenarios") if not args.quiet else scenario_paths):
        scenario_id = os.path.basename(scenario_path).split('.')[0]
        data = np.load(scenario_path, allow_pickle=True)
        initial_state = construct_state_from_npz(data, config)
                
        sdc_route_global = data['sdc_route']
        final_destination = sdc_route_global[-1, :2]

        env_config = waymax_config.EnvironmentConfig(
            max_num_objects=initial_state.num_objects,
            controlled_object=waymax_config.ObjectType.SDC,
            metrics=waymax_config.MetricsConfig(),
            rewards=waymax_config.LinearCombinationRewardConfig(rewards={}),
        )
        env = _env.MultiAgentEnvironment(
            dynamics_model=dynamics_model,
            config=env_config,
        )
        
        # --- Simulation Sub-Loop ---
        metrics_calculator = CustomMetricsCalculator(initial_state, sdc_route_global)

        current_state = env.reset(initial_state)
        rollout_states = [current_state]
        sdc_index = jnp.argmax(current_state.object_metadata.is_sdc).item()
        jit_step = jax.jit(env.step)

        
        imgs = []
        if args.save_video:
            current_timestep = current_state.timestep.item()
            start_idx = current_timestep + 1
            end_idx = min(start_idx + 50, sdc_route_global.shape[0])
            
            immediate_path_for_viz = sdc_route_global[start_idx:end_idx, :2]
            
            
            imgs.append(custom_plot_simulator_state(current_state, action=np.array([0.0, 0.0]),
                                                    final_goal_point=final_destination, 
                                                    immediate_path_points=immediate_path_for_viz))  # Initial state without action

        print(f"Evaluating scenario {scenario_id} with SDC index {sdc_index}...")

        sim_range = range(current_state.remaining_timesteps)
        sim_iterator = (
            tqdm(sim_range, desc=f"Simulating {scenario_id}", leave=False) if not args.quiet
            else sim_range
        )
        for _ in sim_iterator:
            # a. Featurize
            state_dict_np = feature_extractor.extract_features_from_waymax(
                current_state, sdc_route_global
            )
            state_dict_torch = {
                k: torch.from_numpy(v).unsqueeze(0).to(device).float() 
                for k, v in state_dict_np.items()
            }

            # b. Normalize the state dictionary
            normalized_state_dict = normalize_state_dict(state_dict_torch)
            
            # c. Infer Action
            with torch.no_grad():
                action_kinematic = actor(normalized_state_dict).squeeze(0).cpu().detach().numpy()
                
                # Replace the actions with a random actions for testing purposes
                # action_kinematic = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
                # print(f"Action: {action_kinematic}")

            # c. Manually Create the Action PyTree
            all_agent_actions = jnp.zeros((current_state.num_objects, 2))
            final_action_data = all_agent_actions.at[sdc_index].set(jnp.array(action_kinematic))
            action_valid_mask = jnp.zeros((current_state.num_objects, 1), dtype=bool).at[sdc_index].set(True)
            waymax_action = datatypes.Action(data=final_action_data, valid=action_valid_mask)
            
            # d. Step the environment
            current_state = jit_step(current_state, waymax_action)
            rollout_states.append(current_state)
            
            # ts = current_state.timestep.item()
            metrics_calculator.step(current_state)
            
                 
            if args.save_video:
                # Let's show the next 2 seconds (20 points)
                current_timestep = current_state.timestep.item()
                start_idx = current_timestep + 1
                end_idx = min(start_idx + 50, sdc_route_global.shape[0])
                
                immediate_path_for_viz = sdc_route_global[start_idx:end_idx, :2]
                
                imgs.append(custom_plot_simulator_state(current_state, action=action_kinematic,
                                                    final_goal_point=final_destination, 
                                                    immediate_path_points=immediate_path_for_viz))  # Initial state without action


        # --- Finalize metrics after the loop ---
        scenario_metrics = metrics_calculator.finalize(current_state)
        scenario_metrics['scenario_id'] = scenario_id
        all_metrics_summary.append(scenario_metrics)
        print("...Simulation complete.")

        # --- Visualization ---
        if args.save_video:
            print("Generating rollout video from captured frames...")
            output_dir = os.path.join('outputs', args.run_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{scenario_id}_rollout.mp4")
            
            mediapy.write_video(output_path, imgs, fps=10)
            print(f"\n--- ✅ Video saved to: {output_path} ---")
    
    # --- 4. Aggregate and Save Final Results ---
    print("\n" + "="*50)
    print("---           Aggregate Evaluation Results           ---")
    print("="*50)

    if not all_metrics_summary:
        print("No scenarios were successfully evaluated. Exiting.")
        return

    num_evaluated = len(all_metrics_summary)
    
    # --- Calculate aggregate statistics for each pillar ---

    # Pillar 1: Safety
    total_collisions = int(np.sum([m['collided'] for m in all_metrics_summary]))
    total_offroad = int(np.sum([m['offroad'] for m in all_metrics_summary]))
    collision_rate = 100 * total_collisions / num_evaluated
    offroad_rate = 100 * total_offroad / num_evaluated
    
    # For TTC, we are interested in the average of the minimums, but only for runs
    # where a potential interaction occurred (TTC < 999).
    valid_ttc_runs = [m['min_ttc'] for m in all_metrics_summary if m['min_ttc'] < 999.0]
    avg_min_ttc = float(np.mean(valid_ttc_runs)) if valid_ttc_runs else 999.0

    # Pillar 4: Rule Compliance
    total_red_light_violations = int(np.sum([m['red_light_violation'] for m in all_metrics_summary]))
    red_light_violation_rate = 100 * total_red_light_violations / num_evaluated

    # --- Define "successful" runs for calculating conditional metrics ---
    successful_runs = [m for m in all_metrics_summary if not m['collided'] and not m['offroad']]
    num_successful = len(successful_runs)

    if num_successful > 0:
        # Pillar 2: Goal Achievement (on successful runs)
        avg_progression = float(np.mean([m['sdc_progression'] for m in successful_runs]))
        avg_final_dist_to_goal = float(np.mean([m['final_distance_to_goal'] for m in successful_runs]))
        avg_route_adherence = float(np.mean([m['route_adherence_error'] for m in successful_runs]))
        
        # Pillar 3: Comfort & Smoothness (on successful runs)
        avg_max_jerk = float(np.mean([m['max_jerk'] for m in successful_runs]))
        avg_max_lat_accel = float(np.mean([m['max_lateral_acceleration'] for m in successful_runs]))
    else:
        # Handle case where no runs were successful
        avg_progression, avg_final_dist_to_goal, avg_route_adherence = 0.0, 999.0, 999.0
        avg_max_jerk, avg_max_lat_accel = 999.0, 999.0

    # --- Create the final summary dictionary for the JSON file ---
    summary = {
        'run_name': args.run_name,
        'model_path': args.model_path,
        'num_scenarios_evaluated': num_evaluated,
        'metrics_summary': {
            'safety': {
                'collision_rate_percent': collision_rate,
                'offroad_rate_percent': offroad_rate,
                'average_minimum_ttc_seconds': avg_min_ttc,
                'total_collisions': total_collisions,
                'total_offroad': total_offroad,
            },
            'goal_achievement': {
                'success_rate_percent': 100 * num_successful / num_evaluated,
                'num_successful_runs': num_successful,
                'average_progression_on_success_meters': avg_progression,
                'average_final_distance_to_goal_meters': avg_final_dist_to_goal,
                'average_route_adherence_error_meters': avg_route_adherence,
            },
            'comfort': {
                'average_max_jerk_on_success': avg_max_jerk,
                'average_max_lateral_acceleration_on_success': avg_max_lat_accel,
            },
            'rule_compliance': {
                 'red_light_violation_rate_percent': red_light_violation_rate,
                 'total_red_light_violations': total_red_light_violations,
            }
        },
        'details_per_scenario': all_metrics_summary
    }
    
    # --- Print a clean summary to the console ---
    print(f"\n--- SAFETY METRICS ---")
    print(f"  Collision Rate: {collision_rate:.2f}% ({total_collisions} incidents)")
    print(f"  Off-road Rate: {offroad_rate:.2f}% ({total_offroad} incidents)")
    print(f"  Avg. Min TTC (on interacting runs): {avg_min_ttc:.2f} seconds")
    
    print(f"\n--- RULE COMPLIANCE ---")
    print(f"  Red Light Violation Rate: {red_light_violation_rate:.2f}% ({total_red_light_violations} incidents)")

    print(f"\n--- GOAL ACHIEVEMENT (on {num_successful} successful runs) ---")
    print(f"  Success Rate: {100 * num_successful / num_evaluated:.2f}%")
    print(f"  Avg. Progression: {avg_progression:.2f} meters")
    print(f"  Avg. Final Distance to Goal: {avg_final_dist_to_goal:.2f} meters")
    print(f"  Avg. Route Adherence Error: {avg_route_adherence:.2f} meters")

    print(f"\n--- COMFORT (on successful runs) ---")
    print(f"  Avg. Max Jerk: {avg_max_jerk:.2f} m/s^3")
    print(f"  Avg. Max Lateral Acceleration: {avg_max_lat_accel:.2f} m/s^2")

    # --- Determine the output path and save ---
    if args.metrics_file:
        output_path = args.metrics_file
    else:
        output_dir = os.path.join('outputs', args.run_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'final_metrics_summary.json')

    try:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\n--- ✅ Metrics summary saved to: {output_path} ---")
    except Exception as e:
        print(f"\n--- ❌ Error saving metrics file: {e} ---")
        traceback.print_exc()

if __name__ == "__main__":
    main()