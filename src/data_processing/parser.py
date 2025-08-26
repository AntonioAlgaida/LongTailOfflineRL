# src/data_processing/parser.py

# This script should be run with parser environment activated.
# conda activate womd-parser
# python -m src.data_processing.parser

# To know more about the Waymo Open Dataset, visit:
# https://waymo.com/open/data/motion/tfexample

import os
import tensorflow as tf
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
import sys
from typing import Dict
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# Important: Must be imported before Waymo-related imports
# to avoid a protobuf version conflict.
from src.utils.config import load_config

# Now import Waymo specifics
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import map_pb2 # We still need this here in the parser env

# --- Global Configuration ---
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
CONFIG = load_config(CONFIG_PATH)

# --- Main Parsing Logic ---


def parse_scenario(serialized_scenario: bytes) -> Dict:
    """
    (Definitive, Corrected Version) Parses a Waymo scenario, saving all relevant data
    including agent roles, interactions, and map features using the official
    integer IDs from the Waymo tf.Example format.
    """
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(serialized_scenario)

    num_timestamps = len(scenario.timestamps_seconds)
    num_agents = len(scenario.tracks)

    # --- 1. Agent Trajectories and Static Metadata (Single Pass) ---
    trajectories = np.zeros((num_agents, num_timestamps, 9), dtype=np.float32)
    valid_mask = np.zeros((num_agents, num_timestamps), dtype=np.bool_)
    object_ids = np.zeros(num_agents, dtype=np.int32)
    object_types = np.zeros(num_agents, dtype=np.uint8)

    for i, track in enumerate(scenario.tracks):
        object_ids[i] = track.id
        object_types[i] = track.object_type
        for j, state in enumerate(track.states):
            # --- ROBUST, EXPLICIT CASTING ---
            # By explicitly casting each value to float(), we prevent any
            # potential bit-level corruption during the assignment of mixed
            # double/float types into a float32 NumPy array.
            heading_rad = np.float32(state.heading)
            sanitized_heading = (heading_rad + np.pi) % (2 * np.pi) - np.pi
    
            trajectories[i, j] = [
                np.float32(state.center_x),
                np.float32(state.center_y),
                np.float32(state.center_z),
                np.float32(state.length),
                np.float32(state.width),
                np.float32(state.height),
                sanitized_heading,  # <-- Use the sanitized value
                np.float32(state.velocity_x),
                np.float32(state.velocity_y)
            ]
            valid_mask[i, j] = state.valid
            
            # Check each trajectory entry for validity
            # Heading is in radians
            # if np.abs(trajectories[i, j, 6]) > np.pi:
                # raise(f"Invalid heading value {trajectories[i, j, 6]} at track {i}, timestamp {j} on scenario {scenario.scenario_id}")
            
            

    # --- 2. SDC Route and Agent Roles ---
    sdc_route = trajectories[scenario.sdc_track_index]
    # is_sdc flag
    # We use a boolean array where the first column is True for SDC
    # and the second column is True for tracks to predict.
    agent_roles = np.zeros((num_agents, 2), dtype=np.bool_)
    agent_roles[scenario.sdc_track_index, 0] = True
    # is_track_to_predict flag (More efficient version)
    track_to_predict_indices = [track.track_index for track in scenario.tracks_to_predict]
    if track_to_predict_indices:
        agent_roles[track_to_predict_indices, 1] = True
        
    '''
    The agent_roles array is a NumPy array of shape (Number of Agents, 2). For each agent i in the scene,
    the row agent_roles[i] contains two boolean (True/False) values:
    
    Column 0: is_sdc
    Meaning: This flag is True if this agent is the Self-Driving Car (the ego-vehicle that we are
    controlling/learning from). It will be False for all other agents.
    Source: This is set using the scenario.sdc_track_index.
    Why it's useful: While you also store the sdc_track_index separately, having this boolean flag directly
    in an array can sometimes make filtering and vectorized operations easier. For example, to get all non-SDC
    agents, you could simply use the mask ~agent_roles[:, 0]. It's a convenient way to quickly identify the main actor.
    
    Column 1: is_track_to_predict
    Meaning: This flag is True if this agent was marked by Waymo as an "object to be predicted" for their
    internal motion prediction challenges.
    Source: This is set using the scenario.tracks_to_predict list from the protobuf.
    Why it's useful (very important): These agents are not chosen randomly. Waymo flags them because they are
    typically the most interesting or challenging agents in the scene. They are often agents that are interacting
    closely with the SDC or with each other. For your "long-tail" project, this is a powerful, pre-computed
    signal of criticality.
        Potential Use Case: You could create a new heuristic score based on this flag. For example, a timestep
        could be considered more critical if the SDC is in close proximity to an agent where is_track_to_predict
        is True. This is a direct hint from the dataset creators about which interactions are the most important.
    '''

    # The agent_is_interest is a 1D boolean NumPy array of shape (Number of Agents,).
    # For each agent i, the value agent_is_interest[i] is True if that agent is one of the objects_of_interest.
    
    agent_is_interest = np.zeros(num_agents, dtype=np.bool_)
    if scenario.objects_of_interest:
        id_to_idx = {obj_id: i for i, obj_id in enumerate(object_ids)}
        interest_indices = [id_to_idx[obj_id] for obj_id in scenario.objects_of_interest if obj_id in id_to_idx]
        if interest_indices:
            agent_is_interest[interest_indices] = True

    # --- 3. Map Feature Extraction ---
    map_polylines, map_polyline_types, map_polyline_ids = [], [], []
    lane_connectivity = {}
    
    # print(f'Check the map features for scenario {scenario.scenario_id}...')

    for feature in scenario.map_features:
        feature_type_str = feature.WhichOneof('feature_data')
        # print(f"Processing feature type: {feature_type_str} for scenario {scenario.scenario_id}")
        if feature_type_str is None:
            print(f"Warning: Feature {feature.id} has no valid type in scenario {scenario.scenario_id}, skipping.")
            continue
        
        feature_data = getattr(feature, feature_type_str)
        type_id = -1
        polyline_np = None

        if feature_type_str == 'lane':
            # LaneType enum is 1, 2, 3. We use these directly.
            type_id = feature_data.type
            points = feature_data.polyline
            polyline_np = np.array([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float32)
            
            # Extract lane connectivity
            lane_connectivity[feature.id] = {
                'entry': list(feature_data.entry_lanes),
                'exit': list(feature_data.exit_lanes),
            }

        elif feature_type_str == 'road_line':
            # RoadLineType enum is 1-8. We map it to 11-18.
            if feature_data.type != map_pb2.RoadLine.TYPE_UNKNOWN:
                type_id = feature_data.type + 10 # Base ID for RoadLine
                points = feature_data.polyline
                polyline_np = np.array([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float32)
            else:
                print(f"Warning: Unknown RoadLine type in scenario {scenario.scenario_id}, skipping.")

        elif feature_type_str == 'road_edge':
            # RoadEdgeType enum is 1-2. We map it to 21-22.
            if feature_data.type != map_pb2.RoadEdge.TYPE_UNKNOWN:
                type_id = feature_data.type + 20 # Base ID for RoadEdge
                points = feature_data.polyline
                polyline_np = np.array([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float32)
            else:
                print(f"Warning: Unknown RoadEdge type in scenario {scenario.scenario_id}, skipping.")
        
        elif feature_type_str == 'stop_sign':
            type_id = 31 # Use a unique ID in the 30s block
            pos = feature_data.position
            polyline_np = np.array([[float(pos.x), float(pos.y), float(pos.z)]], dtype=np.float32)
        
        elif feature_type_str == 'crosswalk':
            type_id = 41 # Use a unique ID in the 40s block
            points = feature_data.polygon
            polyline_np = np.array([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float32)
            
        elif feature_type_str == 'speed_bump':
            type_id = 51 # Use a unique ID in the 50s block
            points = feature_data.polygon
            polyline_np = np.array([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float32)
        
        elif feature_type_str == 'driveway':
            type_id = 61 # Use a unique ID in the 60s block
            points = feature_data.polygon
            polyline_np = np.array([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float32)
        
        else:
            print(f"Warning: Unsupported feature type '{feature_type_str}' in scenario {scenario.scenario_id}, skipping.")
            continue
        
        if type_id != -1 and polyline_np is not None and polyline_np.shape[0] > 0:
            map_polylines.append(polyline_np)
            map_polyline_types.append(type_id)
            map_polyline_ids.append(feature.id)
        else:
            print(f"Warning: Invalid polyline data for feature type '{feature_type_str}' in scenario {scenario.scenario_id}, skipping.")    

    # --- Extract Dynamic Map Data (Corrected and Enriched Version) ---
    num_traffic_lights = max(len(s.lane_states) for s in scenario.dynamic_map_states) if scenario.dynamic_map_states else 0
    
    # NEW: Change the last dimension from 2 to 4 to include stop_point coordinates
    # Shape: (num_timestamps, num_traffic_lights, 4) -> [lane_id, state_enum, stop_x, stop_y]
    dynamic_map_states = np.zeros((num_timestamps, num_traffic_lights, 4), dtype=np.float32) # Use float for coordinates

    for i, frame_states in enumerate(scenario.dynamic_map_states):
        for j, light_state in enumerate(frame_states.lane_states):
            if j < num_traffic_lights:
                # Get the stop point coordinates, defaulting to 0 if not present
                stop_x = np.float32(light_state.stop_point.x)# if light_state.HasField('stop_point') else 0.0
                stop_y = np.float32(light_state.stop_point.y)# if light_state.HasField('stop_point') else 0.0
                
                # print(f"Processing traffic light state for lane {light_state.lane} at timestamp {i}, light index {j}")
                # print(f"  - State: {light_state.state}, Stop Point: ({stop_x}, {stop_y})")
                
                # Store all four pieces of information
                dynamic_map_states[i, j] = [light_state.lane, light_state.state, stop_x, stop_y]


    return {
        'scenario_id': scenario.scenario_id,
        'timestamps': np.array(scenario.timestamps_seconds, dtype=np.float16),
        'sdc_track_index': scenario.sdc_track_index,
        
        # All agent data
        'object_ids': object_ids,
        'object_types': object_types.astype(np.int8), # OPTIMIZED
        'all_agent_trajectories': trajectories,
        'valid_mask': valid_mask,
        
        # NEW: High-level agent metadata
        'agent_roles': agent_roles,             # Shape (A, 2) -> [is_sdc, is_ttp]
        'agent_is_interest': agent_is_interest, # Shape (A,)
        
        # Ground-truth goal for the planner
        'sdc_route': sdc_route,
        
        # Map data
        'dynamic_map_states': dynamic_map_states,
        'map_polylines': np.array(map_polylines, dtype=object),
        'map_polyline_types': np.array(map_polyline_types, dtype=np.int8),
        'map_polyline_ids': np.array(map_polyline_ids, dtype=np.int64),
        'lane_connectivity': lane_connectivity, # NEW
    }
    

def process_tfrecord_shard(shard_path: str):
    """
    Processes a single .tfrecord file, parsing all its scenarios
    and saving them as individual .npz files.
    
    Args:
        shard_path: The file path to the .tfrecord shard.
    """
    output_dir = os.path.join(CONFIG['data']['processed_npz_dir'], os.path.basename(os.path.dirname(shard_path)))
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = tf.data.TFRecordDataset(shard_path, compression_type='')
    
    num_scenarios_processed = 0
    for serialized_record in dataset:
        parsed_data = parse_scenario(serialized_record.numpy())
        output_path = os.path.join(output_dir, f"{parsed_data['scenario_id']}.npz")
        
        # We need allow_pickle=True for the map_features dictionary
        np.savez_compressed(output_path, **parsed_data, allow_pickle=True)
        num_scenarios_processed += 1
        
    return num_scenarios_processed

def main():
    """
    Main function to find all .tfrecord shards and process them in parallel.
    Includes a safe-guard to clear the previous output directory.
    """
    output_dir = CONFIG['data']['processed_npz_dir']
    
    # --- SAFE DELETION LOGIC ---
    # 1. Check if the directory exists
    if os.path.exists(output_dir):
        # 2. User Confirmation
        print("\n--- WARNING: Existing Data Found ---")
        print(f"The output directory '{output_dir}' already exists.")
        response = input("Do you want to permanently delete it and its contents to start fresh? [y/N]: ")
        
        if response.lower() == 'y':
            try:
                print(f"Deleting directory: {output_dir}")
                shutil.rmtree(output_dir)
                print("Directory deleted successfully.")
            except OSError as e:
                print(f"Error deleting directory {output_dir}: {e}")
                sys.exit(1)
        else:
            print("Aborting. Please move or delete the directory manually if you wish to re-run the parser.")
            sys.exit(0) # Exit gracefully

    print("\nStarting Waymo data parsing...")
    
    # Find all .tfrecord files for both training and validation
    raw_data_dir = CONFIG['data']['raw_data_dir']
    shard_paths = glob(os.path.join(raw_data_dir, '*', '*.tfrecord*'))
    
    if not shard_paths:
        print(f"Error: No .tfrecord files found in {raw_data_dir}.")
        print("Please check the `raw_data_dir` path in your `main_config.yaml`.")
        return

    print(f"Found {len(shard_paths)} .tfrecord shards to process.")
    
    # Use multiprocessing Pool to parallelize the parsing
    num_workers = CONFIG['data'].get('num_workers', cpu_count())
    print(f"Using {num_workers} worker processes.")

    with Pool(processes=num_workers) as pool:
        # tqdm shows progress for the parallel processing
        results = list(tqdm(pool.imap(process_tfrecord_shard, shard_paths), total=len(shard_paths)))

    total_scenarios = sum(results)
    print("\n-------------------------------------------")
    print("Parsing complete!")
    print(f"Total scenarios processed: {total_scenarios}")
    print(f"Output saved to: {output_dir}")
    print("-------------------------------------------")


if __name__ == '__main__':
    # Disable all GPU usage for TensorFlow, as this is a CPU-bound task.
    tf.config.set_visible_devices([], 'GPU')
    main()