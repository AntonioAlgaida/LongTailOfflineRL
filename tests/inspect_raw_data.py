# inspect_raw_data.py
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

# --- Values for a known corrupted sample ---
SCENARIO_ID_TO_FIND = "28fe360951cf98d6"
TRACK_INDEX_TO_FIND = 5
TIMESTEP_TO_FIND = 0

# Path to the specific .tfrecord shard containing the scenario
# (Users will need to adjust this path)
TFRECORD_PATH = "/mnt/d/waymo_datasets/uncompressed/scenario/training/training.tfrecord-00000-of-01000"

def main():
    dataset = tf.data.TFRecordDataset(TFRECORD_PATH, compression_type='')
    print(f"Scanning shard for scenario ID: {SCENARIO_ID_TO_FIND}...")
    
    found = False
    for record in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(record.numpy())
        
        if scenario.scenario_id == SCENARIO_ID_TO_FIND:
            found = True
            print(f"\n--- Found Scenario: {scenario.scenario_id} ---")
            
            track = scenario.tracks[TRACK_INDEX_TO_FIND]
            state = track.states[TIMESTEP_TO_FIND]
            
            print(f"Inspecting Track Index: {TRACK_INDEX_TO_FIND} at Timestep: {TIMESTEP_TO_FIND}")
            print(f"Is State Valid: {state.valid}")
            
            # This value violates the [-pi, pi) contract
            print(f"Corrupted Heading Value: {state.heading}") 
            
            break
            
    if not found:
        print("Scenario not found in the specified shard.")

if __name__ == '__main__':
    main()