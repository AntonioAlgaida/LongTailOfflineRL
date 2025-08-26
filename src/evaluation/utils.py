# src/evaluation/utils.py

import numpy as np
import jax.numpy as jnp
from waymax import datatypes

def get_parser_to_waymax_type_mapping() -> dict:
    """
    Creates a mapping from our custom parser integer IDs to the official
    Waymax.datatypes.MapElementIds enums for visualization.
    https://waymo-research.github.io/waymax/docs/autoapi/waymax/datatypes/index.html#waymax.datatypes.MapElementIds
    """
    # This ensures we use the official enums for max compatibility
    # and makes the code more readable.
    return {
        # --- Lanes (Parser IDs 0-3 match Waymax IDs 0-3) ---
        # Parser ID 0 is 'lane_undefined' -> Waymax LANE_UNDEFINED (0)
        0: datatypes.MapElementIds.LANE_UNDEFINED,
        # Parser ID 1 is 'lane_freeway' -> Waymax LANE_FREEWAY (1)
        1: datatypes.MapElementIds.LANE_FREEWAY,
        # Parser ID 2 is 'lane_surface_street' -> Waymax LANE_SURFACE_STREET (2)
        2: datatypes.MapElementIds.LANE_SURFACE_STREET,
        # Parser ID 3 is 'lane_bike_lane' -> Waymax LANE_BIKE_LANE (3)
        3: datatypes.MapElementIds.LANE_BIKE_LANE,
        
        # --- Road Lines (Parser IDs 10-18 map to Waymax IDs 5-13) ---
        # Parser ID 10 is 'road_line_unknown' -> Waymax ROAD_LINE_UNKNOWN (5)
        10: datatypes.MapElementIds.ROAD_LINE_UNKNOWN,
        # Parser ID 11 is 'road_line_broken_single_white' -> Waymax ROAD_LINE_BROKEN_SINGLE_WHITE (6)
        11: datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE,
        12: datatypes.MapElementIds.ROAD_LINE_SOLID_SINGLE_WHITE,
        13: datatypes.MapElementIds.ROAD_LINE_SOLID_DOUBLE_WHITE,
        14: datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_YELLOW,
        15: datatypes.MapElementIds.ROAD_LINE_BROKEN_DOUBLE_YELLOW,
        16: datatypes.MapElementIds.ROAD_LINE_SOLID_SINGLE_YELLOW,
        17: datatypes.MapElementIds.ROAD_LINE_SOLID_DOUBLE_YELLOW,
        18: datatypes.MapElementIds.ROAD_LINE_PASSING_DOUBLE_YELLOW,
        
        # --- Road Edges (Parser IDs 20-22 map to Waymax IDs 14-16) ---
        # Parser ID 20 is 'road_edge_unknown' -> Waymax ROAD_EDGE_UNKNOWN (14)
        20: datatypes.MapElementIds.ROAD_EDGE_UNKNOWN,
        # Parser ID 21 is 'road_edge_boundary' -> Waymax ROAD_EDGE_BOUNDARY (15)
        21: datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
        # Parser ID 22 is 'road_edge_median' -> Waymax ROAD_EDGE_MEDIAN (16)
        22: datatypes.MapElementIds.ROAD_EDGE_MEDIAN,
        
        # --- Other Features ---
        # Parser ID 30 is 'stop_sign' -> Waymax STOP_SIGN (17)
        30: datatypes.MapElementIds.STOP_SIGN,
        # Parser ID 40 is 'crosswalk' -> Waymax CROSSWALK (18)
        40: datatypes.MapElementIds.CROSSWALK,
        # Parser ID 50 is 'speed_bump' -> Waymax SPEED_BUMP (19)
        50: datatypes.MapElementIds.SPEED_BUMP,
        
        # Parser ID 60 ('driveway') does not have a direct enum in Waymax.
        # We can map it to UNKNOWN.
        60: datatypes.MapElementIds.UNKNOWN,
    }

def get_waymax_to_parser_type_mapping() -> dict:
    """Creates the reverse mapping from Waymax enums to our parser IDs."""
    # This is the inverse of the other mapping function
    parser_to_waymax = get_parser_to_waymax_type_mapping()
    return {v.value: k for k, v in parser_to_waymax.items()}

def construct_state_from_npz(data: np.lib.npyio.NpzFile, config: dict) -> datatypes.SimulatorState:
    """
    (V3 - JAX Compliant) Takes a loaded .npz data object and constructs a
    Waymax SimulatorState with fixed-size, padded RoadgraphPoints.
    """
    print("Constructing Waymax SimulatorState from .npz data...")
    num_agents = data['object_ids'].shape[0]

    # --- 1. Trajectory Data ---
    timestamps_seconds = data['timestamps']
    timestamps_micros = (timestamps_seconds.astype(np.float64) * 1_000_000).astype(np.int64)
    
    log_trajectory = datatypes.Trajectory(
        x=jnp.array(data['all_agent_trajectories'][:, :, 0]),
        y=jnp.array(data['all_agent_trajectories'][:, :, 1]),
        z=jnp.array(data['all_agent_trajectories'][:, :, 2]),
        length=jnp.array(data['all_agent_trajectories'][:, :, 3]),
        width=jnp.array(data['all_agent_trajectories'][:, :, 4]),
        height=jnp.array(data['all_agent_trajectories'][:, :, 5]),
        yaw=jnp.array(data['all_agent_trajectories'][:, :, 6]),
        vel_x=jnp.array(data['all_agent_trajectories'][:, :, 7]),
        vel_y=jnp.array(data['all_agent_trajectories'][:, :, 8]),
        valid=jnp.array(data['valid_mask'], dtype=bool),
        timestamp_micros=jnp.array(timestamps_micros)
    )

    # --- 2. Object Metadata (CORRECTED) ---
    sdc_idx = int(data['sdc_track_index'])
    num_agents = data['object_ids'].shape[0]
    
    is_sdc_mask = jnp.zeros(num_agents, dtype=bool).at[sdc_idx].set(True)
    
    # Handle cases where 'agent_difficulty' might not be saved by the parser
    is_modeled = jnp.zeros(num_agents, dtype=bool)
    if 'agent_difficulty' in data:
        is_modeled = jnp.array(data['agent_difficulty'] > 0, dtype=bool)
        
    # --- NEW: Prepare the required 'is_valid' and 'objects_of_interest' fields ---
    
    # 'is_valid': An object is valid if it's valid at ANY point in the trajectory.
    # We can compute this by taking the .any() along the time axis of the valid_mask.
    is_valid_per_object = jnp.array(data['valid_mask'], dtype=bool).any(axis=1)

    # 'objects_of_interest': Create a boolean mask from the list of IDs.
    objects_of_interest_mask = jnp.zeros(num_agents, dtype=bool)
    if 'objects_of_interest' in data and data['objects_of_interest'].shape[0] > 0:
        # We need to map the object IDs to their indices
        id_to_idx_mapping = {obj_id: i for i, obj_id in enumerate(data['object_ids'])}
        interest_indices = [
            id_to_idx_mapping[obj_id] for obj_id in data['objects_of_interest'] 
            if obj_id in id_to_idx_mapping
        ]
        if interest_indices:
            objects_of_interest_mask = objects_of_interest_mask.at[jnp.array(interest_indices)].set(True)
    # --- END NEW ---

    object_metadata = datatypes.ObjectMetadata(
        ids=jnp.array(data['object_ids']),
        object_types=jnp.array(data['object_types']),
        is_sdc=is_sdc_mask,
        is_controlled=is_sdc_mask,
        is_modeled=is_modeled,
        is_valid=is_valid_per_object,           # <-- ADDED
        objects_of_interest=objects_of_interest_mask # <-- ADDED
    )

    # --- 3. Roadgraph Points (CORRECTED with Padding) ---
    print("Reconstructing and padding RoadgraphPoints...")
    map_polylines = data['map_polylines']
    map_types_parser = data['map_polyline_types']
    map_ids = data['map_polyline_ids']
    type_mapping = get_parser_to_waymax_type_mapping()

    points_x, points_y, points_z, points_ids, points_types = [], [], [], [], []

    for i in range(len(map_polylines)):
        polyline = map_polylines[i]
        parser_type = map_types_parser[i]
        feature_id = map_ids[i]
        # Map to Waymax's enum integer value, defaulting to UNKNOWN
        waymax_type_enum = type_mapping.get(int(parser_type), datatypes.MapElementIds.UNKNOWN)
        waymax_type = waymax_type_enum.value
        
        num_points = polyline.shape[0]
        if num_points > 0:
            points_x.append(polyline[:, 0])
            points_y.append(polyline[:, 1])
            points_z.append(polyline[:, 2])
            points_ids.append(np.full(num_points, feature_id, dtype=np.int32))
            points_types.append(np.full(num_points, waymax_type, dtype=np.int32))

    # Concatenate all real points
    x_np = np.concatenate(points_x) if points_x else np.array([])
    y_np = np.concatenate(points_y) if points_y else np.array([])
    z_np = np.concatenate(points_z) if points_z else np.array([])
    ids_np = np.concatenate(points_ids) if points_ids else np.array([])
    types_np = np.concatenate(points_types) if points_types else np.array([])
    num_real_points = len(x_np)
    
    print(f"Constructed roadgraph with {num_real_points} points.")
    
    # Create the RoadgraphPoints object with the exact number of points.
    roadgraph_points = datatypes.RoadgraphPoints(
        x=jnp.array(x_np),
        y=jnp.array(y_np),
        z=jnp.array(z_np),
        # Direction vectors are not used by our metrics, can be zeros
        dir_x=jnp.zeros(num_real_points, dtype=jnp.float32),
        dir_y=jnp.zeros(num_real_points, dtype=jnp.float32),
        dir_z=jnp.zeros(num_real_points, dtype=jnp.float32),
        ids=jnp.array(ids_np),
        types=jnp.array(types_np),
        # All points we include are valid by definition.
        valid=jnp.ones(num_real_points, dtype=bool)
    )

    # --- 4. Traffic Lights ---
    tl_states_data = data['dynamic_map_states'] # Shape: (T, L, 4) -> [lane_id, state, stop_x, stop_y]
    num_timesteps, num_tl_lanes, _ = tl_states_data.shape
    
    # Our parser stores the stop points at each timestep. For the TrafficLights
    # constructor, which expects a fixed position, we can take the position from
    # the first valid timestep for each traffic light.
    
    # Initialize placeholder arrays for stop point coordinates
    stop_x = np.zeros(num_tl_lanes, dtype=np.float32)
    stop_y = np.zeros(num_tl_lanes, dtype=np.float32)

    for i in range(num_tl_lanes):
        # Find the first valid timestep for this light to get its static position
        valid_frames = np.where(tl_states_data[:, i, 1] > 0)[0]
        if len(valid_frames) > 0:
            first_valid_frame = valid_frames[0]
            stop_x[i] = tl_states_data[first_valid_frame, i, 2]
            stop_y[i] = tl_states_data[first_valid_frame, i, 3]

    # Tile the static stop point positions across all timesteps to match the expected shape (L, T)
    stop_x_tiled = jnp.tile(jnp.array(stop_x)[:, None], (1, num_timesteps))
    stop_y_tiled = jnp.tile(jnp.array(stop_y)[:, None], (1, num_timesteps))

    traffic_lights = datatypes.TrafficLights(
        x=stop_x_tiled,                                       # <-- ADDED
        y=stop_y_tiled,                                       # <-- ADDED
        z=jnp.zeros((num_tl_lanes, num_timesteps)),           # <-- ADDED (with placeholder z)
        state=jnp.array(tl_states_data[:, :, 1].T, dtype=jnp.int32),
        valid=jnp.array(tl_states_data[:, :, 1] > 0, dtype=bool).T,
        lane_ids=jnp.array(tl_states_data[0, :, 0], dtype=jnp.int32)[:, None]
    )

    # --- 5. Final Assembly ---
    # The simulation is initialized at timestep 10 (the 11th step), which is the
    # standard "current time" for the WOMD prediction tasks.
    initial_timestep = 10
    
    return datatypes.SimulatorState(
        log_trajectory=log_trajectory,
        sim_trajectory=log_trajectory,
        object_metadata=object_metadata,
        roadgraph_points=roadgraph_points,
        log_traffic_light=traffic_lights,
        timestep=jnp.array(initial_timestep, dtype=jnp.int32),
    )