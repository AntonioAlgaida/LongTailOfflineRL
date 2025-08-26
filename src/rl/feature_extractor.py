# src/rl/feature_extractor.py

import numpy as np
from typing import Dict, Union, List, Tuple
from scipy.spatial import cKDTree
from src.utils import geometry
from src.evaluation.utils import get_waymax_to_parser_type_mapping
from numpy.lib.npyio import NpzFile
import jax.numpy as jnp

# Use typing for forward declaration of Waymax type
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from waymax.datatypes import SimulatorState

from waymax import datatypes


class FeatureExtractor:
    """
    (V3 - Goal-Conditioned, Structured Output)
    
    A robust feature extractor that converts raw scenario data from .npz files
    into a structured dictionary of feature tensors. This version is goal-conditioned,
    includes the SDC's intended future path, and produces an output ready for
    attention-based models.
    """
    def __init__(self, config: Dict):
        self.features_config = config['features']
        self.map_config = config.get('map', {})
        print("FeatureExtractor (V3 - Goal-Conditioned) initialized.")

    def extract_features(
        self, 
        source: Union[Dict[str, np.ndarray], NpzFile],
        timestep_index: int
    ) -> Dict[str, np.ndarray]:
        """
        Main public method. Orchestrates the feature extraction pipeline for a single timestep.
        """
        # --- 1. Unpack Raw Data ---
        # Extracts all necessary NumPy arrays from the source .npz file for the given timestep.
        try:
            (sdc_state_global, other_agents_global, other_agent_types, 
             lane_polylines_global, traffic_lights_global, sdc_route) = self._unpack_npz_data(source, timestep_index)
        except ValueError as e:
            raise ValueError(f"Failed to unpack data at timestep {timestep_index}: {e}")

        # --- 2. Ego-Centric Transformation ---
        # Establishes the SDC's current position and heading as the origin (0,0) of our coordinate system.
        # This is a critical step for making the features generalizable.
        # sdc_state_global: # [global_x, global_y, global_z, length, width, height, global_yaw, vx, vy]
        ego_pose = sdc_state_global[[0, 1, 6]] # [global_x, global_y, global_yaw]
        
        other_agents_ego = self._transform_agents_to_ego(other_agents_global, ego_pose)
        lane_polylines_ego = [geometry.transform_points(p, ego_pose) for p in lane_polylines_global]
        
        # --- 3. Compute Individual Feature Tensors ---
        # Each helper function is responsible for creating a padded, fixed-size tensor for one entity type.
        ego_features, ego_mask = self._get_ego_features(sdc_state_global)
        agent_features, agents_mask = self._get_agent_features(other_agents_ego, other_agent_types)
        map_features, map_mask = self._get_map_features(lane_polylines_ego)
        traffic_light_features, tl_mask = self._get_traffic_light_features(traffic_lights_global, ego_pose)
        goal_features, goal_mask = self._get_goal_features(sdc_route, timestep_index, ego_pose)

        # --- 4. Assemble and Return Final Dictionary ---
        feature_dict = {
            'ego': ego_features, 'ego_mask': ego_mask,
            'agents': agent_features, 'agents_mask': agents_mask,
            'map': map_features, 'map_mask': map_mask,
            'traffic_lights': traffic_light_features, 'traffic_lights_mask': tl_mask,
            'goal': goal_features, 'goal_mask': goal_mask,
        }
        
        # Final sanity check to prevent data corruption from propagating.
        for key, tensor in feature_dict.items():
            if not np.all(np.isfinite(tensor)):
                raise ValueError(f"Invalid number (NaN/inf) in feature '{key}' at timestep {timestep_index}.")
        
        return feature_dict

    def _unpack_npz_data(self, data: Dict, timestep: int) -> Tuple:
        """Unpacks and filters all necessary raw data arrays from the .npz file."""
        sdc_track_index = data['sdc_track_index'].item() # Use .item() for safety
        
        if not data['valid_mask'][sdc_track_index, timestep]:
            raise ValueError(f"SDC not valid at timestep {timestep}")

        states_at_t = data['all_agent_trajectories'][:, timestep, :]
        valid_mask_at_t = data['valid_mask'][:, timestep]
        sdc_state_global = states_at_t[sdc_track_index]
        
        # Select all agents that are valid AND are not the SDC
        other_agents_mask = valid_mask_at_t & (np.arange(len(valid_mask_at_t)) != sdc_track_index)
        other_agents_global = states_at_t[other_agents_mask]
        other_agent_types = data['object_types'][other_agents_mask]
        
        # Select map features that are lane centerlines
        map_polylines = list(data['map_polylines'])
        map_types = list(data['map_polyline_types'])
        
        # Include TYPE_UNDEFINED (ID 0) as it often represents lanes in intersections.
        lane_polylines_global = [p for p, t in zip(map_polylines, map_types) if t in {0, 1, 2, 3}]
        
        # Select traffic light states that are valid
        dynamic_map_states_t = data['dynamic_map_states'][timestep, :, :]
        traffic_lights_global = dynamic_map_states_t[dynamic_map_states_t[:, 0] > 0]
        
        sdc_route = data['sdc_route']
        
        return sdc_state_global, other_agents_global, other_agent_types, lane_polylines_global, traffic_lights_global, sdc_route

    def _transform_agents_to_ego(self, agents_global: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
        """Transforms agent positions, velocities, and headings to be ego-centric."""
        if agents_global.shape[0] == 0:
            return agents_global
        
        agents_ego = agents_global.copy()
        # Transform positions
        agents_ego[:, :2] = geometry.transform_points(agents_global[:, :2], ego_pose)
        # Transform velocities
        rot_mat = geometry.rotation_matrix(-ego_pose[2])
        agents_ego[:, 7:9] = agents_global[:, 7:9] @ rot_mat.T
        # Transform headings and re-normalize to [-pi, pi]
        agents_ego[:, 6] = (agents_global[:, 6] - ego_pose[2] + np.pi) % (2 * np.pi) - np.pi
        return agents_ego

    def _get_ego_features(self, sdc_state_global: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the SDC's own kinematic features."""
        # Feature: [speed]
        speed = np.linalg.norm(sdc_state_global[7:9])
        # Return a mask for consistency, although it's always True for the ego.
        return np.array([speed], dtype=np.float32), np.array([True], dtype=bool)

    def _get_agent_features(self, other_agents_ego: np.ndarray, other_agent_types: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts features for the N closest agents."""
        num_agents = self.features_config['num_agents']
        # Feature: [x, y, vx, vy, cos(h), sin(h), length, width, is_vehicle, is_ped_or_cyc]
        feature_dim = 10
        
        agent_features = np.zeros((num_agents, feature_dim), dtype=np.float32)
        agent_mask = np.zeros(num_agents, dtype=bool)

        if other_agents_ego.shape[0] == 0:
            return agent_features, agent_mask

        # Find the N closest agents based on Euclidean distance
        distances = np.linalg.norm(other_agents_ego[:, :2], axis=1)
        num_to_take = min(len(distances), num_agents)
        nearest_indices = np.argsort(distances)[:num_to_take]
        
        nearest_agents = other_agents_ego[nearest_indices]
        nearest_types = other_agent_types[nearest_indices]
        
        # Populate the feature matrix for the valid, nearest agents
        agent_features[:num_to_take, 0:2] = nearest_agents[:, [0, 1]]  # x, y
        agent_features[:num_to_take, 2:4] = nearest_agents[:, [7, 8]]  # vx, vy
        headings = nearest_agents[:, 6]
        agent_features[:num_to_take, 4] = np.cos(headings)
        agent_features[:num_to_take, 5] = np.sin(headings)
        agent_features[:num_to_take, 6:8] = nearest_agents[:, [3, 4]] # length, width

        # Simplified one-hot encoding for object type
        type_one_hot = np.zeros((num_to_take, 2), dtype=np.float32)
        type_one_hot[nearest_types == 1, 0] = 1.0                # is_vehicle
        type_one_hot[np.isin(nearest_types, [2, 3]), 1] = 1.0    # is_ped_or_cyc
        agent_features[:num_to_take, 8:10] = type_one_hot
        
        # The mask is the single source of truth for validity
        agent_mask[:num_to_take] = True
        return agent_features, agent_mask

    def _get_map_features(self, lane_polylines_ego: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts features for the M closest lane centerlines."""
        num_lanes = self.features_config['num_map_polylines']
        points_per_lane = self.features_config['map_points_per_polyline']
        # Feature: flattened resampled polyline (x,y) coordinates
        feature_dim = points_per_lane * 2
        
        map_features = np.zeros((num_lanes, feature_dim), dtype=np.float32)
        map_mask = np.zeros(num_lanes, dtype=bool)

        if not lane_polylines_ego:
            return map_features, map_mask

        # Find nearest lanes based on distance to their first point
        # A check for empty polylines is added for robustness
        non_empty_polylines = [p for p in lane_polylines_ego if p.shape[0] > 0]
        if not non_empty_polylines:
            return map_features, map_mask
            
        first_points = np.array([p[0, :2] for p in non_empty_polylines])
        distances = np.linalg.norm(first_points, axis=1)
        
        num_to_take = min(len(distances), num_lanes)
        nearest_indices = np.argsort(distances)[:num_to_take]

        for i, original_idx in enumerate(nearest_indices):
            polyline = non_empty_polylines[original_idx]
            # Resample to a fixed number of points
            resampled = geometry.resample_polyline(polyline[:, :2], points_per_lane)
            map_features[i, :] = resampled.flatten()
        
        map_mask[:num_to_take] = True
        return map_features, map_mask

    def _get_traffic_light_features(self, traffic_lights_global: np.ndarray, ego_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts a feature vector for the single most relevant traffic light."""
        # Feature: [is_red_ahead, dist_to_stop_line]
        tl_features = np.zeros(2, dtype=np.float32)
        # The feature is always considered "valid" even if it's all zeros.
        tl_mask = np.array([True], dtype=bool)
        
        if traffic_lights_global.shape[0] == 0:
            return tl_features, tl_mask

        stop_points_global = traffic_lights_global[:, 2:4]
        distances = np.linalg.norm(stop_points_global - ego_pose[:2], axis=1)
        closest_light_idx = np.argmin(distances)
        closest_light_state = traffic_lights_global[closest_light_idx]
        
        # Check if the light is in a "STOP" state (red arrow, red light, or flashing red)
        is_red = int(closest_light_state[1]) in {1, 4, 7}
        
        # To be relevant, the stop point must be generally in front of the SDC (positive x in ego frame)
        stop_point_ego = geometry.transform_points(closest_light_state[2:4].reshape(1, 2), ego_pose)
        
        if is_red and stop_point_ego[0, 0] > -1.0: # Use -1m buffer for robustness
            tl_features[0] = 1.0 # is_red_light_ahead flag
            tl_features[1] = stop_point_ego[0, 0] # Use longitudinal distance
            
        return tl_features, tl_mask
        
    def _get_goal_features(self, sdc_route: np.ndarray, current_timestep: int, ego_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts future waypoints from the expert's route as the goal."""
        num_goal_points = self.features_config.get('num_goal_points', 5)
        horizon_seconds = np.arange(1, num_goal_points+1)
        horizon_steps = (horizon_seconds * 10).astype(int)
        # Feature: just the ego-centric (x, y) coordinates of the goal points
        feature_dim = 2
        
        goal_features = np.zeros((num_goal_points, feature_dim), dtype=np.float32)
        goal_mask = np.zeros(num_goal_points, dtype=bool)

        target_timesteps = current_timestep + horizon_steps
        # Create a boolean mask of which future timesteps are valid (i.e., within the scenario bounds)
        valid_mask = target_timesteps < sdc_route.shape[0]
        
        valid_target_timesteps = target_timesteps[valid_mask]
        if len(valid_target_timesteps) == 0:
            return goal_features, goal_mask

        # Get global waypoints and transform them to the ego-centric frame
        future_waypoints_global = sdc_route[valid_target_timesteps, :2]
        future_waypoints_ego = geometry.transform_points(future_waypoints_global, ego_pose)
        
        # Populate the feature tensor only for the valid future points
        goal_features[valid_mask, :] = future_waypoints_ego
        goal_mask[valid_mask] = True
        
        return goal_features, goal_mask
        
    def extract_features_from_waymax(self, state: "SimulatorState", sdc_route_global: np.ndarray) -> Dict[str, np.ndarray]:

        """
        (Definitive V2) Extracts a flat feature vector from a live Waymax state.
        This is now a wrapper that calls the same logic as the NPZ-based method.
        """
        # --- 1. Unpack Raw Data from Waymax State ---
        try:
            (sdc_state_global_9d, other_agents_global, other_agent_types, 
             lane_polylines_global, traffic_lights_global) = self._unpack_waymax_data(state)
        except (ValueError, IndexError) as e:
            # Re-raise to be handled by the evaluation loop
            raise ValueError(f"Failed to unpack Waymax data: {e}")

        # --- 2. Transform all coordinates to the SDC's ego-centric frame ---
        # The unpacked sdc_state_global is shape (9,), just what we need for the pose
        ego_pose = sdc_state_global_9d[[0, 1, 6]] # [global_x, global_y, global_yaw]
        
        other_agents_ego = self._transform_agents_to_ego(other_agents_global, ego_pose)
        lane_polylines_ego = [geometry.transform_points(p, ego_pose) for p in lane_polylines_global]
        
        # --- 3. Compute Feature Components ---
        # NOTE: _get_ego_features expects the full 9d state. We need to pass that.
        # This requires a small refactor. Let's pass the 5d state and recalculate speed inside.
        # Let's adjust _get_ego_features to accept the 5d state.
        
        # We need the full SDC state to calculate speed. Let's get it from agent_states_global
        ego_features, ego_mask = self._get_ego_features(sdc_state_global_9d)
        agent_features, agents_mask = self._get_agent_features(other_agents_ego, other_agent_types)
        map_features, map_mask = self._get_map_features(lane_polylines_ego)
        traffic_light_features, tl_mask = self._get_traffic_light_features(traffic_lights_global, ego_pose)
        
        # The goal comes from the pre-loaded full SDC route
        current_timestep = state.timestep.item()
        goal_features, goal_mask = self._get_goal_features(sdc_route_global, current_timestep, ego_pose)

        # --- 4. Assemble and Return Final Dictionary ---
        feature_dict = {
            'ego': ego_features, 'ego_mask': ego_mask,
            'agents': agent_features, 'agents_mask': agents_mask,
            'map': map_features, 'map_mask': map_mask,
            'traffic_lights': traffic_light_features, 'traffic_lights_mask': tl_mask,
            'goal': goal_features, 'goal_mask': goal_mask,
        }
        
        for key, tensor in feature_dict.items():
            if not np.all(np.isfinite(tensor)):
                raise ValueError(f"Invalid number (NaN/inf) in Waymax feature '{key}'.")
        
        return feature_dict

    def _unpack_waymax_data(self, state: "datatypes.SimulatorState") -> Tuple:
        """
        A private helper to translate a Waymax SimulatorState into the same
        simple NumPy array format produced by _unpack_npz_data.
        """
        ts = state.timestep.item()
        num_objects = state.num_objects
        sdc_idx_original = jnp.argmax(state.object_metadata.is_sdc).item()

        # --- 1. Reconstruct full (num_objects, 9) trajectory array for the current timestep ---
        states_at_t = np.stack([
            np.asarray(state.sim_trajectory.x[:, ts]),
            np.asarray(state.sim_trajectory.y[:, ts]),
            np.asarray(state.sim_trajectory.z[:, ts]),
            np.asarray(state.sim_trajectory.length[:, ts]),
            np.asarray(state.sim_trajectory.width[:, ts]),
            np.asarray(state.sim_trajectory.height[:, ts]),
            np.asarray(state.sim_trajectory.yaw[:, ts]),
            np.asarray(state.sim_trajectory.vel_x[:, ts]),
            np.asarray(state.sim_trajectory.vel_y[:, ts])
        ], axis=-1).astype(np.float32)
        
        valid_mask_at_t = np.asarray(state.sim_trajectory.valid[:, ts])

        # --- 2. Get SDC and OTHER agent states ---
        sdc_state_global_9d = states_at_t[sdc_idx_original]
        
        # Filter for agents that are valid AND are not the SDC
        other_agents_mask = valid_mask_at_t & (np.arange(num_objects) != sdc_idx_original)
        other_agents_global = states_at_t[other_agents_mask]
        other_agent_types = np.asarray(state.object_metadata.object_types)[other_agents_mask]
        
        # --- 3. Reconstruct map polylines ---
        # NOTE: Waymax does not provide all map feature types. It only provides
        # lane centerlines. This is a known limitation we must handle.
        rg_points = state.roadgraph_points
        lane_polylines_global = []
        
        # --- This utility must be loaded or available here ---
        type_mapping = get_waymax_to_parser_type_mapping() 
        
        unique_ids = np.unique(np.asarray(rg_points.ids))
        for feature_id in unique_ids:
            if feature_id == -1: continue
            
            mask = np.asarray(rg_points.ids == feature_id)
            points = np.stack([
                np.asarray(rg_points.x)[mask], 
                np.asarray(rg_points.y)[mask], 
                np.asarray(rg_points.z)[mask]
            ], axis=-1)
            
            waymax_type = np.asarray(rg_points.types)[mask][0]
            # --- THIS IS THE KEY TRANSLATION STEP ---
            parser_type = type_mapping.get(int(waymax_type), -1)

            # --- CORRECTED LOGIC ---
            # Now, we apply the SAME filtering logic as in _unpack_npz_data,
            # but we use the TRANSLATED parser_type.
            if parser_type in {0, 1, 2, 3}:
                lane_polylines_global.append(points)

        # --- 4. Reconstruct traffic light data ---
        tl_state = state.log_traffic_light
        traffic_lights_global = np.empty((0, 4), dtype=np.float32) # Default to empty

        if tl_state and tl_state.num_traffic_lights > 0:
            valid_at_t = np.asarray(tl_state.valid[:, ts])
            
            # Reconstruct the (num_lights, 4) array just like our parser does
            traffic_lights_global = np.stack([
                np.asarray(tl_state.lane_ids)[:, 0],
                np.asarray(tl_state.state)[:, ts],
                np.asarray(tl_state.x)[:, 0], # Stop points are static
                np.asarray(tl_state.y)[:, 0]
            ], axis=-1)[valid_at_t].astype(np.float32)

        return (
            sdc_state_global_9d, 
            other_agents_global, 
            other_agent_types, 
            lane_polylines_global,
            traffic_lights_global
        )