# src/evaluation/custom_metrics.py

import numpy as np
import jax.numpy as jnp
from scipy.spatial import cKDTree
from waymax import datatypes
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils import geometry # Assuming you have geometry helpers

# --- Helper function for Bounding Box Corners ---
def _get_bounding_box_corners(x, y, length, width, yaw) -> np.ndarray:
    """Calculates the global coordinates of the four corners of a bounding box."""
    half_l, half_w = length / 2, width / 2
    corners_local = np.array([
        [half_l, half_w],   # Front-right
        [-half_l, half_w],  # Rear-right
        [-half_l, -half_w], # Rear-left
        [half_l, -half_w]   # Front-left
    ])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return corners_local @ rot_matrix.T + np.array([x, y])

# --- Main Metrics Calculator Class ---

class CustomMetricsCalculator:
    """
    A stateful class to calculate a rich suite of metrics incrementally
    during a Waymax simulation rollout.
    """
    def __init__(self, initial_state: datatypes.SimulatorState, sdc_route_global: np.ndarray):
        self.sdc_idx = jnp.argmax(initial_state.object_metadata.is_sdc).item()
        self.sdc_route = sdc_route_global
        
        # --- Initialize metric states ---
        self.is_collided = False
        self.is_offroad = False
        self.red_light_violation = False
        
        # Lists to store timeseries data
        self._min_ttc_series = []
        self._route_adherence_errors = []
        self._jerks = []
        self._lat_accels = []
        
        # Store initial state for progression and comfort calculations
        ts = initial_state.timestep.item()
        self._initial_sdc_pos = self.sdc_route[ts, :2]
        self._prev_sdc_speed = np.linalg.norm(self.sdc_route[ts, 7:9])
        self._prev_sdc_accel = 0.0

    def step(self, state: datatypes.SimulatorState):
        """
        Updates all metrics with the new state from the simulator.
        This should be called at every step of the simulation loop.
        """
        ts = state.timestep.item()
        sdc_state = state.sim_trajectory
        sdc_idx = self.sdc_idx
        
        # --- Update Safety Metrics ---
        if not self.is_collided: self.is_collided = self._check_collisions(state, ts)
        if not self.is_offroad: self.is_offroad = self._check_offroad(state, sdc_idx, ts)
        
        ttc = self._calculate_min_ttc(state, ts)
        if ttc < 999.0: self._min_ttc_series.append(ttc)
            
        # --- Update Comfort Metrics ---
        current_speed = np.linalg.norm([sdc_state.vel_x[self.sdc_idx, ts], sdc_state.vel_y[self.sdc_idx, ts]])
        accel = (current_speed - self._prev_sdc_speed) / 0.1
        jerk = (accel - self._prev_sdc_accel) / 0.1
        self._jerks.append(abs(jerk))
        
        # Lateral accel approx: v * omega (speed * yaw_rate)
        # Yaw rate needs to be calculated from yaw
        yaw_t = sdc_state.yaw[self.sdc_idx, ts]
        yaw_t_minus_1 = sdc_state.yaw[self.sdc_idx, ts - 1] if ts > 0 else yaw_t
        delta_yaw = (yaw_t - yaw_t_minus_1 + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = delta_yaw / 0.1
        self._lat_accels.append(abs(current_speed * yaw_rate))
        
        self._prev_sdc_speed = current_speed
        self._prev_sdc_accel = accel
        
        # --- Update Path Adherence Metric ---
        sdc_pos = np.array([sdc_state.x[self.sdc_idx, ts], sdc_state.y[self.sdc_idx, ts]])
        min_dist_to_route = self._calculate_min_dist_to_route(sdc_pos)
        self._route_adherence_errors.append(min_dist_to_route)

        # --- Update Rule Compliance ---
        if not self.red_light_violation: self.red_light_violation = self._check_red_light(state, ts)

    def finalize(self, final_state: datatypes.SimulatorState) -> dict:
        """
        (Corrected Version) Calculates and returns the final summary dictionary
        with JSON-serializable types.
        """
        ts = final_state.timestep.item()
        
        # --- Finalize Goal Achievement Metrics ---
        final_sdc_pos = np.array([final_state.sim_trajectory.x[self.sdc_idx, ts],
                                  final_state.sim_trajectory.y[self.sdc_idx, ts]])
        final_goal_pos = self.sdc_route[-1, :2]
        
        sdc_progression = np.linalg.norm(final_sdc_pos - self._initial_sdc_pos)
        final_distance_to_goal = np.linalg.norm(final_sdc_pos - final_goal_pos)

        # --- Aggregate Timeseries Metrics ---
        # Use a default value for max/min in case the list is empty
        min_ttc = min(self._min_ttc_series) if self._min_ttc_series else 999.0
        max_jerk = max(self._jerks) if self._jerks else 0.0
        max_lat_accel = max(self._lat_accels) if self._lat_accels else 0.0
        avg_route_adherence = np.mean(self._route_adherence_errors) if self._route_adherence_errors else 0.0
        
        # --- THIS IS THE CRITICAL FIX ---
        # Explicitly cast all numeric types to standard Python floats or bools
        # before returning the dictionary.
        return {
            'collided': bool(self.is_collided),
            'offroad': bool(self.is_offroad),
            'min_ttc': float(min_ttc),
            'sdc_progression': float(sdc_progression),
            'final_distance_to_goal': float(final_distance_to_goal),
            'route_adherence_error': float(avg_route_adherence),
            'max_jerk': float(max_jerk),
            'max_lateral_acceleration': float(max_lat_accel),
            'red_light_violation': bool(self.red_light_violation),
        }

    # --- Private Helper Methods for Metric Calculations ---
    
    def _check_sat_overlap(self, corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
        """
        Checks for overlap between two Oriented Bounding Boxes (OBBs) using the
        Separating Axis Theorem (SAT).
        """
        # Get the four potential separating axes (the normals of the box edges)
        # Edge vectors for box A
        edge1_a = corners_a[1] - corners_a[0]
        edge2_a = corners_a[3] - corners_a[0]
        # Edge vectors for box B
        edge1_b = corners_b[1] - corners_b[0]
        edge2_b = corners_b[3] - corners_b[0]
        
        # The axes are perpendicular to the edges
        axes = [
            np.array([edge1_a[1], -edge1_a[0]]),
            np.array([edge2_a[1], -edge2_a[0]]),
            np.array([edge1_b[1], -edge1_b[0]]),
            np.array([edge2_b[1], -edge2_b[0]])
        ]
        
        for axis in axes:
            # Project all corners of both boxes onto the axis
            proj_a = corners_a @ axis
            proj_b = corners_b @ axis
            
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            
            # Check for separation
            if max_a < min_b or max_b < min_a:
                return False # Found a separating axis, no collision

        return True # No separating axis found, collision detected

    def _check_collisions(self, state, ts) -> bool:
        """
        (V3 - Optimized OBB) Checks for collision using a two-phase approach:
        1. Broad Phase: Quickly filters for agents within a proximity radius.
        2. Narrow Phase: Performs the expensive OBB (SAT) check only on candidates.
        """
        # --- Get SDC's state and corner points ---
        sdc_state = state.sim_trajectory
        sdc_pos = np.array([sdc_state.x[self.sdc_idx, ts], sdc_state.y[self.sdc_idx, ts]])
        sdc_corners = _get_bounding_box_corners(
            sdc_pos[0], sdc_pos[1],
            sdc_state.length[self.sdc_idx, ts], sdc_state.width[self.sdc_idx, ts],
            sdc_state.yaw[self.sdc_idx, ts]
        )
        
        # --- Broad Phase: Proximity-based filtering ---
        
        # Get positions of all other valid objects
        valid_mask = np.asarray(sdc_state.valid[:, ts])
        other_agents_mask = valid_mask & (np.arange(state.num_objects) != self.sdc_idx)
        
        if not np.any(other_agents_mask):
            return False # No other valid agents to collide with

        other_agent_indices = np.where(other_agents_mask)[0]
        other_agent_positions = np.stack([
            np.asarray(sdc_state.x[other_agents_mask, ts]),
            np.asarray(sdc_state.y[other_agents_mask, ts])
        ], axis=-1)
        
        # Calculate distances from SDC to all other agents
        distances = np.linalg.norm(other_agent_positions - sdc_pos, axis=1)
        
        # Define a generous radius. A collision is impossible outside this range.
        proximity_radius = 10.0 # meters
        
        # Get the indices of agents that are close enough to be collision candidates
        candidate_mask = distances < proximity_radius
        candidate_indices = other_agent_indices[candidate_mask]
        
        # --- Narrow Phase: OBB check only on candidates ---
        for i in candidate_indices:
            # Get the corner points of the candidate object's OBB
            obj_corners = _get_bounding_box_corners(
                sdc_state.x[i, ts], sdc_state.y[i, ts],
                sdc_state.length[i, ts], sdc_state.width[i, ts],
                sdc_state.yaw[i, ts]
            )
            
            # Perform the accurate OBB collision check
            if self._check_sat_overlap(sdc_corners, obj_corners):
                return True # Collision detected
                
        return False
            

    def _check_offroad(self, state, sdc_idx: int, ts: int) -> bool:
        """
        (V2 - Researcher Grade) Checks if the SDC is off-road at a specific timestep.

        Uses a two-stage check for robustness and accuracy:
        1. A coarse check for large deviations from any lane.
        2. A fine check for proximity to the nearest physical or legal road boundary.
        """
        sdc_pos = jnp.stack([state.sim_trajectory.x[sdc_idx, ts], 
                            state.sim_trajectory.y[sdc_idx, ts]], axis=-1)
        
        rg_points = state.roadgraph_points
        rg_types = np.asarray(rg_points.types)
        rg_xy = np.asarray(rg_points.xy)
        valid_mask = np.asarray(rg_points.valid)
        
        # --- 1. Coarse Check: Proximity to any Lane Centerline ---
        lane_mask = valid_mask & (rg_types < 4) # Lane types are 0, 1, 2, 3
        if not np.any(lane_mask):
            return True # If there are no lanes in the map data, we are "lost" -> off-road

        lane_points = rg_xy[lane_mask]
        
        # Use KD-Tree for a fast nearest-neighbor search
        lane_kdtree = cKDTree(lane_points)
        min_dist_to_lane, _ = lane_kdtree.query(np.asarray(sdc_pos), k=1)
        
        # If the SDC is very far from any lane, it's definitively off-road.
        if min_dist_to_lane > 5.0: # 5 meters is a large buffer
            return True

        # --- 2. Fine Check: Proximity to nearest Road Boundary ---
        # Your insight: ROAD_EDGE is the key. Let's also include solid lines.
        # Waymax Enums: ROAD_EDGE_BOUNDARY=15, ROAD_EDGE_MEDIAN=16
        # SOLID_SINGLE_WHITE=7, SOLID_DOUBLE_WHITE=8
        # SOLID_SINGLE_YELLOW=11, SOLID_DOUBLE_YELLOW=12
        boundary_types = {15, 16, 7, 8, 11, 12}
        
        boundary_mask = valid_mask & np.isin(rg_types, list(boundary_types))
        
        # If there are no boundaries nearby, we can assume we're on-road (e.g., in a parking lot)
        if not np.any(boundary_mask):
            return False

        boundary_points = rg_xy[boundary_mask]
        boundary_kdtree = cKDTree(boundary_points)
        min_dist_to_boundary, _ = boundary_kdtree.query(np.asarray(sdc_pos), k=1)
        
        # If the SDC is closer than a small threshold to a hard boundary, it's off-road.
        # 0.5 meters is a reasonable threshold to account for vehicle width.
        # print(f"Min distance to boundary: {min_dist_to_boundary:.2f} m")
        if min_dist_to_boundary < 0.5:
            return True
        
        lane_mask = valid_mask & (rg_types < 4)
        if not np.any(lane_mask):
            return True # No lanes means we are off-road.
        
        # We can check if any corner is further than a threshold from the nearest lane point.
        sdc_state = state.sim_trajectory
        sdc_x, sdc_y = sdc_state.x[sdc_idx, ts], sdc_state.y[sdc_idx, ts]
        sdc_l, sdc_w = sdc_state.length[sdc_idx, ts], sdc_state.width[sdc_idx, ts]
        sdc_yaw = sdc_state.yaw[sdc_idx, ts]
        sdc_corners = _get_bounding_box_corners(sdc_x, sdc_y, sdc_l, sdc_w, sdc_yaw)

        lane_kdtree = cKDTree(lane_points)
        corner_distances_to_lane, _ = lane_kdtree.query(sdc_corners, k=1)
        
        # If the corner that is furthest from a lane is more than ~half a lane width away,
        # it's likely off-road. A standard lane is ~3.5m wide.
        max_corner_dist_to_lane = np.max(corner_distances_to_lane)
        
        # print(f"Max corner distance to lane: {max_corner_dist_to_lane:.2f} m")
        if max_corner_dist_to_lane > 2.5: # 2.5m is a very generous buffer
            return True

        return False

    def _calculate_min_ttc(self, state: datatypes.SimulatorState, ts: int) -> float:
        """
        (V1 - Circular Approximation) Calculates the minimum Time-to-Collision (TTC)
        between the SDC and any other valid agent.
        
        This uses a circular approximation for each vehicle and solves a quadratic
        equation to find the time of first intersection, assuming constant velocity.
        """
        sdc_state = state.sim_trajectory
        
        # --- 1. Get SDC's current state ---
        sdc_pos = np.array([sdc_state.x[self.sdc_idx, ts], sdc_state.y[self.sdc_idx, ts]])
        sdc_vel = np.array([sdc_state.vel_x[self.sdc_idx, ts], sdc_state.vel_y[self.sdc_idx, ts]])
        # Use the diagonal of the bounding box as a conservative radius estimate
        sdc_radius = np.sqrt(sdc_state.length[self.sdc_idx, ts]**2 + sdc_state.width[self.sdc_idx, ts]**2) / 2.0

        # --- 2. Get states of all OTHER valid agents ---
        valid_mask = np.asarray(sdc_state.valid[:, ts])
        other_agents_mask = valid_mask & (np.arange(state.num_objects) != self.sdc_idx)
        
        if not np.any(other_agents_mask):
            return 999.0

        other_indices = np.where(other_agents_mask)[0]
        
        obj_pos = np.stack([np.asarray(sdc_state.x[other_indices, ts]), 
                            np.asarray(sdc_state.y[other_indices, ts])], axis=-1)
        obj_vel = np.stack([np.asarray(sdc_state.vel_x[other_indices, ts]), 
                            np.asarray(sdc_state.vel_y[other_indices, ts])], axis=-1)
        obj_radii = np.sqrt(np.asarray(sdc_state.length[other_indices, ts])**2 + 
                            np.asarray(sdc_state.width[other_indices, ts])**2) / 2.0

        # --- 3. Vectorized TTC Calculation ---
        
        # Relative kinematics
        relative_pos = obj_pos - sdc_pos
        relative_vel = obj_vel - sdc_vel
        
        # Sum of radii for collision check
        sum_radii = sdc_radius + obj_radii
        
        # Coefficients of the quadratic equation: a*t^2 + b*t + c = 0
        # a = V_rel . V_rel
        # b = 2 * (P_rel . V_rel)
        # c = (P_rel . P_rel) - R_sum^2
        a = np.sum(relative_vel * relative_vel, axis=1)
        b = 2 * np.sum(relative_pos * relative_vel, axis=1)
        c = np.sum(relative_pos * relative_pos, axis=1) - sum_radii**2
        
        # Calculate the discriminant: b^2 - 4ac
        discriminant = b**2 - 4 * a * c
        
        # --- 4. Solve for TTC ---
        
        # We only care about cases where a collision is possible (discriminant >= 0)
        # and the vehicles are actually moving towards each other (b < 0).
        collision_possible_mask = (discriminant >= 0) & (b < 0)
        
        if not np.any(collision_possible_mask):
            return 999.0
            
        # Filter for the relevant components
        a_f = a[collision_possible_mask]
        b_f = b[collision_possible_mask]
        discriminant_f = discriminant[collision_possible_mask]
        
        # The solution for t is (-b - sqrt(discriminant)) / (2a)
        # We take the negative root because it's the *first* time of intersection.
        ttc_values = (-b_f - np.sqrt(discriminant_f)) / (2 * a_f)
        
        # The final TTC is the minimum positive TTC among all candidates
        min_ttc = np.min(ttc_values)
        
        return float(min_ttc) if min_ttc > 0 else 999.0

    def _calculate_min_dist_to_route(self, sdc_pos) -> float:
        # Find perpendicular distance from sdc_pos to the sdc_route polyline
        min_dist = np.inf
        for i in range(len(self.sdc_route) - 1):
            p1 = self.sdc_route[i, :2]
            p2 = self.sdc_route[i+1, :2]
            dist = geometry.perpendicular_distance_point_to_line_segment(sdc_pos, p1, p2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _check_red_light(self, state: datatypes.SimulatorState, ts: int) -> bool:
        """
        (V2 - Lane Aware) Checks for red light violations by first identifying
        the SDC's current lane and then checking the state of the traffic light
        that specifically controls that lane.
        """
        tl_state = state.log_traffic_light
        if not (tl_state and tl_state.num_traffic_lights > 0 and np.any(np.asarray(tl_state.valid[:, ts]))):
            return False

        # --- 1. Get SDC's current position and heading vector ---
        sdc_pos = np.array([state.sim_trajectory.x[self.sdc_idx, ts], 
                            state.sim_trajectory.y[self.sdc_idx, ts]])
        sdc_yaw = state.sim_trajectory.yaw[self.sdc_idx, ts]
        sdc_heading_vec = np.array([np.cos(sdc_yaw), np.sin(sdc_yaw)])
        
        # --- 2. Find the SDC's Most Probable Current Lane ID ---
        rg_points = state.roadgraph_points
        rg_types = np.asarray(rg_points.types)
        rg_xy = np.asarray(rg_points.xy)
        rg_ids = np.asarray(rg_points.ids)
        valid_mask = np.asarray(rg_points.valid)
        
        lane_mask = valid_mask & (rg_types < 4)
        if not np.any(lane_mask): return False

        lane_points_xy = rg_xy[lane_mask]
        lane_point_ids = rg_ids[lane_mask]
        
        # Broad phase: Find the 5 nearest lane points to the SDC
        lane_kdtree = cKDTree(lane_points_xy)
        distances, indices = lane_kdtree.query(sdc_pos, k=5)
        
        # Get the unique lane IDs corresponding to these nearby points
        candidate_lane_ids = np.unique(lane_point_ids[indices])
        
        best_lane_id = -1
        min_score = np.inf

        # Narrow phase: Score each candidate lane
        for lane_id in candidate_lane_ids:
            # Get all points for this specific lane
            lane_poly = rg_xy[rg_ids == lane_id]
            
            # Find the closest segment on this lane's polyline
            min_dist_to_segment = np.inf
            segment_direction = None
            for j in range(len(lane_poly) - 1):
                p1, p2 = lane_poly[j, :2], lane_poly[j+1, :2]
                dist = geometry.perpendicular_distance_point_to_line_segment(sdc_pos, p1, p2)
                if dist < min_dist_to_segment:
                    min_dist_to_segment = dist
                    vec = p2 - p1
                    norm = np.linalg.norm(vec)
                    if norm > 1e-3:
                        segment_direction = vec / norm

            if segment_direction is None: continue

            # Score based on heading alignment (dot product)
            # Cosine similarity: 1 for parallel, -1 for opposite
            heading_similarity = np.dot(sdc_heading_vec, segment_direction)
            
            # We want to penalize lanes going in the opposite direction
            if heading_similarity < 0:
                continue # This lane is going the wrong way, discard it

            # Final score: combination of distance and heading difference
            # Lower score is better. We penalize misalignment.
            # (1 - heading_similarity) is 0 for perfect alignment, 2 for opposite
            score = min_dist_to_segment + (1 - heading_similarity) * 2.0 # Weight heading diff
            
            if score < min_score:
                min_score = score
                best_lane_id = lane_id

        if best_lane_id == -1:
            return False # Could not confidently determine a lane

        # --- 3. Check the Controlling Traffic Light for the BEST lane ---
        valid_at_t = np.asarray(tl_state.valid[:, ts])
        controlling_lane_ids = np.asarray(tl_state.lane_ids)[:, 0][valid_at_t]
        light_states = np.asarray(tl_state.state)[:, ts][valid_at_t]
        stop_points = np.stack([
            np.asarray(tl_state.x)[:, 0], np.asarray(tl_state.y)[:, 0]
        ], axis=-1)[valid_at_t]
        
        for i, controlled_lane_id in enumerate(controlling_lane_ids):
            if controlled_lane_id == best_lane_id:
                relevant_light_state = light_states[i]
                is_red = int(relevant_light_state) in {1, 4, 7}
                
                if is_red:
                    dist_to_stop = np.linalg.norm(sdc_pos - stop_points[i])
                    sdc_speed = np.linalg.norm([state.sim_trajectory.vel_x[self.sdc_idx, ts],
                                                state.sim_trajectory.vel_y[self.sdc_idx, ts]])
                    
                    # Violation: light is red, we are close to the line, AND we are moving
                    if dist_to_stop < 3.0 and sdc_speed > 1.0:
                        return True
                break # Found the relevant light
                
        return False