# src/evaluation/custom_visualization.py

import numpy as np
import jax.numpy as jnp
import matplotlib.pylab as plt
from typing import Dict

from waymax import datatypes
from waymax.visualization import utils
# Import the original functions to reuse their robust logic
from waymax.visualization import plot_trajectory, plot_traffic_light_signals_as_points
import matplotlib.patches as patches # <-- Add this import
import jax

def plot_action_indicator(
    ax, 
    sdc_x: float, 
    sdc_y: float, 
    sdc_yaw: float, 
    sdc_l: float,
    action: np.ndarray
):
    """
    (V2 - Simplified) Overlays a graphic on the SDC to visualize its action.
    Accepts raw numeric values instead of a PyTree.
    """
    accel, yaw_rate = action[0], action[1]
    
    # --- 1. Draw Acceleration Bar ---
    start_point_local = np.array([sdc_l / 2, 0])
    bar_length = np.clip(accel / 5.0, -1, 1) * (sdc_l * 1.5)
    end_point_local = np.array([sdc_l / 2 + bar_length, 0])
    
    cos_yaw, sin_yaw = np.cos(sdc_yaw), np.sin(sdc_yaw)
    rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    start_point_global = start_point_local @ rot_matrix.T + np.array([sdc_x, sdc_y])
    end_point_global = end_point_local @ rot_matrix.T + np.array([sdc_x, sdc_y])
    
    bar_color = 'lime' if accel > 0 else 'red'
    ax.plot([start_point_global[0], end_point_global[0]], 
            [start_point_global[1], end_point_global[1]], 
            color=bar_color, linewidth=5, alpha=0.8, zorder=10)

    # --- 2. Draw Steering Arc ---
    if abs(yaw_rate) > 0.05: # Only draw for significant turns
        
        # The center of our arc is the SDC's current heading
        center_angle_deg = np.rad2deg(sdc_yaw)
        
        # The total width of the arc we want to draw
        arc_span_deg = abs(yaw_rate) * 180
        
        # Determine the start and end angles for the arc
        # If yaw_rate is positive (left turn), the arc is counter-clockwise
        # If yaw_rate is negative (right turn), the arc is clockwise
        if yaw_rate > 0: # Left Turn
            theta1 = center_angle_deg
            theta2 = center_angle_deg + arc_span_deg
        else: # Right Turn
            theta1 = center_angle_deg - arc_span_deg
            theta2 = center_angle_deg
            
        arc_radius = sdc_l * 1.5 # Make the arc slightly larger than the car
        
        arc = patches.Arc((sdc_x, sdc_y), arc_radius*2, arc_radius*2,
                          angle=0, theta1=theta1, theta2=theta2,
                          color='tab:blue', linewidth=3, zorder=10, alpha=0.8,
                          linestyle='-')
        ax.add_patch(arc)
        

def custom_plot_roadgraph_points_waymax_style(ax, rg_pts):
    """
    Custom roadgraph plotting function that mimics the original Waymax style
    but includes ALL road line types, using the official color palette.
    """
    if rg_pts.valid.sum() == 0:
        return

    valid_mask = np.asarray(rg_pts.valid)
    xy = np.asarray(rg_pts.xy)[valid_mask]
    rg_type = np.asarray(rg_pts.types)[valid_mask]
    
    # --- The Waymax Official ROAD_GRAPH_COLORS (from your provided source) ---
    ROAD_GRAPH_COLORS = {
        # Lanes are a light grey
        0: np.array([230, 230, 230]) / 255.0, # LANE_UNDEFINED (Added)
        1: np.array([230, 230, 230]) / 255.0, # 'LaneCenter-Freeway',
        2: np.array([230, 230, 230]) / 255.0, # 'LaneCenter-SurfaceStreet',
        3: np.array([230, 230, 230]) / 255.0, # 'LaneCenter-BikeLane',
        
        # Road Lines have distinct colors
        5: np.array([240, 240, 240]) / 255.0, # Unknown (off-white)
        6: np.array([255, 255, 255]) / 255.0, # BrokenSingleWhite
        7: np.array([255, 255, 255]) / 255.0, # SolidSingleWhite
        8: np.array([255, 255, 255]) / 255.0, # SolidDoubleWhite
        
        # --- Yellow Road Lines are now YELLOW --
        9: np.array([255, 255, 0]) / 255.0,   # 'RoadLine-BrokenSingleYellow',
        10: np.array([255, 255, 0]) / 255.0,  # 'RoadLine-BrokenDoubleYellow'
        13: np.array([255, 255, 0]) / 255.0,  # 'RoadLine-PassingDoubleYellow',
        
        # A slightly darker, more "golden" yellow for solid lines
        11: np.array([255, 215, 0]) / 255.0,  # 'RoadLine-SolidSingleYellow', (Gold)
        12: np.array([255, 215, 0]) / 255.0,  # 'RoadLine-SolidDoubleYellow',
        
        13: np.array([120, 120, 120]) / 255.0,# 'RoadLine-PassingDoubleYellow',
        # Road Edges are a darker grey
        15: np.array([80, 80, 80]) / 255.0,   # 'RoadEdgeBoundary',
        16: np.array([80, 80, 80]) / 255.0,   # 'RoadEdgeMedian',
        # Other Features
        17: np.array([255, 0, 0]) / 255.0,    # 'StopSign', (Red)
        18: np.array([200, 200, 200]) / 255.0,# 'Crosswalk', (Light Grey)
        19: np.array([200, 200, 200]) / 255.0, # 'SpeedBump',
    }
    _RoadGraphDefaultColor = (0.9, 0.9, 0.9)

    for curr_type in np.unique(rg_type):
        p1 = xy[rg_type == curr_type]
        rg_color = ROAD_GRAPH_COLORS.get(int(curr_type), _RoadGraphDefaultColor)
        
        # Stop signs are a single point, so they need a marker
        if curr_type == 17: # STOP_SIGN
             ax.plot(p1[:, 0], p1[:, 1], 's', color=rg_color, ms=6) # 's' for square
        else:
             ax.plot(p1[:, 0], p1[:, 1], '.', color=rg_color, ms=2)

def custom_plot_simulator_state(
    state: datatypes.SimulatorState,
    use_log_traj: bool = False,
    action: np.ndarray = None,
    final_goal_point: np.ndarray = None, # <-- NEW
    immediate_path_points: np.ndarray = None # <-- NEW
) -> np.ndarray:
    
    """
    (V4 - With Action Viz) Custom simulator state plotting function.
    """
    viz_config = utils.VizConfig()
    fig, ax = utils.init_fig_ax(viz_config)
    
    ax.set_facecolor('white')
    
    traj_to_plot = state.log_trajectory if use_log_traj else state.sim_trajectory
    sdc_idx = jnp.argmax(state.object_metadata.is_sdc).item()
    is_controlled = state.object_metadata.is_sdc
    ts = state.timestep.item()
    
    # Plot trajectory and map (no change)
    plot_trajectory(ax, traj_to_plot, is_controlled, time_idx=ts, add_label=False)
    custom_plot_roadgraph_points_waymax_style(ax, state.roadgraph_points)
    plot_traffic_light_signals_as_points(ax, state.log_traffic_light, ts)
    
    # --- NEW: Plot the action indicator if provided ---
    if action is not None:
        # Extract the raw SDC state values directly
        sdc_x = traj_to_plot.x[sdc_idx, ts]
        sdc_y = traj_to_plot.y[sdc_idx, ts]
        sdc_yaw = traj_to_plot.yaw[sdc_idx, ts]
        sdc_l = traj_to_plot.length[sdc_idx, ts]
        
        # Call the simplified helper function
        plot_action_indicator(ax, sdc_x, sdc_y, sdc_yaw, sdc_l, action)

    if final_goal_point is not None:
        ax.plot(
            final_goal_point[0], # x
            final_goal_point[1], # y
            '*',                 # Marker style: star
            color='magenta',
            markersize=20,       # Make it large and visible
            markeredgecolor='white',
            label='Final Destination',
            zorder=12
        )
    
    if immediate_path_points is not None and immediate_path_points.shape[0] > 0:
        num_path_points = immediate_path_points.shape[0]
        
        # Create a gradient of sizes: larger dots for closer points
        sizes = np.linspace(12, 2, num_path_points)
        
        # Create a gradient of colors: brighter for closer points
        # Using a colormap is the best way to do this
        cmap = plt.get_cmap('winter') # A good magenta-yellow colormap
        colors = cmap(np.linspace(0, 1, num_path_points))
        
        # Use scatter plot for variable size and color
        ax.scatter(
            immediate_path_points[:, 0], # x coordinates
            immediate_path_points[:, 1], # y coordinates
            s=sizes,                     # `s` is for size in scatter
            c=colors,                    # `c` is for color
            label='Immediate Path',
            zorder=5
        )
        
        # Add a subtle connecting line
        ax.plot(
            immediate_path_points[:, 0],
            immediate_path_points[:, 1],
            '-',
            color='magenta',
            linewidth=1.0,
            alpha=0.3, # Make the line very faint
            zorder=10
        )
        
    current_xy = traj_to_plot.xy[:, state.timestep.item(), :]
    xy = current_xy[is_controlled]
    origin_x, origin_y = xy[0, :2]
    ax.axis((
      origin_x - viz_config.back_x, origin_x + viz_config.front_x,
      origin_y - viz_config.back_y, origin_y + viz_config.front_y,
    ))

    return utils.img_from_fig(fig)