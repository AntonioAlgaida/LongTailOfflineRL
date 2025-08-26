# src/data_processing/validate_parser_output.py

# conda activate wwm
# python -m src.data_processing.validate_parser_output

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.patches import Patch, Polygon, Rectangle, Circle
# from waymo_open_dataset.protos import map_pb2 # ### NEW: Import to access enums ###

from matplotlib.animation import FuncAnimation
from glob import glob
import random

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config

# --- Configuration ---
# CONFIG = load_config()

def visualize_scenario(npz_path: str, output_dir: str = 'outputs/validation_videos'):
    """
    Loads a scenario from a DECOUPLED .npz file and creates an animated video.
    This script should be runnable in ANY environment with numpy and matplotlib.
    """
    print(f"Loading scenario from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # Load all data from the .npz file
    scenario_id = data['scenario_id']
    all_trajectories = data['all_agent_trajectories']
    valid_mask = data['valid_mask']
    object_types = data['object_types']
    sdc_track_index = data['sdc_track_index']
    dynamic_map_states = data['dynamic_map_states']
    map_polylines = data['map_polylines']
    map_polyline_types = data['map_polyline_types']
    num_agents, num_timesteps, _ = all_trajectories.shape

    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor('darkslategray') # Darker background for better contrast
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"Scenario Visualization: {scenario_id}")

    map_styles = {
        # Lane Centerlines
        0: {'color': 'gray', 'linewidth': 1.0, 'label': 'Lane (Undefined)'},
        1: {'color': 'deepskyblue', 'linewidth': 1.5, 'label': 'Lane (Freeway)'},
        2: {'color': 'dimgray', 'linewidth': 1.0, 'label': 'Lane (Surface Street)'},
        3: {'color': 'green', 'linewidth': 1.0, 'label': 'Lane (Bike Lane)'},
        
        # Road Lines
        10: {'color': 'white', 'linestyle': ':', 'linewidth': 0.5}, # Unknown
        11: {'color': 'white', 'linestyle': '--'},   # BROKEN_SINGLE_WHITE
        12: {'color': 'white', 'linestyle': '-'},    # SOLID_SINGLE_WHITE
        13: {'color': 'white', 'linestyle': '-'},    # SOLID_DOUBLE_WHITE
        14: {'color': 'yellow', 'linestyle': '--'},  # BROKEN_SINGLE_YELLOW
        15: {'color': 'yellow', 'linestyle': '--'},  # BROKEN_DOUBLE_YELLOW
        16: {'color': 'yellow', 'linestyle': '-'},   # SOLID_SINGLE_YELLOW
        17: {'color': 'yellow', 'linestyle': '-'},   # SOLID_DOUBLE_YELLOW
        18: {'color': 'yellow', 'linestyle': '--'},  # PASSING_DOUBLE_YELLOW
        
        # Road Edges
        20: {'color': 'moccasin', 'linewidth': 1.0}, # Unknown
        21: {'color': 'moccasin', 'linewidth': 1.5}, # ROAD_EDGE_BOUNDARY
        22: {'color': 'moccasin', 'linewidth': 1.5}, # ROAD_EDGE_MEDIAN
        
        # Other Features
        30: {'marker': 'o', 'color': 'red', 'markersize': 6, 'zorder': 3}, # stop_sign
        40: {'facecolor': 'whitesmoke', 'alpha': 0.3, 'zorder': 1}, # crosswalk
        50: {'facecolor': 'goldenrod', 'alpha': 0.5, 'zorder': 1},  # speed_bump
        60: {'facecolor': 'silver', 'alpha': 0.3, 'zorder': 1}      # driveway
    }
        
    # --- Plot Static Map Features ---
    for i, polyline in enumerate(map_polylines):
        polyline_type_idx = map_polyline_types[i]
        style = map_styles.get(polyline_type_idx)

        if not style:
            continue

        if 'marker' in style: # Handle point-based features like stop signs
             # For stop signs, the polyline is just a single point of its position
            ax.plot(polyline[0, 0], polyline[0, 1], **style)
        elif 'facecolor' in style: # Handle polygon features
            ax.add_patch(Polygon(polyline[:, :2], closed=True, **style))
        else: # Handle polyline features
            label = style.pop('label', None) 
            ax.plot(polyline[:, 0], polyline[:, 1], **style, label=label)
            if label: style['label'] = label # Add it back

    # --- Animation Objects ---
    agent_patches = []
    agent_colors = { 1: 'royalblue', 2: 'orange', 3: 'lime', 4: 'fuchsia' }
    
    for i in range(num_agents):
        color = 'cyan' if i == sdc_track_index else agent_colors.get(object_types[i], 'lightgray')
        patch = Rectangle((0, 0), 0, 0, color=color, zorder=2)
        ax.add_patch(patch)
        agent_patches.append(patch)
        
    ### NEW: Create plot artists for traffic lights ###
    # We will draw a small circle at the stop point of each traffic light
    traffic_light_artists = []
    if dynamic_map_states.shape[1] > 0: # Check if there are any traffic lights
        for j in range(dynamic_map_states.shape[1]):
            # Get the stop point from the first valid frame
            stop_point = None
            for t in range(num_timesteps):
                if dynamic_map_states[t, j, 2] != 0.0 or dynamic_map_states[t, j, 3] != 0.0:
                    stop_point = (dynamic_map_states[t, j, 2], dynamic_map_states[t, j, 3])
                    break
            if stop_point:
                circle = Circle(stop_point, radius=1.0, color='black', zorder=3, visible=False)
                ax.add_patch(circle)
                traffic_light_artists.append(circle)
            else:
                traffic_light_artists.append(None) # Placeholder if no stop point found


    # --- Legend (Now comprehensive) ---
    legend_elements = [
        # Agent Types
        Patch(facecolor='cyan', edgecolor='black', label='SDC'),
        Patch(facecolor='royalblue', edgecolor='black', label='Vehicle'),
        Patch(facecolor='orange', edgecolor='black', label='Pedestrian'),
        Patch(facecolor='lime', edgecolor='black', label='Cyclist'),
        
        # Line Types
        plt.Line2D([0], [0], color='gray', lw=2, label='Lane Centerline'),
        plt.Line2D([0], [0], color='white', linestyle='-', lw=1, label='Solid White Line'),
        plt.Line2D([0], [0], color='white', linestyle='--', lw=1, label='Broken White Line'),
        plt.Line2D([0], [0], color='yellow', linestyle='-', lw=1, label='Solid Yellow Line'),
        plt.Line2D([0], [0], color='yellow', linestyle='--', lw=1, label='Broken Yellow Line'),
        plt.Line2D([0], [0], color='moccasin', lw=2, label='Road Edge (Curb)'),

        # Area and Point Types
        Patch(facecolor='whitesmoke', alpha=0.3, label='Crosswalk'),
        Patch(facecolor='silver', alpha=0.3, label='Driveway'),
        Patch(facecolor='goldenrod', alpha=0.5, label='Speed Bump'),
        plt.Line2D([0], [0], marker='o', color='w', label='Stop Sign', markerfacecolor='red', markersize=8),

        # Dynamic Elements
        Patch(facecolor='green', label='Traffic Light (Go)'),
        Patch(facecolor='yellow', label='Traffic Light (Caution)'),
        Patch(facecolor='red', label='Traffic Light (Stop)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')


    # Animation update function
    def update(frame_idx):
        ax.set_title(f"Scenario: {scenario_id} (Timestep {frame_idx+1}/{num_timesteps})")
        sdc_state = all_trajectories[sdc_track_index, frame_idx]
        sdc_x, sdc_y = sdc_state[0], sdc_state[1]
        ax.set_xlim(sdc_x - 75, sdc_x + 75)
        ax.set_ylim(sdc_y - 75, sdc_y + 75)

        for i in range(num_agents):
            patch = agent_patches[i]
            if valid_mask[i, frame_idx]:
                state = all_trajectories[i, frame_idx]
                x, y, length, width, heading = state[0], state[1], state[3], state[4], state[6]
                patch.set_width(length)
                patch.set_height(width)
                corner_offset = np.array([-length / 2., -width / 2.])
                rot_matrix = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
                rotated_corner = rot_matrix @ corner_offset
                patch.set_xy((x + rotated_corner[0], y + rotated_corner[1]))
                patch.angle = np.rad2deg(heading)
                patch.set_visible(True)
            else:
                patch.set_visible(False)
                
        for j, artist in enumerate(traffic_light_artists):
            if artist is None: continue
            light_state_enum = dynamic_map_states[frame_idx, j, 1]
            if light_state_enum == 0:
                artist.set_visible(False)
            else:
                artist.set_visible(True)
                if light_state_enum in [3, 6]: artist.set_color('green')
                elif light_state_enum in [2, 5, 8]: artist.set_color('yellow')
                else: artist.set_color('red')

        return agent_patches + traffic_light_artists + [ax.title]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scenario_id}.mp4")
    
    print(f"Creating animation for timestep 0 to {num_timesteps-1}...")
    anim = FuncAnimation(fig, update, frames=num_timesteps, blit=False, interval=100)
    
    anim.save(output_path, writer='ffmpeg', fps=10)
    print(f"Animation saved to: {output_path}")
    plt.close(fig)

def main():
    """Finds a random scenario and visualizes it."""
    validation_npz_dir = os.path.join(CONFIG['data']['processed_npz_dir'], 'validation')
    all_npz_files = glob(os.path.join(validation_npz_dir, '*.npz'))

    if not all_npz_files:
        print(f"Error: No .npz files found in {validation_npz_dir}.")
        print("Please ensure the parser has run successfully.")
        return

    # Select a random file to validate
    random_file = random.choice(all_npz_files)
    
    visualize_scenario(random_file)

if __name__ == '__main__':
    # You may need to install ffmpeg:
    # On Ubuntu/WSL: sudo apt-get install ffmpeg
    # On macOS (with Homebrew): brew install ffmpeg
    main()