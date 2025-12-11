"""
CARLA Data Collection - Waypoint Following (Reliable!)
Uses manual waypoint navigation instead of buggy autopilot

Author: Kip Chemweno
Date: December 2025

This approach:
- Follows road waypoints directly (no autopilot crashes!)
- Naturally includes all turn types (left, right, straight)
- Multiple maps for variety
- Much more stable and reliable

Usage:
    python carla_data_collection_waypoints.py --samples 3000
"""

import os
import sys
import time
import random
import numpy as np
import cv2
from pathlib import Path
import json
import argparse
from collections import deque

try:
    import carla
    print(f"✓ CARLA module imported")
except ImportError:
    print("✗ CARLA not installed: pip install carla==0.9.16")
    sys.exit(1)


class WaypointFollowingCollector:
    """Reliable data collector using waypoint following."""

    WEATHER_PRESETS = {
        'clear': carla.WeatherParameters.ClearNoon,
        'cloudy': carla.WeatherParameters.CloudyNoon,
        'wet': carla.WeatherParameters.WetNoon,
        'sunset': carla.WeatherParameters.ClearSunset,
    }

    STABLE_MAPS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']

    def __init__(self, output_dir: str, host='localhost', port=2000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.output_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)

        self.host = host
        self.port = port
        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.camera = None

        self.data_records = []
        self.current_image = None
        self.frame_count = 0

        self.image_width = 1280
        self.image_height = 720

        # Waypoint following
        self.target_speed = 20.0  # km/h - slow and safe

    def connect(self):
        """Connect to CARLA."""
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(15.0)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            print(f"✓ Connected to CARLA")
            print(f"  Current map: {self.map.name}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def load_map(self, map_name: str):
        """Load a specific map."""
        print(f"\nLoading map: {map_name}...")
        try:
            self.world = self.client.load_world(map_name)
            time.sleep(5.0)  # Wait for map to load
            self.map = self.world.get_map()
            print(f"✓ Map loaded: {map_name}")
            return True
        except Exception as e:
            print(f"✗ Failed to load {map_name}: {e}")
            return False

    def setup_vehicle(self):
        """Spawn vehicle at a good location."""
        print("Spawning vehicle...")
        try:
            bp_library = self.world.get_blueprint_library()
            vehicle_bp = bp_library.filter('model3')[0]

            # Get spawn points and choose one on a main road
            spawn_points = self.map.get_spawn_points()

            if not spawn_points:
                print("✗ No spawn points available")
                return False

            # Try to find a good spawn point
            for attempt in range(10):
                spawn_point = random.choice(spawn_points)

                try:
                    self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    print(f"✓ Vehicle spawned")

                    # Give physics time to settle
                    time.sleep(0.5)
                    return True
                except:
                    if attempt < 9:
                        continue
                    else:
                        print("✗ Failed to spawn vehicle")
                        return False

            return False

        except Exception as e:
            print(f"✗ Spawn failed: {e}")
            return False

    def setup_camera(self):
        """Attach camera."""
        try:
            bp_library = self.world.get_blueprint_library()
            camera_bp = bp_library.find('sensor.camera.rgb')

            camera_bp.set_attribute('image_size_x', str(self.image_width))
            camera_bp.set_attribute('image_size_y', str(self.image_height))
            camera_bp.set_attribute('fov', '90')

            camera_transform = carla.Transform(
                carla.Location(x=1.5, z=1.4),
                carla.Rotation(pitch=-5.0)
            )

            self.camera = self.world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=self.vehicle
            )

            self.camera.listen(lambda image: self._process_image(image))

            print("✓ Camera attached")
            return True
        except Exception as e:
            print(f"✗ Camera setup failed: {e}")
            return False

    def _process_image(self, image):
        """Camera callback."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.image_height, self.image_width, 4))
        self.current_image = array[:, :, :3]

    def set_weather(self, weather_name: str):
        """Set weather."""
        weather = self.WEATHER_PRESETS.get(weather_name, carla.WeatherParameters.ClearNoon)
        self.world.set_weather(weather)
        return weather_name

    def get_steering_angle(self):
        """Get current steering angle."""
        control = self.vehicle.get_control()
        return control.steer * 0.436  # Convert to radians

    def calculate_steering_to_waypoint(self, waypoint):
        """
        Calculate steering needed to reach waypoint.

        Args:
            waypoint: Target CARLA waypoint

        Returns:
            Steering value in range [-1, 1]
        """
        # Get vehicle transform
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_forward = vehicle_transform.get_forward_vector()

        # Vector to waypoint
        target_location = waypoint.transform.location
        target_vector = target_location - vehicle_location
        target_vector = np.array([target_vector.x, target_vector.y, 0.0])

        # Normalize
        target_distance = np.linalg.norm(target_vector)
        if target_distance < 0.1:
            return 0.0

        target_vector = target_vector / target_distance

        # Vehicle forward vector
        forward_vector = np.array([vehicle_forward.x, vehicle_forward.y, 0.0])
        forward_vector = forward_vector / np.linalg.norm(forward_vector)

        # Calculate angle between vectors
        dot_product = np.clip(np.dot(forward_vector, target_vector), -1.0, 1.0)
        angle = np.arccos(dot_product)

        # Determine turn direction (cross product)
        cross = np.cross(forward_vector, target_vector)
        if cross[2] < 0:
            angle = -angle

        # Convert to steering (more aggressive for sharper turns)
        steering = np.clip(angle * 1.5, -1.0, 1.0)

        return steering

    def apply_control(self, steering, target_speed=None):
        """
        Apply vehicle control.

        Args:
            steering: Steering value [-1, 1]
            target_speed: Target speed in km/h (default: self.target_speed)
        """
        if target_speed is None:
            target_speed = self.target_speed

        # Get current velocity
        velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # Simple PID-like throttle control
        speed_error = target_speed - current_speed

        if speed_error > 0:
            throttle = np.clip(speed_error / 10.0, 0.0, 0.75)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-speed_error / 10.0, 0.0, 1.0)

        # Apply control
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=brake,
            hand_brake=False,
            manual_gear_shift=False
        )

        self.vehicle.apply_control(control)

    def collect_frame(self, current_weather: str, current_map: str):
        """Collect one frame."""
        if self.current_image is None:
            return False

        steering = self.get_steering_angle()

        image_filename = f"carla_{self.frame_count:06d}.jpg"
        image_path = self.images_dir / image_filename

        image_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image_bgr)

        self.data_records.append({
            'image_path': image_filename,
            'steering_angle': float(steering),
            'frame': self.frame_count,
            'map': current_map,
            'weather': current_weather
        })

        self.frame_count += 1
        return True

    def follow_waypoints_and_collect(self, num_samples: int, map_name: str, weather_list: list):
        """
        Drive by following waypoints and collect data.

        This is much more reliable than autopilot!
        """
        print(f"\n{'='*70}")
        print(f"COLLECTING ON {map_name}")
        print(f"{'='*70}")

        samples_per_weather = num_samples // len(weather_list)
        weather_idx = 0
        current_weather = self.set_weather(weather_list[weather_idx])

        # Wait for camera
        print("Waiting for camera...")
        for i in range(10):
            if self.current_image is not None:
                print("✓ Camera ready!\n")
                break
            time.sleep(0.5)

        if self.current_image is None:
            print("✗ Camera timeout")
            return 0

        start_count = len(self.data_records)
        last_update = time.time()
        last_waypoint_time = time.time()

        print(f"Driving and collecting... Target: {num_samples} samples\n")

        try:
            while len(self.data_records) - start_count < num_samples:
                # Change weather periodically
                samples_collected = len(self.data_records) - start_count
                if samples_collected > 0 and samples_collected % samples_per_weather == 0:
                    weather_idx = (weather_idx + 1) % len(weather_list)
                    current_weather = self.set_weather(weather_list[weather_idx])

                # Get vehicle location
                vehicle_location = self.vehicle.get_location()

                # Get next waypoint (ahead on the road)
                current_waypoint = self.map.get_waypoint(
                    vehicle_location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )

                if current_waypoint is None:
                    print("  ⚠ Lost waypoint, respawning...")
                    return len(self.data_records) - start_count

                # Get waypoint ahead (5 meters forward)
                next_waypoints = current_waypoint.next(5.0)

                if not next_waypoints:
                    # Try to get waypoint farther ahead
                    next_waypoints = current_waypoint.next(10.0)

                if next_waypoints:
                    target_waypoint = next_waypoints[0]

                    # Calculate steering
                    steering = self.calculate_steering_to_waypoint(target_waypoint)

                    # Apply control
                    self.apply_control(steering, target_speed=self.target_speed)

                    last_waypoint_time = time.time()
                else:
                    # No waypoint found, slow down
                    self.apply_control(0.0, target_speed=5.0)

                # Check if stuck (no waypoint for 10 seconds)
                if time.time() - last_waypoint_time > 10.0:
                    print("  ⚠ Vehicle stuck, respawning...")
                    return len(self.data_records) - start_count

                # Collect frame every frame (not skipping)
                self.collect_frame(current_weather, map_name)

                # Progress update
                if time.time() - last_update > 3.0:
                    progress = samples_collected / num_samples * 100
                    print(f"  Progress: {samples_collected}/{num_samples} ({progress:.1f}%) | Weather: {current_weather}")
                    last_update = time.time()

                # Small delay
                time.sleep(0.1)  # 10 Hz control loop, faster collection

        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")

        collected = len(self.data_records) - start_count
        print(f"\n✓ Collected {collected} samples on {map_name}")
        return collected

    def collect_diverse_dataset(self, total_samples: int, maps_to_use: list, weather_list: list):
        """Collect across multiple maps."""
        print(f"\n{'='*70}")
        print(f"WAYPOINT-FOLLOWING COLLECTION")
        print(f"{'='*70}")
        print(f"Target: {total_samples} samples")
        print(f"Maps: {len(maps_to_use)}")
        print(f"Weather: {', '.join(weather_list)}")
        print(f"Method: Waypoint following (no autopilot!)")
        print(f"{'='*70}\n")

        samples_per_map = total_samples // len(maps_to_use)

        for i, map_name in enumerate(maps_to_use):
            print(f"\n[Map {i + 1}/{len(maps_to_use)}]")

            # 1) Always cleanup old actors BEFORE loading a new map
            if self.vehicle is not None or self.camera is not None:
                self.cleanup_actors()
                self.current_image = None  # reset so camera wait works correctly
                time.sleep(1.0)

            # 2) Load the map (even for i == 0, so we’re explicit)
            if not self.load_map(map_name):
                print(f"⚠ Skipping {map_name}")
                continue

            # 3) Now spawn vehicle + camera in this (fresh) world
            if not self.setup_vehicle():
                continue
            if not self.setup_camera():
                continue

            # Collect on this map
            if i == len(maps_to_use) - 1:
                # Last map gets remaining samples
                remaining = total_samples - len(self.data_records)
                samples_this_map = remaining
            else:
                samples_this_map = samples_per_map

            collected = self.follow_waypoints_and_collect(
                samples_this_map,
                map_name,
                weather_list
            )

            print(f"✓ Total collected so far: {len(self.data_records)}/{total_samples}")

        print(f"\n{'='*70}")
        print(f"✓ COLLECTION COMPLETE!")
        print(f"{'='*70}")
        print(f"Total samples: {len(self.data_records)}")

        return self.data_records

    def save_data(self):
        """Save collected data."""
        print(f"\n{'='*70}")
        print("SAVING DATA")
        print(f"{'='*70}")

        import pandas as pd
        df = pd.DataFrame(self.data_records)

        csv_path = self.output_dir / 'carla_steering.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV saved: {csv_path}")

        metadata = {
            'total_samples': len(self.data_records),
            'image_width': self.image_width,
            'image_height': self.image_height,
            'collection_method': 'waypoint_following',
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'maps_used': list(df['map'].unique()),
            'weather_conditions': list(df['weather'].unique())
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Metadata saved: {metadata_path}")

        # Statistics
        steering = df['steering_angle'].values
        left = (steering < -0.05).sum()
        straight = ((steering >= -0.05) & (steering <= 0.05)).sum()
        right = (steering > 0.05).sum()

        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(df):,}")
        print(f"  Mean steering: {steering.mean():.4f} rad ({np.degrees(steering.mean()):.2f}°)")
        print(f"  Std steering: {steering.std():.4f} rad ({np.degrees(steering.std()):.2f}°)")
        print(f"  Min: {steering.min():.4f} rad ({np.degrees(steering.min()):.2f}°)")
        print(f"  Max: {steering.max():.4f} rad ({np.degrees(steering.max()):.2f}°)")
        print(f"\nSteering Distribution:")
        print(f"  Left turns:  {left:5d} ({left/len(steering)*100:5.1f}%)")
        print(f"  Straight:    {straight:5d} ({straight/len(steering)*100:5.1f}%)")
        print(f"  Right turns: {right:5d} ({right/len(steering)*100:5.1f}%)")

        print(f"\nSamples by Map:")
        for map_name in df['map'].unique():
            count = (df['map'] == map_name).sum()
            print(f"  {map_name:12s}: {count:5d} ({count/len(df)*100:5.1f}%)")

        print(f"{'='*70}\n")

    def cleanup_actors(self):
        """Cleanup actors."""
        if self.camera:
            self.camera.destroy()
            self.camera = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None

    def cleanup(self):
        """Final cleanup."""
        print("\nCleaning up...")
        self.cleanup_actors()
        print("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description='CARLA waypoint-following data collection')

    default_output = r"C:\Users\kchem\OneDrive\Documents\ECE 4424\Capstone\data\carla"

    parser.add_argument('--samples', type=int, default=3000)
    parser.add_argument('--output', type=str, default=default_output)
    parser.add_argument('--maps', nargs='+',
                        default=WaypointFollowingCollector.STABLE_MAPS,
                        help='Maps to use (stable ones recommended)')
    parser.add_argument('--weather', nargs='+',
                       default=['clear', 'cloudy', 'wet', 'sunset'])

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("CARLA WAYPOINT-FOLLOWING COLLECTION")
    print(f"{'='*70}")
    print(f"Target: {args.samples} samples")
    print(f"Maps: {', '.join(args.maps)}")
    print(f"Weather: {', '.join(args.weather)}")
    print(f"\n✓ Uses waypoint following (NO AUTOPILOT!)")
    print(f"✓ Much more reliable - no crashes!")
    print(f"✓ Naturally includes all turn types")
    print(f"{'='*70}\n")

    input("Press Enter when CARLA is ready...")

    collector = WaypointFollowingCollector(args.output)

    try:
        if not collector.connect():
            return

        collector.collect_diverse_dataset(
            args.samples,
            args.maps,
            args.weather
        )

        if len(collector.data_records) > 0:
            collector.save_data()

            print(f"\n{'='*70}")
            print("✓ SUCCESS!")
            print(f"{'='*70}")
            print(f"Collected: {len(collector.data_records)} samples")
            print(f"Location: {args.output}")
            print(f"\nNext steps:")
            print(f"  1. Train CARLA model:")
            print(f"     python train.py --data_source carla --epochs 30 --num_workers 0")
            print(f"  2. Create hybrid dataset:")
            print(f"     python create_hybrid_dataset.py")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()