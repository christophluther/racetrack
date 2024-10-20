from itertools import repeat, product
from typing import Tuple, Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.controller import ControlledVehicle


class RacetrackEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20]
                    },
                    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                    "grid_step": [5, 5],
                    "absolute": False
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 1000,
                "screen_height": 1000,
                "centering_position": [0.5, 0.5],
                "new_reward": True, # CL: bool to toggle new reward fct (line 101)
                "terminate_off_road": True, # CL: terminate if car goes off-road
                "hit": False,   # CL: allows 'hits' without crash (i.e., car acts as 'ghost car')
                "restrict_init_collision": 20,  # CL: Change the distance to prevent init collision
                "speed_limits": [None, 10, 10, 10, 10, 10, 10, 10, 10],  # CL: Speed limits for road segments
            }
        )
        return config

    # added by CL
    def crash_or_hit(self):
        if self.config["hit"]:
            # return True if vehicle hit another car (assumes same position in grid), allows to pass through other cars
            return self.vehicle.hit
        else:
            # return True if the vehicle is crashed
            return self.vehicle.crashed

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        # CL: Above: Original reward, below alternative
        if self.config["new_reward"]:
            # CL: New reward, negative for crash or hit ("ghost collision"), else original
            reward = ((1 - self.crash_or_hit()) * reward + self.crash_or_hit() * self.config[
                "collision_reward"] - self._is_terminated())
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2),
            "action_reward": np.linalg.norm(action),  # penalise actions
            # CL: Allow for "ghost collisions"
            "collision_reward": self.crash_or_hit(),  # is per default equivalent to self.vehicle.crashed
            "on_road_reward": self.vehicle.on_road,  # bool
        }

    # CL: Terminate if crashed (except "ghost collisions") and optionally if off-road
    def _is_terminated(self) -> bool:
        if self.config["terminate_off_road"]:
            while self.vehicle.on_road:
                if self.config["hit"]:
                    return False
                else:
                    return self.vehicle.crashed
            return True # return True if vehicle is not on road
        else:
            if self.config["hit"]:
                return False
            else:
                return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    # original road network by @supperted825
    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = self.config["speed_limits"]

        # Initialise First Lane
        lane = StraightLane(
            [42, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, 5],
                [100, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2],
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + 5,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # 3 - Vertical Straight
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [120, -20],
                [120, -30],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [125, -20],
                [125, -30],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[4],
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2 + 5,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[5],
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3,
                np.deg2rad(0),
                np.deg2rad(137),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[5],
            ),
        )

        # 6 - Slant
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [55.7, -15.7],
                [35.7, -35.7],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [59.3934, -19.2],
                [39.3934, -39.2],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(315),
                np.deg2rad(170),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(315),
                np.deg2rad(165),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(170),
                np.deg2rad(56),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(170),
                np.deg2rad(58),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(240),
                np.deg2rad(270),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[8],
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(238),
                np.deg2rad(268),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6+rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6+rng.uniform(high=3))
            # Prevent early collisions # CL: is now a parameter in configuration
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < self.config["restrict_init_collision"]:
                    break
            else:
                self.road.vehicles.append(vehicle)


class RacetrackEnvLarge(RacetrackEnv):
    """
    A variant of racetrack-v0 with wider lanes.
    """

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        # CL: Config speed limits manually, default is [None, 25, 15, 25, 15, 25, 15, 25, 15]
        speedlimits = self.config["speed_limits"]

        # Initialise First Lane
        lane = StraightLane([44, 5.1], [131, 5.1], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=10,
                            speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b",
                     StraightLane([44, 15.1], [131, 15.1], line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  width=10, speed_limit=speedlimits[1]))
        # 10 instead of five moved the lane down

        # 2 - Circular Arc #1
        center1 = [130.1, -14.8]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(91), np.deg2rad(4), width=10,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1 + 10, np.deg2rad(91), np.deg2rad(4), width=10,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([150, -13.8], [150, -51.9],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=10,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([160, -11.6], [160, -51.6],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=10,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [135.1, -50]
        radii2 = 15
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-180), width=10,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2 + 10, np.deg2rad(0), np.deg2rad(-180), width=10,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [95.5, -52]
        radii3 = 15
        net.add_lane("e", "f",
                     CircularLane(center3, radii3 + 10, np.deg2rad(0), np.deg2rad(136), width=10,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=10,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 6 - Slant
        net.add_lane("f", "g", StraightLane([77.7, -34.7], [32.7, -68.3],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=10,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([86.3934, -40.2], [37.3934, -77.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=10,
                                            speed_limit=speedlimits[6]))

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -48.1]
        radii4 = 25
        net.add_lane("g", "h",
                     CircularLane(center4, radii4, np.deg2rad(312), np.deg2rad(180), width=10,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4 + 10, np.deg2rad(310), np.deg2rad(180), width=10,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 8 - Vertical Straight
        net.add_lane("h", "i", StraightLane([-6.8, -13], [-6.8, -50.5],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=10,
                                            speed_limit=speedlimits[3]))
        net.add_lane("h", "i", StraightLane([-16.8, -13], [-16.8, -50.5],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=10,
                                            speed_limit=speedlimits[3]))

        # 7b
        center4_b = [18.1, -13.1]
        net.add_lane("i", "j",
                     CircularLane(center4_b, radii4, np.deg2rad(180), np.deg2rad(56), width=10,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("i", "j",
                     CircularLane(center4_b, radii4 + 10, np.deg2rad(180), np.deg2rad(58), width=10,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 9 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 33.8]
        radii5 = 18.5
        net.add_lane("j", "a",
                     CircularLane(center5, radii5 + 10, np.deg2rad(240), np.deg2rad(275), width=10,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("j", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(273), width=10,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road


class RacetrackEnvLoop(RacetrackEnv):
    """
    A variant of racetrack with a standard loop shape and additional config options.
    Name racetrack-loop
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h', 'long_off', 'lat_off', 'ang_off'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 180,
                "collision_reward": -3,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 20,
                "screen_width": 1000,
                "screen_height": 1000,
                "centering_position": [0.5, 0.5],
                "terminate_off_road": True, # CL: terminate if car goes off-road
                "hit": False,   # CL: allows 'hits' without crash (i.e., car acts as 'ghost car')
                "restrict_init_collision": 20,  # CL: Change the distance to prevent init collision
                "speed_limits": [None, 20, 18, 20, 18, 20, 18, 20, 18],  # CL: Speed limits for road segments
                "extra_speed": [1, 0.5, 0],  # CL: More speed on three innermost lanes
                "length_v1": 100,  # CL: length of straight segment (in loop)
                "no_lanes": 6,  # CL: Integer number of lanes
                "scenario_1": False,  # CL: Custom obstacle course on bottom straight
                "scenario_2": False,  # CL: Custom obstacle course on bottom straight
                "rand_object": None,  # CL: None = No objects, 0: random no. of objects, int: fixed no. of objects
                "max_objects": 2,  # CL: maximum number of objects per lane
                "reward_speed_range": [20, 30],
                "spawns": False,
                # for highway reward
                "high_speed_reward": 1,
                "right_lane_reward": 1,
                "normalize_reward": True,
                "off_road_penalty": 10,
            }
        )
        return config

    def _reward(self, action: np.ndarray) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        if rewards["on_road_reward"] == 0:
            reward = -self.config["off_road_penalty"]
        print("Reward: ", reward)
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random
        rng_int = np.random.default_rng()

        if self.config["no_lanes"] == 0:
            no_lanes = rng_int.integers(2, high=7)
        else:
            no_lanes = self.config["no_lanes"]

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.integers(no_lanes)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))
            print(lane_index)
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6+rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6+rng.uniform(high=3))
            # Prevent early collisions # CL: is now a parameter in configuration
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < self.config["restrict_init_collision"]:
                    break
            else:
                self.road.vehicles.append(vehicle)

    def _make_road(self) -> None:
        net = RoadNetwork()

        # define rng
        rng = self.np_random
        rng_int = np.random.default_rng()

        # Set Speed Limits for Road Sections - default [None, 20, 18, 20, 18, 20, 18, 20, 18]
        speedlimits = self.config["speed_limits"]
        # set extra speed for innermost lanes - default [1, 0.5, 0]
        extra_speed = self.config["extra_speed"]

        if self.config["length_v1"] == 0:
            length_v1 = rng_int.integers(100, high=200)
        else:
            length_v1 = self.config["length_v1"]

        if self.config["no_lanes"] == 0:
            no_lanes = rng_int.integers(2, high=7)
        else:
            no_lanes = self.config["no_lanes"]

        # Lane 1: Initialise First Inner Lane
        lane = StraightLane(
            [0, 0],
            [length_v1 + 1, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1] + extra_speed[0],
        )
        self.lane = lane

        # successively add lanes
        net.add_lane("a", "b", lane)

        # CL: This for loop must be separate for every segment bcs segment names have to be introduced
        for i in range(1, no_lanes-1):
            # add additional lanes between inner and outer lane
            if i < int(len(extra_speed)):
                extra = extra_speed[i]
            else:
                extra = 0

            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, i * 5],
                    [length_v1 + 1, i * 5],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[1] + extra,
                ),
            )

        # Lane 1: Outer Lane
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, (no_lanes-1) * 5],
                [length_v1 + 1, (no_lanes-1) * 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Turn 1: Inner Lane
        center1 = [length_v1, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2] + extra_speed[0],
            ),
        )

        for i in range(1, no_lanes-1):
            # add additional lanes between inner and outer lane
            if i < int(len(extra_speed)):
                extra = extra_speed[i]
            else:
                extra = 0
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1 + i * 5,
                    np.deg2rad(90),
                    np.deg2rad(0),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.NONE),
                    speed_limit=speedlimits[2] + extra,
                ),
            )

        # Turn 1: Outer Lane
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + (no_lanes-1) * 5,
                np.deg2rad(90),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # Vertical Straight 1: Inner Lane
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [length_v1 + 20, -20],
                [length_v1 + 20, -50],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3] + extra_speed[0],
            ),
        )

        for i in range(1, no_lanes-1):
            # add additional lanes between inner and outer lane
            if i < int(len(extra_speed)):
                extra = extra_speed[i]
            else:
                extra = 0
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [length_v1 + 20 + i * 5, -20],
                    [length_v1 + 20 + i * 5, -50],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[3] + extra,
                ),
            )

        # Vertical Straight 1: Outer Lane
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [length_v1 + 20 + (no_lanes-1) * 5, -20],
                [length_v1 + 20 + (no_lanes-1) * 5, -50],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # Turn 2: Inner Lane
        center2 = [length_v1 + 5, -50]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[4] + extra_speed[0],
            ),
        )

        for i in range(1, no_lanes-1):
            # add additional lanes between inner and outer lane
            if i < int(len(extra_speed)):
                extra = extra_speed[i]
            else:
                extra = 0
            net.add_lane(
                "d",
                "e",
                CircularLane(
                    center2,
                    radii2 + i * 5,
                    np.deg2rad(0),
                    np.deg2rad(-90),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.NONE),
                    speed_limit=speedlimits[4] + extra,
                ),
            )

        # Turn 2: Outer Lane
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2 + (no_lanes-1) * 5,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # Horizontal Straight 2: Inner Lane
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [length_v1  + 5, -65],
                [-5, -65],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[5]  + extra_speed[0],
            ),
        )

        for i in range(1, no_lanes-1):
            # add additional lanes between inner and outer lane
            if i < int(len(extra_speed)):
                extra = extra_speed[i]
            else:
                extra = 0
            net.add_lane(
                "e",
                "f",
                StraightLane(
                    [length_v1 + 5, -(65 + i * 5)],
                    [-5, -(65 + i * 5)],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[5] + extra,
                ),
            )

        # Horizontal Straight 2: Outer Lane
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [length_v1 + 5, -(65 + (no_lanes-1) * 5)],
                [-5, -(65+ (no_lanes-1) * 5)],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[5],
            ),
        )

        # Turn 3: Inner Lane
        center4 = [-5, -50]
        radii4 = 15
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(-90),
                np.deg2rad(-180),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[6] + extra_speed[0],
            ),
        )

        for i in range(1,no_lanes-1):
            # add additional lanes between inner and outer lane
            if i < int(len(extra_speed)):
                extra = extra_speed[i]
            else:
                extra = 0
            net.add_lane(
                "f",
                "g",
                CircularLane(
                    center4,
                    radii4 + i * 5,
                    np.deg2rad(-90),
                    np.deg2rad(-180),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.NONE),
                    speed_limit=speedlimits[6] + extra,
                ),
            )

        # Turn 3: Outer Lane
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center4,
                radii4 + (no_lanes-1) * 5,
                np.deg2rad(-90),
                np.deg2rad(-180),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[6],
            ),
        )

        # Straight 4: Inner Lane
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [-20, -50],
                [-20, -20],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[7] + extra_speed[0],
            ),
        )

        for i in range(1, no_lanes-1):
            # add additional lanes between inner and outer lane
            if i < int(len(extra_speed)):
                extra = extra_speed[i]
            else:
                extra = 0
            net.add_lane(
                "g",
                "h",
                StraightLane(
                    [-20 - i * 5, -50],
                    [-20 - i * 5, -20],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[7] + extra,
                ),
            )

        # Straight 4: Outer Lane
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [-20 - (no_lanes-1) * 5, -50],
                [-20 - (no_lanes-1) * 5, -20],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[7],
            ),
        )

        # Turn 4: Inner Lane
        center6 = [0, -20]
        radii6 = 20
        net.add_lane(
            "h",
            "a",
            CircularLane(
                center6,
                radii6,
                np.deg2rad(180),
                np.deg2rad(90),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[8] + extra_speed[0],
            ),
        )

        for i in range(1, no_lanes-1):
            # add additional lanes between inner and outer lane
            if i < int(len(extra_speed)):
                extra = extra_speed[i]
            else:
                extra = 0
            net.add_lane(
                "h",
                "a",
                CircularLane(
                    center6,
                    radii6 + i * 5,
                    np.deg2rad(180),
                    np.deg2rad(90),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimits[8] + extra,
                ),
            )

        # Turn 4: Outer Lane
        net.add_lane(
            "h",
            "a",
            CircularLane(
                center6,
                radii6 + (no_lanes - 1) * 5,
                np.deg2rad(180),
                np.deg2rad(90),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

        if self.config["scenario_1"]:
            # a custom obstacle course that challenges the agent
            if length_v1 < 90:
                print("Warning: Track too short for scenario, you may want to consider longer track")

            # Still obstacle
            obstacle_0 = Obstacle(self.road, [length_v1-10,0])
            self.road.objects.append(obstacle_0)

            # Still obstacle
            obstacle_1 = Obstacle(self.road, [length_v1-30,5])
            self.road.objects.append(obstacle_1)

            # Still obstacle
            obstacle_2 = Obstacle(self.road, [length_v1-60,10])
            self.road.objects.append(obstacle_2)

            # Still obstacle
            obstacle_3 = Obstacle(self.road, [length_v1-10, 10])
            self.road.objects.append(obstacle_3)

            # Still obstacle
            obstacle_4 = Obstacle(self.road, [length_v1-40, 15])
            self.road.objects.append(obstacle_4)

            # Still obstacle
            obstacle_5 = Obstacle(self.road, [length_v1-20, 20])
            self.road.objects.append(obstacle_5)

        # scenario 2 with two open lanes
        if self.config["scenario_2"]:
            # a custom obstacle course that challenges the agent

            # Still obstacle
            obstacle_1 = Obstacle(self.road, [length_v1,5])
            self.road.objects.append(obstacle_1)

            # Still obstacle
            obstacle_2 = Obstacle(self.road, [length_v1,10])
            self.road.objects.append(obstacle_2)

            # Still obstacle
            obstacle_3 = Obstacle(self.road, [length_v1, 15])
            self.road.objects.append(obstacle_3)

            # Still obstacle
            obstacle_4 = Obstacle(self.road, [length_v1, 20])
            self.road.objects.append(obstacle_4)


        if self.config["rand_object"] is not None:

            if self.config["rand_object"] == 0:
                no_obj = rng_int.integers(1, high=no_lanes * self.config["max_objects"])
            else:
                no_obj = np.min([self.config["rand_object"], no_lanes*self.config["max_objects"]])

            j = 0
            while j < no_obj:
                # choose lane
                lane = rng.choice(range(0, no_lanes, 1))
                # chose segment (one of the four straights)
                segment = rng.choice(range(0, 4, 1))
                if segment == 0:
                    longi = rng.uniform(1, length_v1)
                    obstacle = Obstacle(self.road, [longi, 5*lane])
                    self.road.objects.append(obstacle)
                elif segment == 1:
                    lat = rng.uniform(-49, -21)
                    obstacle = Obstacle(self.road, [length_v1+20+(5*lane), lat])
                    self.road.objects.append(obstacle)
                elif segment == 2:
                    longi = rng.uniform(-4, length_v1-1)
                    obstacle = Obstacle(self.road, [longi, -65-(5*lane)])
                    self.road.objects.append(obstacle)
                elif segment == 3:
                    lat = rng.uniform(-49, -21)
                    obstacle = Obstacle(self.road, [-20-(5*lane), lat])
                    self.road.objects.append(obstacle)
                else:
                    pass
                j+=1

class RacetrackEnvSpawns(RacetrackEnvLoop):
    """
    A variant of racetrack loop with custom spawns for other vehicles.
    Name racetrack-loop-v2
    """

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        if self.config["spawns"]:
            self.controlled_vehicles = []
            lane_index = ("a", "b", 3)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=10)
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            # crashe
            # Front vehicle
            # vehicle_1 = IDMVehicle.make_on_lane(self.road, ("a", "b", 0), longitudinal=20, speed=12)
            # self.road.vehicles.append(vehicle_1)
            # vehicle_2 = IDMVehicle.make_on_lane(self.road, ("a", "b", 1), longitudinal=20, speed=12)
            # self.road.vehicles.append(vehicle_2)
            # vehicle_3 = IDMVehicle.make_on_lane(self.road, ("a", "b", 5), longitudinal=50, speed=11)
            # self.road.vehicles.append(vehicle_3)
            # vehicle_4 = IDMVehicle.make_on_lane(self.road, ("a", "b", 7), longitudinal=40, speed=11)
            # self.road.vehicles.append(vehicle_4)

            # works fine
            # vehicle_1 = IDMVehicle.make_on_lane(self.road, ("a", "b", 1), longitudinal=20, speed=11)
            # self.road.vehicles.append(vehicle_1)
            # vehicle_2 = IDMVehicle.make_on_lane(self.road, ("a", "b", 2), longitudinal=20, speed=11)
            # self.road.vehicles.append(vehicle_2)
            # vehicle_3 = IDMVehicle.make_on_lane(self.road, ("a", "b", 5), longitudinal=20, speed=10)
            # self.road.vehicles.append(vehicle_3)
            # vehicle_4 = IDMVehicle.make_on_lane(self.road, ("a", "b", 7), longitudinal=20, speed=10)
            # self.road.vehicles.append(vehicle_4)

        else:
            # Controlled vehicles
            self.controlled_vehicles = []
            for i in range(self.config["controlled_vehicles"]):
                lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                    self.road.network.random_lane_index(rng)
                controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                                 longitudinal=rng.uniform(20, 50))
                self.controlled_vehicles.append(controlled_vehicle)
                self.road.vehicles.append(controlled_vehicle)
            # Front vehicle
            vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(("b", "c", 0)).length
                                              ),
                                              speed=6+rng.uniform(high=3))
            self.road.vehicles.append(vehicle)

            # Other vehicles
            # CL number of other vehicles chosen randomly anyway lol
            for i in range(rng.integers(self.config["other_vehicles"])):
                random_lane_index = self.road.network.random_lane_index(rng)
                vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                                  longitudinal=rng.uniform(
                                                      low=0,
                                                      high=self.road.network.get_lane(random_lane_index).length
                                                  ),
                                                  speed=6+rng.uniform(high=3))
                # Prevent early collisions # CL: is now a parameter in configuration
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < self.config["restrict_init_collision"]:
                        break
                else:
                    self.road.vehicles.append(vehicle)


class RacetrackEnvEight(RacetrackEnvLoop):
    """
    A variant of racetrack-loop with an upper and a lower loop
    - lower loop allows for higher speed but can be blocked by obstacles
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h', 'long_off', 'lat_off', 'ang_off'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -3,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 1000,
                "screen_height": 1000,
                "centering_position": [0.5, 0.5],
                "new_reward": False, # CL: bool to toggle new reward fct (line 101)
                "terminate_off_road": True, # CL: terminate if car goes off-road
                "hit": False,   # CL: allows 'hits' without crash (i.e., car acts as 'ghost car')
                "restrict_init_collision": 20,  # CL: Change the distance to prevent init collision
                "speed_limits": [None, 20, 18, 15, 15, 20, 18, 20, 18],  # CL: Speed limits for road segments
                "extra_speed": [1, 0.5, 0],  # CL: More speed on three innermost lanes
                "rand_object": None,  # CL: None = No objects, 0: random no. of objects, int: fixed no. of objects
                "max_objects": 4,  # CL: maximum number of objects per lane
                "reward_speed_range": [18, 30],
                "prob": 0.5,    # probability of a road block (two in total)
                "pos": 1,  # position of the road block
                "rand_indicator": False,  # CL: indicator blocks per default same status as block, True: rand
                "off_road_penalty": 1,
            }
        )
        return config

    def _make_road(self) -> None:
        net = RoadNetwork()

        # define rng
        rng = self.np_random
        rng_int = np.random.default_rng()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = self.config["speed_limits"]
        extra_speed = self.config["extra_speed"]

        ########### pt. a ############

        # Lane A: Initialise First outer Lane
        lane = StraightLane(
            [-0.2, 0],
            [11.9, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # Lane A: inner Lane
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [-0.7, 5],
                [12.1, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]  + extra_speed[2],
            ),
        )

        ########### pt. b ############

        # Lane B: outer Lane
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [11.6, 0],
                [30.2, 0],
                line_types=(LineType.NONE, LineType.NONE),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Lane B: inner Lane
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [12.1, 5],
                [30.2, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]  + extra_speed[2],
            ),
        )

        ########### pt. c ############

        # Lane C: outer Lane
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [30.2, 0],
                [92.6, 0],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Lane C: inner Lane
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [30.2, 5],
                [93.1, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]  + extra_speed[2],
            ),
        )

        ########### pt. d ############

        # Lane D: outer Lane
        net.add_lane(
            "d",
            "e",
            StraightLane(
                [92.6, 0],
                [114, 0],
                line_types=(LineType.NONE, LineType.NONE),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Lane D: innter Lane
        net.add_lane(
            "d",
            "e",
            StraightLane(
                [93.1, 5],
                [115, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]  + extra_speed[2],
            ),
        )

        ########### pt. e ############

        # Lane E: outer Lane
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [114, 0],
                [180, 0],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Lane E: inner Lane
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [115, 5],
                [180, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]  + extra_speed[2],
            ),
        )

        # Turn F: inner Lane
        center1 = [180, 25]
        radii1 = 20
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(-90),
                np.deg2rad(0),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[2] + extra_speed[2],
            ),
        )

        # Turn F: Outer Lane
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center1,
                radii1 + 5,
                np.deg2rad(-90),
                np.deg2rad(0),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[2],
            ),
        )

        # Lane G - Inner Lane
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [200, 24],
                [200, 36],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]+ extra_speed[2],
            ),
        )

        # Lane G - Outer Lane
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [205, 25],
                [205, 36],
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Turn H: Inner Lane
        center2 = [180, 35]
        radii2 = 20
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(90),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[2] + extra_speed[2],
            ),
        )

        # Turn H: Outer Lane
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center2,
                radii2 + 5,
                np.deg2rad(0),
                np.deg2rad(90),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[2],
            ),
        )

        # Lane I: Inner Lane
        net.add_lane(
            "i",
            "j",
            StraightLane(
                [180.5, 55],
                [0, 55],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1] + extra_speed[2],
            ),
        )

        # Lane I: Outer Lane
        net.add_lane(
            "i",
            "j",
            StraightLane(
                [180, 60],
                [0, 60],
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )


        # Turn J: Inner Lane
        center3 = [0, 35]
        radii3 = 20
        net.add_lane(
            "j",
            "k",
            CircularLane(
                center3,
                radii3,
                np.deg2rad(90),
                np.deg2rad(180),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[2] + extra_speed[2],
            ),
        )

        # Turn J: Outer Lane
        net.add_lane(
            "j",
            "k",
            CircularLane(
                center3,
                radii3 + 5,
                np.deg2rad(90),
                np.deg2rad(180),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[2],
            ),
        )

        # Lane K - Inner Lane
        net.add_lane(
            "k",
            "l",
            StraightLane(
                [-20, 36],
                [-20, 25],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1] + extra_speed[2],
            ),
        )

        # Lane K - Outer Lane
        net.add_lane(
            "k",
            "l",
            StraightLane(
                [-25, 36],
                [-25, 25],
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Turn L: Inner Lane
        center4 = [0, 25]
        radii4 = 20
        net.add_lane(
            "l",
            "a",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(180),
                np.deg2rad(270),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[2] + extra_speed[2],
            ),
        )

        # Turn L: Outer Lane
        net.add_lane(
            "l",
            "a",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(180),
                np.deg2rad(270),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[2],
            ),
        )

        #### NOW THE FIRST DEVIATION (shorter upper loop)

        # Deviation turn M
        center5 = [92.6, -20]
        radii5 = 20
        net.add_lane(
            "e",
            "m",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "e",
            "m",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.NONE),
                speed_limit=speedlimits[3],
            ),
        )

        net.add_lane(
            "e",
            "m",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(39),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "e",
            "m",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(39),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[3],
            ),
        )

        # Vertical Straight N
        net.add_lane(
            "m",
            "n",
            StraightLane(
                [112.6, -20],
                [112.6, -32],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "m",
            "n",
            StraightLane(
                [117.6, -20],
                [117.6, -32],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # Turn O
        center6 = [97.6, -30]
        radii6 = 15
        net.add_lane(
            "n",
            "o",
            CircularLane(
                center6,
                radii6,
                np.deg2rad(0),
                np.deg2rad(-93),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "n",
            "o",
            CircularLane(
                center6,
                radii6 + 5,
                np.deg2rad(0),
                np.deg2rad(-93),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[3],
            ),
        )

        # Lane P: Inner Lane
        net.add_lane(
            "o",
            "p",
            StraightLane(
                [97.6, -45],
                [27.5, -45],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # Lane P: Outer Lane
        net.add_lane(
            "o",
            "p",
            StraightLane(
                [97.6, -50],
                [27.5, -50],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # Turn Q
        center7 = [27.5, -30]
        radii7 = 15
        net.add_lane(
            "p",
            "q",
            CircularLane(
                center7,
                radii7,
                np.deg2rad(-90),
                np.deg2rad(-182),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "p",
            "q",
            CircularLane(
                center7,
                radii7 + 5,
                np.deg2rad(-90),
                np.deg2rad(-182),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[3],
            ),
        )

        # Vertical Straight R
        net.add_lane(
            "q",
            "r",
            StraightLane(
                [12.5, -30],
                [12.5, -18],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "q",
            "r",
            StraightLane(
                [7.5, -30],
                [7.5, -18],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # Turn S
        center8 = [32.5, -20]
        radii8 = 20
        net.add_lane(
            "s",
            "b",
            CircularLane(
                center8,
                radii8,
                np.deg2rad(180),
                np.deg2rad(90),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "s",
            "b",
            CircularLane(
                center8,
                radii8 + 5,
                np.deg2rad(180),
                np.deg2rad(140),
                width=5,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[3],
            ),
        )

        net.add_lane(
            "s",
            "b",
            CircularLane(
                center8,
                radii8,
                np.deg2rad(180),
                np.deg2rad(90),
                width=5,
                clockwise=False,
                line_types=(LineType.NONE, LineType.NONE),
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "s",
            "b",
            CircularLane(
                center8,
                radii8 + 5,
                np.deg2rad(180),
                np.deg2rad(90),
                width=5,
                clockwise=False,
                line_types=(LineType.NONE, LineType.NONE),
                speed_limit=speedlimits[2],
            ),
        )


        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

        # Place random objects on racetrack-eight
        if self.config["rand_object"] is not None:

            if self.config["rand_object"] == 0:
                no_obj = rng_int.integers(1, high=2 * self.config["max_objects"])
            else:
                no_obj = np.min([self.config["rand_object"], 2*self.config["max_objects"]])

            j = 0
            while j < no_obj:
                # choose lane
                lane = rng.choice(range(0, 2, 1))
                # chose segment (one of the four straights)
                segment = rng.choice(range(0, 3, 1))
                if segment == 0:
                    longi = rng.uniform(1, 180)
                    obstacle = Obstacle(self.road, [longi, 5*lane])
                    self.road.objects.append(obstacle)
                elif segment == 1:
                    longi = rng.uniform(1, 180)
                    obstacle = Obstacle(self.road, [longi, 55 + 5*lane])
                    self.road.objects.append(obstacle)
                elif segment == 2:
                    longi = rng.uniform(21, 102)
                    obstacle = Obstacle(self.road, [longi, -52.5-(5*lane)])
                    self.road.objects.append(obstacle)
                else:
                    pass
                j+=1

        # Block the high speed lane

        # place the roadblocks in pt. G

        # draw status of roadblocks
        block_one = np.random.binomial(1, self.config["prob"])
        block_two = np.random.binomial(1, self.config["prob"])

        if self.config["pos"] == 0:
            position = rng.choice(range(1, 4, 1))
        else:
            position = self.config["pos"]

        if block_one == 1:
            if position == 1:
                obstacle_0 = Obstacle(self.road, [200,30])
                self.road.objects.append(obstacle_0)
            elif position == 2:
                obstacle_0 = Obstacle(self.road, [110,0])
                self.road.objects.append(obstacle_0)
            elif position == 3:
                obstacle_0 = Obstacle(self.road, [150,0])
                self.road.objects.append(obstacle_0)

            if self.config["rand_indicator"] is True:
                indicator_1 = np.random.binomial(1, 0.9)
                if indicator_1 == 1:
                    obstacle_1 = Obstacle(self.road, [70,-5])
                    self.road.objects.append(obstacle_1)
            else:
                obstacle_1 = Obstacle(self.road, [70,-5])
                self.road.objects.append(obstacle_1)

        if block_one == 0:
            if self.config["rand_indicator"] is True:
                indicator_1 = np.random.binomial(1, 0.1)
                if indicator_1 == 1:
                    obstacle_1 = Obstacle(self.road, [70,-5])
                    self.road.objects.append(obstacle_1)

        if block_two == 1:
            if position == 1:
                obstacle_2 = Obstacle(self.road, [205, 30])
                self.road.objects.append(obstacle_2)
            elif position == 2:
                obstacle_2 = Obstacle(self.road, [110, 5])
                self.road.objects.append(obstacle_2)
            elif position == 3:
                obstacle_2 = Obstacle(self.road, [150, 5])
                self.road.objects.append(obstacle_2)

            if self.config["rand_indicator"] is True:
                indicator_2 = np.random.binomial(1, 0.9)
                if indicator_2 == 1:
                    obstacle_3 = Obstacle(self.road, [75, -5])
                    self.road.objects.append(obstacle_3)
            else:
                obstacle_3 = Obstacle(self.road, [75, -5])
                self.road.objects.append(obstacle_3)

        if block_two == 0:
            if self.config["rand_indicator"] is True:
                indicator_2 = np.random.binomial(1, 0.1)
                if indicator_2 == 1:
                    obstacle_3 = Obstacle(self.road, [75, -5])
                    self.road.objects.append(obstacle_3)
