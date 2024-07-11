# CL: Racetrack env large with all extra features so far (reward, termination, ghost car ..)
from itertools import repeat, product
from typing import Tuple, Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle


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
                "collision_reward": 1000,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 1000,
                "screen_height": 1000,
                "centering_position": [0.5, 0.5],
                "new_reward": True,                 # CL: Created new reward function; used if int
                "restrict_init_collision": 20,      # CL: Change the distance to prevent init collision
                "hit": False,                       # CL: Allows 'hits' but continuing, i.e., 'ghost car'
                "terminate_off_road": False,        # CL: terminate if car goes off-road
                "speed_limits": [None, 25, 15, 25, 15, 25, 15, 25, 15],     # CL: Speed limits for road segments
                "extra_speed": [10, 5, 2],          # CL: More speed on innermost lanes
                "no_lanes": 6,                      # CL: Integer number of lanes
                "rand_object": 0,                   # CL: No. of random object on road
                "scenario_1": False,                # CL: Custom scenario 1
                "length_v1": 200,                   # CL: length of track for v1
                "max_objects": 4,                   # CL: maximum number of objects per lane
                "rand_indicator": False,            # CL: indicator blocks per default same status as block, True: rand
                "prob": 0.5,
                "reward_speed_range": [20, 30],
                "spawns": False,
            }
        )
        return config

    # CL: Define a method that either is hit or crash
    def crash_or_hit(self):
        if self.config["hit"]:
            # hit other car but pass through it (ego vehicle is 'ghost car')
            return self.vehicle.hit
        else:
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
            reward = ((1-self.crash_or_hit()) * reward + self.crash_or_hit() * self.config["collision_reward"] \
                     - self._is_terminated())
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            "action_reward": np.linalg.norm(action),
            # CL: Allow for "ghost collisions"
            "collision_reward": self.crash_or_hit(),
            "on_road_reward": self.vehicle.on_road,
        }

    # CL: Terminate if crashed (except "ghost collisions") and optionally if off-road
    def _is_terminated(self) -> bool:
        if self.config["terminate_off_road"]:
            while self.vehicle.on_road:
                if self.config["hit"]:
                    return False
                else:
                    return self.vehicle.crashed
            return True
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

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

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


'''
    def _make_ghost_object(self) -> None:
        """
        Populate a road with several several random objects that alter the observed state but are irrelevant for
        the optimal action.
        """

        # e.g., to draw a random lane index
        rng = self.np_random

        # boxes (made frm objects, same as vehicles, but set velocity to zero and alter colour)
        self.ghost_objects = []
        for i in range(self.config["ghost_objects"]):
            lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            ghost_object = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=0,
                                                                             longitudinal=rng.uniform(20, 50))

            self.ghost_objects.append(ghost_object)
            self.road.objects.append(ghost_object)

        # First object
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
            # Prevent early collisions # CL is now a parameter in configuration
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < self.config["restrict_init_collision"]:
                    break
            else:
                self.road.vehicles.append(vehicle)
'''

# TODO Make large class and one with more lanes
class RacetrackEnvLarge(RacetrackEnv):
    """
    A variant of racetrack-v0 with larger circuit:
    """
    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        # CL: Config speed limits manually
        speedlimits = self.config["speed_limits"]

        # Initialise First Lane # TODO (CL): PASST
        lane = StraightLane([44, 5.1], [131, 5.1], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=10, speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b", StraightLane([44, 15.1], [131, 15.1], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=10, speed_limit=speedlimits[1]))
        # 10 instead of five moved the lane down

        # 2 - Circular Arc #1 TODO (CL):PASST
        center1 = [130.1, -14.8]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(91), np.deg2rad(4), width=10,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+10, np.deg2rad(91), np.deg2rad(4), width=10,
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
                     CircularLane(center2, radii2+10, np.deg2rad(0), np.deg2rad(-180), width=10,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [95.5, -52]
        radii3 = 15
        net.add_lane("e", "f",
                     CircularLane(center3, radii3+10, np.deg2rad(0), np.deg2rad(136), width=10,
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
                     CircularLane(center4, radii4+10, np.deg2rad(310), np.deg2rad(180), width=10,
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
                     CircularLane(center4_b, radii4+10, np.deg2rad(180), np.deg2rad(58), width=10,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 9 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 33.8]
        radii5 = 18.5
        net.add_lane("j", "a",
                     CircularLane(center5, radii5+10, np.deg2rad(240), np.deg2rad(275), width=10,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("j", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(273), width=10,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road



class RacetrackEnvV1(RacetrackEnv):
    """
    A variant of racetrack-v0 with more lanes and a loop:
    Actual name racetrack-loop
    """
    def _make_road(self) -> None:
        net = RoadNetwork()

        # define rng
        rng = self.np_random
        rng_int = np.random.default_rng()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = self.config["speed_limits"]
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

        # CL: This for loop must be separate for every segment bcs segment names have to be introduced
        for i in range(2,no_lanes):
            """Add additional lanes between """
            if i < int(len(extra_speed)):
                extra = extra_speed[i-1]
            else:
                extra = 0
            # successively add lanes
            net.add_lane("a", "b", lane)
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, (i - 1) * 5],
                    [length_v1 + 1, (i - 1) * 5],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[1] + extra,
                ),
            )

        # Lane 1: Outer Lane
        net.add_lane("a", "b", lane)
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

        for i in range(2,no_lanes):
            """Add additional lanes between """
            if i < int(len(extra_speed)):
                extra = extra_speed[i-1]
            else:
                extra = 0
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1 + (i - 1) * 5,
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

        for i in range(2,no_lanes):
            """Add additional lanes between """
            if i < int(len(extra_speed)):
                extra = extra_speed[i-1]
            else:
                extra = 0
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [length_v1 + 20 + (i - 1) * 5, -20],
                    [length_v1 + 20 + (i - 1) * 5, -50],
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

        for i in range(2,no_lanes):
            """Add additional lanes between """
            if i < int(len(extra_speed)):
                extra = extra_speed[i-1]
            else:
                extra = 0
            net.add_lane(
                "d",
                "e",
                CircularLane(
                    center2,
                    radii2 + (i - 1) * 5,
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

        for i in range(2,no_lanes):
            """Add additional lanes between """
            if i < int(len(extra_speed)):
                extra = extra_speed[i-1]
            else:
                extra = 0
            net.add_lane(
                "e",
                "f",
                StraightLane(
                    [length_v1 + 5, -(65 + (i - 1) * 5)],
                    [-5, -(65 + (i - 1) * 5)],
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
                [length_v1 + 5, -(65+(no_lanes-1) * 5)],
                [-5, -(65+(no_lanes-1) * 5)],
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

        for i in range(2,no_lanes):
            """Add additional lanes between """
            if i < int(len(extra_speed)):
                extra = extra_speed[i-1]
            else:
                extra = 0
            net.add_lane(
                "f",
                "g",
                CircularLane(
                    center4,
                    radii4 + (i - 1) * 5,
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

        for i in range(2,no_lanes):
            """Add additional lanes between """
            if i < int(len(extra_speed)):
                extra = extra_speed[i-1]
            else:
                extra = 0
            net.add_lane(
                "g",
                "h",
                StraightLane(
                    [-20 - (i - 1) * 5, -50],
                    [-20 - (i - 1) * 5, -20],
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

        for i in range(2,no_lanes):
            """Add additional lanes between """
            if i < int(len(extra_speed)):
                extra = extra_speed[i-1]
            else:
                extra = 0
            net.add_lane(
                "h",
                "a",
                CircularLane(
                    center6,
                    radii6 + (i - 1) * 5,
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

class RacetrackEnvV2(RacetrackEnv):
    """
    A variant of racetrack-v0 with more loops:
    """

    def _make_road(self) -> None:
        net = RoadNetwork()

        # define rng
        rng = self.np_random
        rng_int = np.random.default_rng()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = self.config["speed_limits"]
        extra_speed = self.config["extra_speed"]

        ########### pt. a ############

        # Lane A: Initialise First Inner Lane
        lane = StraightLane(
            [-0.2, 0],
            [14.4, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # Lane A: Outer Lane
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [-0.7, 5],
                [14.6, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]  + extra_speed[2],
            ),
        )

        ########### pt. b ############

        # Lane B: Inner Lane
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [14.1, 0],
                [30.2, 0],
                line_types=(LineType.NONE, LineType.NONE),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Lane B: Outer Lane
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [14.6, 5],
                [30.2, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]  + extra_speed[2],
            ),
        )

        ########### pt. c ############

        # Lane C: Inner Lane
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

        # Lane C: Outer Lane
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

        # Lane D: Inner Lane
        net.add_lane(
            "d",
            "e",
            StraightLane(
                [92.6, 0],
                [109.5, 0],
                line_types=(LineType.NONE, LineType.NONE),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Lane D: Outer Lane
        net.add_lane(
            "d",
            "e",
            StraightLane(
                [93.1, 5],
                [110, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]  + extra_speed[2],
            ),
        )

        ########### pt. e ############

        # Lane E: Inner Lane
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [109, 0],
                [180, 0],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Lane E: Outer Lane
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [110, 5],
                [180, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1]  + extra_speed[2],
            ),
        )

        # # Lane F: Inner Lane
        # net.add_lane(
        #     "f",
        #     "g",
        #     StraightLane(
        #         [140, 0],
        #         [150, 0],
        #         line_types=(LineType.NONE, LineType.NONE),
        #         width=5,
        #         speed_limit=speedlimits[5]  + extra_speed[0],
        #     ),
        # )
#
        # # Lane F: Outer Lane
        # net.add_lane(
        #     "f",
        #     "g",
        #     StraightLane(
        #         [140, 5],
        #         [150, 5],
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         width=5,
        #         speed_limit=speedlimits[5],
        #     ),
        # )
#
        # # Lane G: Inner Lane
        # net.add_lane(
        #     "g",
        #     "h",
        #     StraightLane(
        #         [150, 0],
        #         [180, 0],
        #         line_types=(LineType.CONTINUOUS, LineType.NONE),
        #         width=5,
        #         speed_limit=speedlimits[5]  + extra_speed[0],
        #     ),
        # )
#
        # # Lane G: Outer Lane
        # net.add_lane(
        #     "g",
        #     "h",
        #     StraightLane(
        #         [150, 5],
        #         [180, 5],
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         width=5,
        #         speed_limit=speedlimits[5],
        #     ),
        # )

        # Turn F: Inner Lane
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
                speed_limit=speedlimits[3] + extra_speed[2],
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
                speed_limit=speedlimits[3],
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

        # Lane M - Inner Lane
        net.add_lane(
            "e",
            "m",
            StraightLane(
                [93.8, -0.5],
                [116.7, -18.3],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[1] + extra_speed[2],
            ),
        )

        # Lane M - Outer Lane
        net.add_lane(
            "e",
            "m",
            StraightLane(
                [107.5, -4.7],
                [120.6, -14.9],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Turn N: Inner Lane
        center5 = [102.8, -32.5]
        radii5 = 20
        net.add_lane(
            "m",
            "n",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(45),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2] + extra_speed[2],
            ),
        )

        # Turn N: Outer Lane
        net.add_lane(
            "m",
            "n",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(45),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # Turn O - Inner Lane
        center6 = [102.8, -32.5]
        radii6 = 20
        net.add_lane(
            "n",
            "o",
            CircularLane(
                center6,
                radii6,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2] + extra_speed[2],
            ),
        )

        # Turn O - Outer Lane
        net.add_lane(
            "n",
            "o",
            CircularLane(
                center6,
                radii6 + 5,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # Lane P: Inner Lane
        net.add_lane(
            "o",
            "p",
            StraightLane(
                [103.3, -52.5],
                [20, -52.5],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[4],
            ),
        )

        # Lane P: Outer Lane
        net.add_lane(
            "o",
            "p",
            StraightLane(
                [102.8, -57.5],
                [20, -57.5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[4],
            ),
        )

        # Turn Q: Inner Lane
        center7 = [20, -32.5]
        radii7 = 20
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
                speed_limit=speedlimits[2] + extra_speed[2],
            ),
        )

        # Turn Q: Outer Lane
        net.add_lane(
            "p",
            "q",
            CircularLane(
                center7,
                radii7 + 5,
                np.deg2rad(-90),
                np.deg2rad(-180),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # Turn R: Inner Lane
        center8 = [20, -32.5]
        radii8 = 20
        net.add_lane(
            "q",
            "r",
            CircularLane(
                center8,
                radii8,
                np.deg2rad(-178),
                np.deg2rad(-225),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2] + extra_speed[2],
            ),
        )

        # Turn R: Outer Lane
        net.add_lane(
            "q",
            "r",
            CircularLane(
                center8,
                radii8 + 5,
                np.deg2rad(-180),
                np.deg2rad(-225),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # Lane S: Inner Lane
        net.add_lane(
            "r",
            "b",
            StraightLane(
                [6.1, -18.3],                  # [(20+20*cos(pi/4)), -32.5 + 20*sin(pi/4)]
                [29 , -0.5],                                            # [(15+25*cos(pi/4)), -32.5 + 25*sin(pi/4)]
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[1] + extra_speed[2],
            ),
        )

        # Lane S: Outer Lane
        net.add_lane(
            "r",
            "b",
            StraightLane(
                [2.2, -14.9],                    # [15+25*cos(pi/4), -32.5 + 25*sin(pi/4)]
                [15.3, -4.7],                    # [-(10+30*cos(pi/4)), -32.5 + 25*sin(pi/4)]
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

        # Place random objects on racetrack v2
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

        # Block the highspeed lane

        # place the roadblocks in pt. G

        # draw status of roadblocks
        block_one = np.random.binomial(1, self.config["prob"])
        block_two = np.random.binomial(1, self.config["prob"])

        if block_one == 1:
            obstacle_0 = Obstacle(self.road, [200,30])
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
            obstacle_2 = Obstacle(self.road, [205, 30])
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

class RacetrackEnvV3(RacetrackEnv):
    """Different reward functions"""
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
                "new_reward": True,                 # CL: Created new reward function; used if int
                "restrict_init_collision": 20,      # CL: Change the distance to prevent init collision
                "hit": False,                       # CL: Allows 'hits' but continuing, i.e., 'ghost car'
                "terminate_off_road": False,        # CL: terminate if car goes off-road
                "speed_limits": [None, 25, 15, 25, 15, 25, 15, 25, 15],     # CL: Speed limits for road segments
                "extra_speed": [10, 5, 2],          # CL: More speed on innermost lanes
                "no_lanes": 6,                      # CL: Integer number of lanes
                "rand_object": 0,                   # CL: No. of random object on road
                "scenario_1": False,                # CL: Custom scenario 1
                "length_v1": 200,                   # CL: length of track for v1
                "max_objects": 4,                   # CL: maximum number of objects per lane
                "rand_indicator": False,            # CL: indicator blocks per default same status as block, True: rand
                "prob": 0.5,
                "reward_speed_range": [20, 30],
                "spawns": False,
                "reward_fct": "default",             # CL: also: "highway", "copilot", "custom", "eight"
                "high_speed_reward": 1,
                "right_lane_reward": 1,
                "normalize_reward": True,
                "collision_penalty_min": -1.0,
                "collision_penalty_max": 0.0,
                "off_road_penalty_min": -1.0,
                "off_road_penalty_max": 0.0,
                "speed_reward_min": 0.0,
                "speed_reward_max": 12.0,  # Assuming the maximum speed is 30
                "action_penalty_min": -0.1,
                "action_penalty_max": 0.0,
                "speed_factor": 1.0,
                "action_factor": 1.0,
            }
        )
        return config

    if self.config["reward_fct"] == "default":
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
                reward = ((1 - self.crash_or_hit()) * reward + self.crash_or_hit() * self.config["collision_reward"] \
                          - self._is_terminated())
            return reward

        def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
            _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
            return {
                "lane_centering_reward": 1
                                         / (1 + self.config["lane_centering_cost"] * lateral ** 2),
                "action_reward": np.linalg.norm(action),    # CL: Penalize actions
                # CL: Allow for "ghost collisions"
                "collision_reward": self.crash_or_hit(),    # that is 0 or 1
                "on_road_reward": self.vehicle.on_road,     # that is 0 or 1
            }

    elif self.config["reward_fct"] == "highway":
        def _reward(self, action: Action) -> float:
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
            return reward

        def _rewards(self, action: Action) -> Dict[Text, float]:
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

    elif self.config["reward_fct"] == "copilot":
        def _reward(self, action: np.ndarray) -> float:
            rewards = self._rewards(action)
            normalized_rewards = {
                name: utils.lmap(reward, [self.config[name + "_min"], self.config[name + "_max"]], [0, 1])
                for name, reward in rewards.items()
            }
            reward = sum(
                self.config.get(name, 0) * normalized_reward for name, normalized_reward in normalized_rewards.items()
            )
            return reward

        def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
            _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
            return {
                "collision_penalty": -1000.0 if self.vehicle.crashed else 0.0,
                "off_road_penalty": -100.0 if not self.vehicle.on_road else 0.0,
                "speed_reward": self.vehicle.speed,
                "action_penalty": -0.1,
            }

    elif self.config["reward_fct"] == "custom":
        def _reward(self, action: Action) -> float:
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
            return reward

        def _rewards(self, action: Action) -> Dict[Text, float]:
            neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
            lane = (
                self.vehicle.target_lane_index[2]
                if isinstance(self.vehicle, ControlledVehicle)
                else self.vehicle.lane_index[2]
            )
            # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
            forward_speed = self.vehicle.speed

            return {
                "collision_reward": (- float(self.vehicle.crashed) * (-self.config["collision_reward"])),
                "action_reward": np.linalg.norm(action) * self.config["action_factor"],    # CL: Penalize actions
                "high_speed_reward": (forward_speed/(self.config["speed_limits"][1]+self.config["extra_speed"][0])
                                      * self.config["speed_factor"]),
                "off_road": - float(self._is_terminated()),
            }

    elif self.config["reward_fct"] == "eight":
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
                reward = ((1-self.crash_or_hit()) * reward - self.crash_or_hit() * self.config["collision_reward"] \
                         - self._is_terminated())
            return reward

        def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
            _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
            scaled_speed = utils.lmap(
                self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
            )
            return {
                "lane_centering_reward": 1
                / (1 + self.config["lane_centering_cost"] * lateral**2),
                "action_reward": np.linalg.norm(action),
                # CL: Allow for "ghost collisions"
                "collision_reward": self.crash_or_hit(),
                "on_road_reward": self.vehicle.on_road,
                "high_speed_reward": np.clip(scaled_speed, 0, 1),
            }

    else:
        print("No reward function selected.")
