try:
    import msgpackrpc
except ImportError:
    raise ImportError("pip install msgpack-rpc-python")

from .types import *
from .vehicle import *
from .road import *
from .road_network_generator import *
from .pedestrian import *
from .animal import *
from .route_generator import *
from .road_work import *
from .speed_limit import *
from .multirotor import *

import sys
import os
import time
import math
import logging
import sys
import argparse
import enum
from .error import TransportError, TimeoutError


class bcolors:
    HEADER = "\033[95m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


class Client:
    def __init__(self, host="127.0.0.1", port=8080, timeout=10):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.server_ip = host
        self.server_port = port
        self.reconnect_clients()
        self.collision_client = None
        try:
            server_info = self.client.call("GetServerNetInfo")
            self.local_client = self.client
            if server_info == "":
                self.server_ip = host
                self.server_port = port
            else:
                ip_port = server_info.split(":")
                self.server_ip = ip_port[0]
                self.server_port = int(ip_port[1])
                self.client = msgpackrpc.Client(
                    msgpackrpc.Address(self.server_ip, self.server_port),
                    timeout=timeout,
                    pack_encoding="utf-8",
                    unpack_encoding="utf-8",
                )
        except TransportError:
            raise TransportError(
                "\n Make sure SimStar is running from port: %d at: %s"
                % (int(port), host)
            )
        self.compare_simulator_version_with_api()

    def ping(self):
        pong = self.client.call("ping")
        return pong

    def get_ego_id(self):
        return self.client.call("GetVehicleOfInterestID")

    def create_vehicle_from_id(self, vehicle_id):
        return Vehicle(self.local_client, vehicle_id)

    def create_road_generator(
        self,
        spawn_location=WayPoint(0.0, 0.0, 0.0),
        spawn_rotation=WayPointOrientation(0.0, 0.0, 0.0),
        number_of_lanes=3,
    ):
        road_id = self.client.call(
            "SpawnRoad", spawn_location, spawn_rotation, number_of_lanes
        )
        time.sleep(0.5)
        return RoadGenerator(self.client, road_id)

    def create_road_from_id(self, road_id):
        return RoadGenerator(self.client, road_id)

    def get_roads(self):
        return self.client.call("GetRoads")

    def get_traffic_lights(self):
        return self.client.call("GetTrafficLights")

    def reconnect_clients(self):
        self.collision_client = None
        start_time = time.time()
        time.sleep(1)
        while time.time() - start_time < self.timeout:
            try:
                self.local_client = msgpackrpc.Client(
                    msgpackrpc.Address(self.host, self.port),
                    timeout=self.timeout,
                    pack_encoding="utf-8",
                    unpack_encoding="utf-8",
                )
                self.client = msgpackrpc.Client(
                    msgpackrpc.Address(self.server_ip, self.server_port),
                    timeout=self.timeout,
                    pack_encoding="utf-8",
                    unpack_encoding="utf-8",
                )
                # a call is needed to trigger TransportError. otherwise passes
                self.ping()
                break
            except TransportError:
                print("[simstar] reconnecting after reset level")
                time.sleep(1)

    def reset_level(self):
        self.client.call("ResetLevel")
        self.reconnect_clients()

    def open_env(self, env):
        self.client.call("OpenEnvironment", Environments(env).name)
        self.reconnect_clients()

    def adjust_view(self, x_in_m, y_in_m, z_in_m, roll=0, pitch=0, yaw=0, fov=90.0):
        self.client.call(
            "AdjustActiveView", x_in_m, y_in_m, z_in_m, roll, pitch, yaw, fov
        )

    def spawn_vehicle(
        self,
        actor=0,
        distance=20,
        lateral_offset=0,
        lane_id=1,
        initial_speed=30,
        set_speed=50,
        trajectory_data=[],
        vehicle_type=EVehicleType.Sedan1,
        vehicle_model=EVehicleModel.PhysX,
        vehicle_color=None,
        vehicle_tag="",
    ):
        if actor == 0:
            if lateral_offset != 0:
                vehicle_id = self.client.call(
                    "SpawnVehicleTo",
                    distance,
                    lateral_offset,
                    lane_id,
                    initial_speed,
                    set_speed,
                    trajectory_data,
                    vehicle_type,
                    vehicle_model,
                )
            else:
                vehicle_id = self.client.call(
                    "SpawnVehicle",
                    distance,
                    lane_id,
                    initial_speed,
                    set_speed,
                    trajectory_data,
                    vehicle_type,
                    vehicle_model,
                )
        else:
            if lateral_offset != 0:
                vehicle_id = self.client.call(
                    "SpawnVehicleRelativeToActor",
                    actor.get_ID(),
                    distance,
                    lateral_offset,
                    lane_id,
                    initial_speed,
                    set_speed,
                    trajectory_data,
                    vehicle_type,
                    vehicle_model,
                )
            else:
                vehicle_id = self.client.call(
                    "SpawnVehicleRelativeTo",
                    actor.get_ID(),
                    distance,
                    lane_id,
                    initial_speed,
                    set_speed,
                    trajectory_data,
                    vehicle_type,
                    vehicle_model,
                )
        vehicle = Vehicle(self.local_client, vehicle_id)
        if vehicle_color != None:
            vehicle.set_color(vehicle_color)
        if vehicle_tag != "":
            vehicle.set_tag(vehicle_tag)
        return vehicle

    def spawn_merging_vehicle(
        self,
        actor=0,
        distance=0,
        initial_speed=30,
        set_speed=50,
        trajectory_data=[],
        vehicle_type=EVehicleType.Sedan1,
        vehicle_model=EVehicleModel.PhysX,
        vehicle_color=None,
        vehicle_tag="",
    ):
        if actor == 0:
            actor_id = 0
        else:
            actor_id = actor.get_ID()
        vehicle_id = self.client.call(
            "SpawnMergingVehicle",
            actor_id,
            distance,
            initial_speed,
            set_speed,
            trajectory_data,
            vehicle_type,
            vehicle_model,
        )
        vehicle = Vehicle(self.local_client, vehicle_id)
        if vehicle_color != None:
            vehicle.set_color(vehicle_color)
        if vehicle_tag != "":
            vehicle.set_tag(vehicle_tag)
        return vehicle

    def spawn_vehicle_to(
        self,
        vehicle_pose,
        initial_speed=50,
        set_speed=50,
        trajectory_data=[],
        vehicle_type=EVehicleType.Sedan1,
        vehicle_model=EVehicleModel.PhysX,
        vehicle_color=None,
        vehicle_tag="",
    ):
        vehicle_id = self.client.call(
            "SpawnVehicleByLocation",
            vehicle_pose,
            initial_speed,
            set_speed,
            trajectory_data,
            vehicle_type,
            vehicle_model,
        )
        vehicle = Vehicle(self.local_client, vehicle_id)
        if vehicle_color != None:
            vehicle.set_color(vehicle_color)
        if vehicle_tag != "":
            vehicle.set_tag(vehicle_tag)
        return vehicle

    def find_vehicle_with_tag(self, tag):
        vehicle = None
        vehicle_id = self.client.call("GetVehicleByTag", tag)
        if vehicle_id != -1:
            vehicle = Vehicle(self.local_client, vehicle_id)
        return vehicle

    def add_spawn_point(self, pose_data):
        self.client.call("AddSpawnPoint", pose_data)

    def get_spawn_points(self):
        return self.client.call("GetSpawnPoints")

    def check_collisions(self):
        if not self.collision_client:
            host, port, timeout = self.server_ip, self.server_port, 0
            self.collision_client = msgpackrpc.Client(
                msgpackrpc.Address(host, port),
                timeout=timeout,
                pack_encoding="utf-8",
                unpack_encoding="utf-8",
            )

        return self.collision_client.call("CheckCollisions")

    def stop_collision_check(self):
        return self.client.call_async("StopLookingForCollisions")

    def spawn_pedestrian(self, actor=0, distance=20, from_right_to_left=True):
        pedestrian_id = -1
        if actor != 0:
            pedestrian_id = self.client.call(
                "SpawnPedestrianRelativeTo",
                actor.get_ID(),
                distance,
                from_right_to_left,
            )
        return Pedestrian(self.client, pedestrian_id)

    def spawn_random_walker(
        self, walking_speed=5, pedestrian_type=EPedestrianType.Female1
    ):
        pedestrian_id = self.client.call(
            "SpawnRandomWalker", walking_speed, pedestrian_type
        )
        return Pedestrian(self.client, pedestrian_id)

    def spawn_random_walkers(self, density=ETrafficDensity.Low):
        return self.client.call("SpawnRandomWalkers", density)

    def spawn_pedestrian_to_location(
        self,
        location,
        target_location,
        walking_speed=5,
        pedestrian_type=EPedestrianType.Female1,
        pedestrian_tag="",
    ):
        pedestrian_id = self.client.call(
            "SpawnPedestrianToLocation",
            location,
            walking_speed,
            target_location,
            pedestrian_type,
        )
        pedestrian = Pedestrian(self.client, pedestrian_id)
        if pedestrian_tag != "":
            pedestrian.set_tag(pedestrian_tag)
        return pedestrian

    def spawn_pedestrian(
        self,
        location,
        walking_speed=5,
        pedestrian_type=EPedestrianType.Female1,
        pedestrian_tag="",
    ):
        pedestrian_id = self.client.call(
            "SpawnPedestrian", location, walking_speed, pedestrian_type
        )
        pedestrian = Pedestrian(self.client, pedestrian_id)
        if pedestrian_tag != "":
            pedestrian.set_tag(pedestrian_tag)
        return pedestrian

    def find_pedestrian_with_tag(self, tag):
        pedestrian = None
        pedestrian_id = self.client.call("GetPedestrianByTag", tag)
        if pedestrian_id != -1:
            pedestrian = Pedestrian(self.client, pedestrian_id)
        return pedestrian

    def spawn_pedestrian_relative_to_location(
        self,
        origin,
        relative_spawn_location,
        target_location,
        walking_speed=5,
        pedestrian_type=EPedestrianType.Female1,
    ):
        pedestrian_id = self.client.call(
            "SpawnPedestrianRelativeToLocation",
            origin,
            relative_spawn_location,
            walking_speed,
            target_location,
            pedestrian_type,
        )
        return Pedestrian(self.client, pedestrian_id)

    def start_moving_all_pedestrians(self):
        self.client.call("StartMovingAllPedestrians")

    def spawn_animal(self, location, animal_type):
        animal_id = self.client.call("SpawnAnimal", location, animal_type)
        return Animal(self.client, animal_id)

    def set_complex_model_parameters(self, parameters):
        self.client.call("SetComplexModelParameters", parameters)

    def display_vehicle(self, vehicle):
        vehicle_id = vehicle.get_ID()
        self.client.call("DisplayVehicle", vehicle_id)

    # 0,0,0 of the simulation world location corresponds to this reference in geo location
    def set_world_gnss_referece(self, latitude, longitude, altitude=0.0):
        geo_location = GeoLocation(latitude, longitude, altitude)
        self.client.call("SetWorldGeoLocation", geo_location)

    def create_world_from_xodr(self, xodr_content, add_mesh=True, add_debug=False):
        # TODO: should we change to Highway Map before this?
        self.client.call("CreateFromOpenDrive", xodr_content, add_mesh, add_debug)

    def spawn_procedural_building(self, corner_points, height=0, levels=0):
        return self.client.call(
            "SpawnProceduralBuilding", corner_points, height, levels
        )

    def create_procedural_foliage(self, corner_points, foliage_type):
        return self.client.call("CreateProceduralFoliage", corner_points, foliage_type)

    def set_draw_distance(self, distance=300, set_for_road=True):
        return self.client.call("SetDrawDistance", distance, set_for_road)

    def generate_navigation_area(self):
        return self.client.call("GenerateNavigationArea")

    def set_lane_change_disabled(self, enable=True):
        self.client.call("SetLaneChangeDisabled", enable)

    def get_vehicle_ground_truths(self, in_ego_frame=False):
        vehicle_positions = []
        vehicle_positions = self.client.call("GetVehicleGroundTruths")
        return vehicle_positions

    def get_simulator_version(self):
        sim_version = self.client.call("GetVersion")
        return sim_version

    def compare_simulator_version_with_api(self):
        try:
            sim_version = self.get_simulator_version()
            import simstar

            api_version = simstar.__version__
            if sim_version != api_version:
                print(
                    bcolors.WARNING
                    + "[simstar] [Warning] Version Mismatch"
                    + bcolors.ENDC
                )
                print("[simstar] API Version: ", api_version)
                print("[simstar] Simulator Version: ", sim_version)
        except ImportError:
            print("simstar import error")

    def change_control_helper_option(self, option=0):
        self.client.call("ChangeOptionACCorLCC", option)

    def auto_pilot_agents(self, agents):
        agent_ids = []
        for agent in agents:
            agent_ids.append(agent.get_ID())

        self.client.call("AutoPilotAgents", agent_ids)

    def remove_actors(self, actors):
        actor_ids = []
        for actor in actors:
            actor_ids.append(actor.get_ID())
        self.client.call("DeleteActors", actor_ids)

    def set_custom_model_flags(self, agent_flag, ego_flag=False):
        self.client.call("SetCustomModelUseFlags", ego_flag, agent_flag)

    def set_complex_model_frequency(self, frequency=1000.0):
        self.client.call("SetComplexModelFrequency", frequency)

    def create_infinite_highway(
        self,
        is_curved,
        min_radius,
        is_traffic_enabled,
        time_interval=5,
        vehicle_model=0,
    ):
        is_created = self.client.call(
            "InfiniteHighway",
            is_curved,
            min_radius,
            is_traffic_enabled,
            time_interval,
            vehicle_model,
        )
        return is_created

    def check_for_all_vehicle_accidents(self):
        return self.client.call("GetCollidedVehicles")

    def generate_race_track(self, track_name=TrackName.IstanbulPark):
        road_net_gen = RoadNetworkGenerator()
        way_points = road_net_gen.get_way_points(track_name)
        return way_points

    def set_road_network(self, way_points, width_scale=1.0):
        self.client.call("SetRoadNetwork", way_points, True, False, False, width_scale)

    def set_sync_mode(self, is_active, time_dilation=1.0):
        self.local_client.call(
            "SetSynchronousMode", bool(is_active), float(time_dilation)
        )

    def get_tick_number(self):
        return self.local_client.call("GetPassedTickNumber")

    def tick(self):
        self.local_client.call("Tick")

    def blocking_tick(self):
        return self.local_client.call("TickWithWait")

    def tick_given_times(self, tick_num):
        return self.local_client.call("TickGivenTimes", tick_num)

    def set_sync_timeout(self, timeout=30):
        self.local_client.call("SetSyncTimeout", timeout)

    def get_sim_time(self):
        return self.client.call("GetSimTime")

    def get_waypoints(
        self, vehicle, lane_id=1, starting_offset=0, offset=0, count=20, interval=10
    ):
        return self.client.call(
            "GetWayPoints",
            vehicle.get_ID(),
            lane_id,
            starting_offset,
            offset,
            count,
            interval,
        )

    def get_road_generators(self):
        return self.client.call("GetRoadGenerators")

    def get_lanepoints(self, road=0, interval=10):
        if road == 0:
            return self.client.call("GetLanePoints", -1, interval)
        else:
            return self.client.call("GetLanePoints", road.get_ID(), interval)

    def create_route_generator(self, coordinate):
        return RouteGenerator(self.client, coordinate)

    def set_vehicle_transparency(self, transparency):
        self.client.call("SetVehicleTransparency", transparency)

    def set_vehicle_lookahead_distances(self, distances):
        return self.client.call("SetVehicleLookaheadDistances", distances)

    def set_max_speed_for_curvatures(self, curvature_speeds):
        return self.client.call("SetMaxSpeedForCurvatures", curvature_speeds)

    def spawn_road_work(self, actor=0, distance=0, lane=1, length=10):
        road_work_id = self.client.call(
            "SpawnRoadWorkRelativeTo", actor.get_ID(), distance, lane, length
        )
        return RoadWork(self.client, road_work_id)

    def spawn_speed_limit(
        self, actor=0, distance=0, speed_limit=ESpeedLimitState().SpeedLimit_30
    ):
        speed_limit_id = self.client.call(
            "SpawnSpeedLimitRelativeTo", actor.get_ID(), distance, speed_limit
        )
        return SpeedLimit(self.client, speed_limit_id)

    def spawn_traffic_sign_relatively(self, actor=0, distance=0, sign="AcilanKopru"):
        sign_id = self.client.call(
            "SpawnTrafficSignRelativeTo", actor.get_ID(), distance, sign
        )
        return TrafficSign(self.client, sign_id)

    def get_traffic_signs(self):
        return self.client.call("GetTrafficSigns")

    def get_coordinate_frame(self):
        return self.client.call("GetSimCoordinateFrame")

    def set_coordinate_frame(self, cf):
        return self.client.call("SetSimCoordinateFrame", cf)

    def set_fixed_frame_rate(self, frame_rate=60):
        return self.client.call("SetFixedFrameRate", frame_rate)

    def set_weather(self, rain=0, snow=0, fog=0, dust=0, delay=0.0):
        if delay == 0.0:
            return self.client.call("UpdateWeather", rain, snow, fog, dust)
        else:
            return self.client.call(
                "UpdateWeatherAfterTime", rain, snow, fog, dust, delay
            )

    def set_weather_after_distance(
        self, actor, rain=0, snow=0, fog=0, dust=0, distance=10.0
    ):
        actor_id = actor.get_ID()
        return self.client.call(
            "UpdateWeatherAfterDistance", actor_id, rain, snow, fog, dust, distance
        )

    def set_time_of_day(self, hour=12, time_passing_speed=0):
        return self.client.call("SetTimeOfDay", hour, time_passing_speed)

    def set_lane_change_disabled(self, is_disabled):
        self.client.call("SetLaneChangeDisabled", is_disabled)

    def get_scenario_list(self):
        return self.client.call("GetScenarioList")

    def get_scenario_description(self, scenario_name):
        return self.client.call("GetScenarioDescription", scenario_name)

    def load_scenario(self, scenario_name, should_start=False):
        if should_start:
            return self.client.call("LoadAndStartScenario", scenario_name)
        else:
            return self.client.call("LoadScenario", scenario_name)

    def save_scenario(
        self, scenario_name, scenario_description="", extended_description=None
    ):
        if extended_description != None:
            self.client.call(
                "SaveScenarioExtendedDescription", scenario_name, extended_description
            )
        else:
            self.client.call("SaveScenario", scenario_name, scenario_description)

    def preview_scenario(self, scenario_name, preview_time=10.0):
        self.client.call("PreviewScenario", scenario_name, preview_time)

    def clear_preview_trajectory(self):
        self.client.call("ClearPreviewTrajectory")

    def create_open_crg_road(
        self,
        road_length=100.0,
        road_width=10.0,
        bump_profile=[],
        bump_drawing=EDrawType.NoDrawing,
    ):
        self.client.call(
            "GenerateOpenCrgRoad", road_length, road_width, bump_profile, bump_drawing
        )
        self.reconnect_clients()

    def load_open_crg(self, path_to_file):
        self.client.call("LoadOpenCrg", path_to_file)
        self.reconnect_clients()

    def get_opencrg_road_bounds(self):
        return self.client.call("GetOpenCrgRoadBounds")

    def get_opencrg_road_geometry(self, points_2d):
        return self.client.call("GetOpenCrgRoadGeometry", points_2d)

    def add_overpass(self, height=10, add_exit=True, add_merge=True):
        return self.client.call("AddOverPass", height, add_exit, add_merge)

    def get_pose_of_point_along_road_spline(self, road_generator, lane_id, distance):
        road_generator_id = road_generator.get_ID()
        return self.client.call(
            "GetPoseOfPointAlongRoadSpline", road_generator_id, lane_id, distance
        )

    def get_mouse_location(self):
        return self.client.call("GetMouseLocation")

    def get_traffic_sign_list(self):
        return self.client.call("GetTrafficSignList")

    def spawn_traffic_sign(self, sign_name, pose_data):
        sign_id = self.client.call("SpawnTrafficSign", sign_name, pose_data)
        return TrafficSign(self.client, sign_id)

    def set_traffic_lights(self, traffic_lights):
        return self.client.call("SetTrafficLightData", traffic_lights)

    def start_city_traffic(self, density):
        return self.client.call("CityTraffic", density)

    def generate_city_traffic(self, density, is_heavy_vehicles_enabled=False):
        return self.client.call(
            "GenerateCityTraffic", density, is_heavy_vehicles_enabled
        )

    def measure_fps(self):
        """
        measure FPS from passed Tick number
        timeout in case sim is in sync mode
        """
        timeout_sec = 2
        num_ticks_to_wait = 20
        tick_num = self.get_tick_number()
        target_tick_num = tick_num + num_ticks_to_wait
        start_time = time.time()

        while (time.time() - start_time) < timeout_sec:
            tick_num = self.get_tick_number()
            if tick_num == target_tick_num:
                break
        time_diff = time.time() - start_time
        fps = num_ticks_to_wait / time_diff
        return fps

    def get_random_spawn_location(self):
        return self.client.call("GetRandomWalkableLocation")

    def spawn_multirotor(self, name, position):
        id = self.client.call("SpawnMultirotor", name, position)
        if id != -1:
            return Multirotor(self.local_client, name, id)
        else:
            return None

    def get_all_spots_info(self):
        return self.client.call("GetAllSpotsInfo")

    def get_all_spots_info_ego_frame(self, position=EPosition.Center):
        return self.client.call("GetAllSpotsInfoEgoFrame", position)

    def get_spot_info(self, spot_name):
        return self.client.call("GetSpotInfo", spot_name)

    def get_spot_info_ego_frame(self, spot_name, position=EPosition.Center):
        return self.client.call("GetSpotInfoEgoFrame", spot_name, position)

    def update_spot_info(
        self, spot_name, grid_width=3, grid_height=6, lane_thickness=0.3
    ):
        return self.client.call(
            "UpdateSpotInfo", spot_name, grid_width, grid_height, lane_thickness
        )

    def make_spot_available(self, spot_name):
        return self.client.call("MakeSpotAvailable", spot_name)

    def make_spot_unavailable(self, spot_name, offset_x=0, offset_y=0, yaw_in_degree=0):
        return self.client.call(
            "MakeSpotUnavailable", spot_name, offset_x, offset_y, yaw_in_degree
        )

    def spawn_traffic_cone(self, location, scale=1.0):
        return self.client.call("SpawnTrafficCone", location, scale)

    def limit_physics_simulation(self, is_enabled, radius_around_ego=0):
        return self.client.call("LimitPhysicsSimulation", is_enabled, radius_around_ego)
