import gym
import collections as col
import numpy as np
import time
import json
from datetime import datetime
from gym import spaces


try:
    import simstar
except ImportError:
    print("go to PythonAPI folder where setup.py is located")
    print("python setup.py install")


class SaveEpisode:
    def __init__(self):
        self.save_data = {}

    def save_step(self, inputs, observation, name="undefined"):
        if name not in self.save_data:
            self.save_data[name] = []
        # clip action values in case simstar returns values out of border
        inputs["throttle"] = np.clip(inputs["throttle"], 0, 1)
        inputs["steer"] = np.clip(inputs["steer"], -1, 1)
        inputs["brake"] = np.clip(inputs["brake"], 0, 1)
        self.save_data[name].append(
            {
                "action": inputs,
                "data": observation,
            }
        )

    def print_to_file(self, filename=None):
        if not filename:
            filename = (
                str(datetime.now().strftime("[SimStar] %d-%m-%Y %H.%M.%S")) + ".json"
            )
        if not self.save_data:
            return
        with open(filename, "w") as outfile:
            json.dump(self.save_data, outfile, ensure_ascii=False, indent=4)

    def reset(self):
        self.save_data = {}

    def __del__(self):
        del self.save_data


class SensoredVehicle(simstar.Vehicle):
    def __init__(self, vehicle: simstar.Vehicle, track_sensor, opponent_sensor, tag=""):
        super().__init__(vehicle.client, vehicle._ID)
        self.track_sensor = track_sensor
        self.opponent_sensor = opponent_sensor
        self.tag = tag


class SimstarEnv(gym.Env):
    def __init__(
        self,
        track=simstar.Environments.Racing,
        add_opponents=False,
        synronized_mode=False,
        num_opponents=3,
        speed_up=1,
        host="127.0.0.1",
        port=8080,
        save=False,
        ego_driving_type=simstar.DriveType.API, # NOTE: Keyboard: (for manual driving) and API: (for driving via API)
        create_track=True,
        opponent_pose_data=[],
    ):

        """

        Args:
            track (simstar.Environments, optional): SimStar Environment. Defaults to simstar.Environments.Racing.
            add_opponents (bool, optional): Add opponents or not. Defaults to False.
            synronized_mode (bool, optional): Use sync mode. Defaults to False.
            num_opponents (int, optional): Number of opponents, `add_opponents` needs to be True. Defaults to 3.
            speed_up (int, optional): Speed up factor. Defaults to 1.
            host (str, optional): SimStar host. Defaults to "127.0.0.1".
            port (int, optional): SimStar port. Defaults to 8080.
            save (bool, optional): Save vehicle inputs and observations at every step to file. Defaults to False.
            ego_driving_type (simstar.DriveType, optional): Choose driving type for ego vehicle . Defaults to simstar.DriveType.API.
            create_track (bool, optional): Create track from scratch(True) or use already existing one(False). Defaults to True.
            opponent_pose_data ([simstar.PoseData], optional): Optional positions of opponents. Defaults to [].

        Raises:
            simstar.TransportError: Raised when connection could not be set up with SimStar
        """

        self.c_w = 0.02  # out of track penalty weight

        self.add_opponents = add_opponents  # True: adds opponent vehicles; False: removes opponent vehicles
        self.number_of_opponents = num_opponents  # agent_locations, agent_speeds, and lane_ids sizes has to be the same
        self.agent_locations = [
            -10,
            -20,
            10,
        ]  # opponents' meters offset relative to ego vehicle
        self.opponent_poses = (
            opponent_pose_data  # overrides agent_locations if not empty
        )
        self.agent_speeds = [45, 80, 55]  # opponent vehicle speeds in km/hr
        self.lane_ids = [
            1,
            1,
            1,
        ]  # make sure that the lane ids are not greater than number of lanes

        self.ego_lane_id = (
            1  # make sure that ego vehicle lane id is not greater than number of lanes
        )
        self.ego_start_offset = (
            25  # ego vehicle's offset from the starting point of the road
        )
        self.default_speed = 160  # km/hr
        self.set_ego_speed = 60  # km/hr
        self.road_width = 14  # meters

        self.track_sensor_size = 19
        self.opponent_sensor_size = 18

        self.time_step_slow = 0
        self.terminal_judge_start = (
            100  # if ego vehicle does not have progress for 100 steps, terminate
        )
        self.termination_limit_progress = (
            6  # if progress of the ego vehicle is less than 6 for 100 steps, terminate
        )

        # the type of race track to generate
        self.track_name = track

        self.synronized_mode = (
            synronized_mode  # simulator waits for update signal from client if enabled
        )
        self.speed_up = speed_up  # how faster should simulation run. up to 6x.
        self.host = host
        self.port = port

        self.hz = 10  # fixed control frequency
        self.fps = 60  # fixed simulation FPS
        self.tick_number_to_sample = self.fps / self.hz
        self.sync_step_num = int(self.tick_number_to_sample / self.speed_up)

        self.driving_type = ego_driving_type

        self.save = save
        if self.save:
            self.saver = SaveEpisode()

        try:
            self.client = simstar.Client(host=self.host, port=self.port, timeout=20)
            self.client.ping()
        except simstar.TimeoutError or simstar.TransportError:
            raise simstar.TransportError(
                "******* Make sure a Simstar instance is open and running at port %d*******"
                % (self.port)
            )

        print("[SimstarEnv] initializing environment, this may take a while...")
        if create_track:
            self.client.open_env(self.track_name)
            time.sleep(20)
        else:
            old_vehicles = []
            all_info = self.client.get_vehicle_ground_truths()
            for info in all_info:
                old_vehicles.append(self.client.create_vehicle_from_id(info["id"]))

            self.client.remove_actors(old_vehicles)

        # get main road
        self.road = None
        all_roads = self.client.get_roads()

        if len(all_roads) > 0:
            road_main = all_roads[8]
            road_id = road_main["road_id"]
            self.road = simstar.RoadGenerator(self.client, road_id)

        # a list contaning all vehicles
        self.actor_list = []

        # disable lane change for automated actors
        self.client.set_lane_change_disabled(is_disabled=True)

        # input space.
        high = np.array([np.inf, np.inf, 1.0, 1.0])
        low = np.array([-np.inf, -np.inf, 0.0, 0.0])
        self.observation_space = spaces.Box(low=low, high=high)

        # action space: [steer, accel-brake]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.default_action = [0.0, 1.0]

        self.prev_angle = 0.0

        self.last_step_time = time.time()
        self.apply_settings()

    def apply_settings(self):
        print("[SimstarEnv] sync: ", self.synronized_mode, " speed up: ", self.speed_up)
        self.client.set_sync_timeout(10)
        self.client.set_sync_mode(self.synronized_mode, self.speed_up)

    def save_episode(self):
        if self.save:
            self.saver.print_to_file()

    def reset(self):
        print("[SimstarEnv] actors are destroyed")
        time.sleep(0.5)

        if self.save:
            self.saver.reset()

        self.time_step_slow = 0

        # delete all the actors
        self.client.remove_actors(self.actor_list)
        self.actor_list.clear()

        # spawn a vehicle
        self.main_vehicle = self.client.spawn_vehicle(
            actor=self.road,
            distance=self.ego_start_offset,
            lane_id=self.ego_lane_id,
            initial_speed=0,
            set_speed=self.set_ego_speed,
            vehicle_type=simstar.EVehicleType.F1Racing,
        )

        self.simstar_step()
        print("[SimstarEnv] main vehicle ID: ", self.main_vehicle.get_ID())

        # attach appropriate sensors to the vehicle
        track_sensor_settings = simstar.DistanceSensorParameters(
            enable=True,
            draw_debug=False,
            add_noise=False,
            position=simstar.PositionRPC(0.0, 0.0, -0.90),
            orientation=simstar.OrientationRPC(0.0, 0.0, 0.0),
            minimum_distance=0.2,
            maximum_distance=200.0,
            fov=190.0,
            update_frequency_in_hz=60.0,
            number_of_returns=self.track_sensor_size,
            query_type=simstar.QueryType.Static,
        )

        track_sensor = self.main_vehicle.add_sensor(
            simstar.ESensorType.Distance, track_sensor_settings
        )

        self.simstar_step()

        opponent_sensor_settings = simstar.DistanceSensorParameters(
            enable=True,
            draw_debug=False,
            add_noise=False,
            position=simstar.PositionRPC(2.0, 0.0, 0.2),
            orientation=simstar.OrientationRPC(0.0, 0.0, 0.0),
            minimum_distance=0.0,
            maximum_distance=20.0,
            fov=216.0,
            update_frequency_in_hz=60.0,
            number_of_returns=self.opponent_sensor_size,
            query_type=simstar.QueryType.Dynamic,
        )

        opponent_sensor = self.main_vehicle.add_sensor(
            simstar.ESensorType.Distance, opponent_sensor_settings
        )

        self.main_vehicle = SensoredVehicle(
            self.main_vehicle, track_sensor, opponent_sensor, "ego_agent"
        )

        # add all actors to the acor list
        self.actor_list.append(self.main_vehicle)

        # include other vehicles
        if self.add_opponents:

            # define other vehicles with set speeds and initial locations
            for i in range(self.number_of_opponents):
                new_agent = None

                try:
                    if self.opponent_poses:
                        new_agent = self.client.spawn_vehicle_to(
                            vehicle_pose=self.opponent_poses[i],
                            initial_speed=0,
                            set_speed=self.agent_speeds[i],
                            vehicle_type=simstar.EVehicleType.F1Racing,
                        )
                    else:
                        new_agent = self.client.spawn_vehicle(
                            actor=self.main_vehicle,
                            distance=self.agent_locations[i],
                            lane_id=self.lane_ids[i],
                            initial_speed=0,
                            set_speed=self.agent_speeds[i],
                            vehicle_type=simstar.EVehicleType.F1Racing,
                        )
                except IndexError as e:
                    print(
                        "[SimstarEnv] agent_locations, agent_speeds, and lane_ids"
                        + "sizes has to be the same with number of opponents"
                    )
                self.simstar_step()
                track_sensor = new_agent.add_sensor(
                    simstar.ESensorType.Distance, track_sensor_settings
                )
                self.simstar_step()
                opponent_sensor = new_agent.add_sensor(
                    simstar.ESensorType.Distance, opponent_sensor_settings
                )
                self.simstar_step()

                new_agent = SensoredVehicle(
                    new_agent, track_sensor, opponent_sensor, f"agent_{i}"
                )

                # define drive controllers for each agent vehicle
                new_agent.set_controller_type(simstar.DriveType.Auto)
                # enable rvo driving
                new_agent.enable_rvo_driving(0.25)
                self.actor_list.append(new_agent)

            self.simstar_step()

        self.simstar_step()

        # set as display vehicle to follow from simstar
        self.client.display_vehicle(self.main_vehicle)

        self.simstar_step()

        # set drive type for ego vehicle
        self.main_vehicle.set_controller_type(self.driving_type)

        self.simstar_step()

        simstar_obs = self.get_simstar_obs(self.main_vehicle)
        observation = self.make_observation(simstar_obs)
        return observation

    def calculate_reward(self, simstar_obs):
        collision = simstar_obs["damage"]
        reward = 0.0
        done = False
        summary = {"end_reason": None}

        trackPos = simstar_obs["trackPos"]
        angle = simstar_obs["angle"]
        spd = np.sqrt(simstar_obs["speedX"] ** 2 + simstar_obs["speedY"] ** 2)

        progress = spd * (np.cos(angle) - np.abs(np.sin(angle)))
        reward = progress

        if collision:
            print("[SimstarEnv] collision with opponent vehicle")
            reward -= self.c_w * spd * spd

        if np.abs(trackPos) >= 0.9:
            print("[SimstarEnv] finish episode due to road deviation")
            reward = -300
            summary["end_reason"] = "road_deviation"
            done = True

        if progress < self.termination_limit_progress:
            if self.terminal_judge_start < self.time_step_slow:
                print("[SimstarEnv] finish episode due to agent is too slow")
                reward = -100
                summary["end_reason"] = "too_slow"
                done = True
        else:
            self.time_step_slow = 0

        self.progress_on_road = self.main_vehicle.get_progress_on_road()

        self.time_step_slow += 1

        return reward, done, summary

    def get_progress_on_road(self):
        return self.progress_on_road

    def step(self, action):
        # send inputs if it is driven via API
        if self.driving_type == simstar.DriveType.API:
            self.action_to_simstar(action, self.main_vehicle)

        # required to continue simulation in sync mode
        self.simstar_step()

        control_inputs = self.main_vehicle.get_control_inputs()

        simstar_obs = self.get_simstar_obs(self.main_vehicle)
        observation = self.make_observation(simstar_obs)
        reward, done, summary = self.calculate_reward(simstar_obs)

        if self.save:
            # save without normalizing
            self.saver.save_step(control_inputs, simstar_obs, self.main_vehicle.tag)
            # save agents
            actions = self.get_agent_actions()
            raw_obs = self.get_agent_observations(raw=True)
            for v, vehicle in enumerate(self.actor_list):
                if vehicle.get_ID() == self.main_vehicle.get_ID():
                    continue

                self.saver.save_step(
                    actions[v - 1],
                    raw_obs[v - 1],
                    vehicle.tag,
                )

        return observation, reward, done, summary

    def ms_to_kmh(self, ms):
        return 3.6 * ms

    def clear(self):
        self.client.remove_actors(self.actor_list)

    def end(self):
        self.clear()

    # [steer, accel] input
    def action_to_simstar(self, action, vehicle_ref):
        steer = float(action[0])
        accel = float(action[1])

        steer = steer * 0.5

        if accel >= 0:
            throttle = accel
            brake = 0.0
        else:
            brake = abs(accel)
            throttle = 0.0

        vehicle_ref.control_vehicle(steer=steer, throttle=throttle, brake=brake)

    def simstar_step(self):
        step_num = int(self.sync_step_num)
        if self.synronized_mode:
            for i in range(step_num):
                self.client.blocking_tick()
        else:
            time_diff_to_be = 1 / 60 * step_num
            time_diff_actual = time.time() - self.last_step_time
            time_to_wait = time_diff_to_be - time_diff_actual
            if time_to_wait > 0.0:
                time.sleep(time_to_wait)
        self.last_step_time = time.time()

    def get_simstar_obs(self, vehicle_ref):
        vehicle_state = vehicle_ref.get_vehicle_state_self_frame()
        speed_x_kmh = abs(self.ms_to_kmh(float(vehicle_state["velocity"]["X_v"])))
        speed_y_kmh = abs(self.ms_to_kmh(float(vehicle_state["velocity"]["Y_v"])))
        opponents = vehicle_ref.opponent_sensor.get_detections()
        track = vehicle_ref.track_sensor.get_detections()
        road_deviation = vehicle_ref.get_road_deviation_info()

        retry_counter = 0
        while (
            len(track) < self.track_sensor_size
            or len(opponents) < self.opponent_sensor_size
        ):
            self.simstar_step()
            time.sleep(0.1)
            opponents = vehicle_ref.opponent_sensor.get_detections()
            track = vehicle_ref.track_sensor.get_detections()
            print("err")
            retry_counter += 1
            if retry_counter > 1000:
                raise RuntimeError("Track Sensor shape error. Exited")

        # deviation from road in radians
        angle = float(road_deviation["yaw_dev"])

        # hot fix for very rare angle observation difference
        if (abs(self.prev_angle - angle)>=(np.pi/2)):
            angle = self.prev_angle

        self.prev_angle = angle

        # deviation from road center in meters
        trackPos = float(road_deviation["lat_dev"]) / self.road_width

        # if collision occurs, True. else False
        damage = bool(vehicle_ref.check_for_collision())

        simstar_obs = {
            "angle": angle,
            "speedX": speed_x_kmh,
            "speedY": speed_y_kmh,
            "opponents": opponents,
            "track": track,
            "trackPos": trackPos,
            "damage": damage,
        }

        return simstar_obs

    def make_observation(self, simstar_obs):
        names = ["angle", "speedX", "speedY", "opponents", "track", "trackPos"]
        Observation = col.namedtuple("Observation", names)

        # normalize observation
        return Observation(
            angle=np.array(simstar_obs["angle"], dtype=np.float32) / 1.0,
            speedX=np.array(simstar_obs["speedX"], dtype=np.float32)
            / self.default_speed,
            speedY=np.array(simstar_obs["speedY"], dtype=np.float32)
            / self.default_speed,
            opponents=np.array(simstar_obs["opponents"], dtype=np.float32) / 20.0,
            track=np.array(simstar_obs["track"], dtype=np.float32) / 200.0,
            trackPos=np.array(simstar_obs["trackPos"], dtype=np.float32) / 1.0,
        )

    def get_agent_observations(self, raw=False):
        states = []
        for vehicle in self.actor_list:
            if vehicle.get_ID() != self.main_vehicle.get_ID():
                raw_state = self.get_simstar_obs(vehicle)
                if raw:
                    states.append(raw_state)
                else:
                    pretty = self.make_observation(raw_state)
                    states.append(pretty)

        return states

    def set_agent_actions(self, action_list):
        num_actions = len(action_list)
        num_agents = len(self.actor_list) - 1
        if num_actions == num_agents:
            action_index = 0
            for vehicle in self.actor_list:
                if vehicle.get_ID() != self.main_vehicle.get_ID():
                    action = action_list[action_index]
                    self.action_to_simstar(action, vehicle)
                    action_index += 1
        else:
            print("[SimstarEnv] Warning! Agent number not equal to action number")

    def get_agent_actions(self):
        action_list = []
        for vehicle in self.actor_list:
            if vehicle.get_ID() != self.main_vehicle.get_ID():
                control_inputs = vehicle.get_control_inputs()
                action_list.append(control_inputs)

        return action_list

    def change_opponent_control_to_api(self):
        self.simstar_step()
        for vehicle in self.actor_list:
            vehicle.set_controller_type(simstar.DriveType.API)

    def __del__(self):
        # reset sync mod so that user can interact with simstar
        if self.synronized_mode:
            pass
            # TODO: reset sync mod gives timeout, fix
            # self.client.set_sync_mode(False)

        if self.save:
            del self.saver
