# -*- coding: utf-8 -*-
from enum import Enum

class Environments(Enum):
    Highway = 0
    OffRoad = 1
    DownTown = 2
    ParkingLot = 3
    ITU = 4
    Racing = 5

class EDriverProfile():
    calm = 0
    moderate = 1
    aggressive = 2

class ETrafficLightState():
    red = 0
    yellow = 1
    green = 2

class SimCoordinateFrame(Enum):
	LeftHand = 0,
	RightHand = 1

class ESpeedLimitState():
    SpeedLimit_0 = 0
    SpeedLimit_10 = 1
    SpeedLimit_20 = 2
    SpeedLimit_30 = 3
    SpeedLimit_40 = 4
    SpeedLimit_50 = 5
    SpeedLimit_60 = 6
    SpeedLimit_70 = 7
    SpeedLimit_80 = 8
    SpeedLimit_90 = 9
    SpeedLimit_100 = 10
    SpeedLimit_110 = 11
    SpeedLimit_120 = 12
    SpeedLimit_130 = 13

class ETrafficSign():
    SpeedLimit_10 = 1
    SpeedLimit_20 = 2
    SpeedLimit_30 = 3
    SpeedLimit_40 = 4
    SpeedLimit_50 = 5
    SpeedLimit_60 = 6
    SpeedLimit_70 = 7
    SpeedLimit_80 = 8
    SpeedLimit_90 = 9
    SpeedLimit_100 = 10
    SpeedLimit_110 = 11
    SpeedLimit_120 = 12
    SpeedLimit_130 = 13
    ExitSign = 14
    StopSign = 15
    TrafficLightRed = 16
    TrafficLightYellow = 17
    TrafficLightGreen = 18

class CruiseControl():
    NoCruiseAssist = 0
    ACC = 1
    CC = 2

class LaneAssist():
    NoLaneAssist = 0
    LCC = 1

class EVehicleModel():
    PhysX = 0
    DiscreteModel = 1
    ComplexModel = 2
    SimpleModel = 3

class EControlMethod():
    Throttle = 4
    Acceleration = 2
    Speed = 3
    EngineTorque = 1

class EVehicleType():
    Sedan1 = 0
    Sedan2 = 1
    Hatchback1 = 2
    Hatchback2 = 3
    Hatchback3 = 4
    Suv1 = 5
    Suv2 = 6
    Suv3 = 7
    Sport = 8
    Pickup = 9
    BoxTruck1 = 10
    BoxTruck2 = 11
    FlatbedTrack1 = 12
    FlatbedTrack2 = 13
    Motorbike1 = 14
    Motorbike2 = 15
    Motorbike3 = 16
    Bus = 17
    Bicycle = 18
    DeliveryRobot = 19
    Tractor = 20
    F1Racing = 21
    CargoBike = 22
    SedanSuspension = 23
    Jaguar = 24
    JaguarSuspension = 25

class ECoordinateFrame():
    World = 0
    Vehicle = 1
    Sensor = 2

class EEventType():
    NoVal = 0
    LeftLaneChange = 1
    RightLaneChange = 2

class ETrajectoryType():
    Base = 0
    Longitudinal = 1
    Lateral = 2

class EDirection():
    Right = 0
    Left = 1
    Straight = 2


class EWalkingDirection():
    Right_to_left = 0
    Left_to_right = 1
    Straight = 2

class ESignalDirection():
    Left = 0
    Right = 1
    Both = 2
    
class EColor():
    Black = 0
    White = 1
    Red = 2
    Green = 3
    Blue = 4
    Yellow = 5

class EPedestrianType():
    Boy1 = 0
    Girl1 = 1
    Male1 = 2
    Female1 = 3
    Male2 = 4
    Male3 = 5
    Male4 = 6
    Male5 = 7
    Male6 = 8
    Male7 = 9
    Male8 = 10
    Male9 = 11
    Female2 = 12
    Female3 = 13
    Female4 = 14
    Female5 = 15
    Female6 = 16
    Female7 = 17
    Female8 = 18
    
class EAnimal():
    DeerStag = 0
    DeerDoe = 1
    Fox = 2
    Pig = 3
    Wolf = 4
    
class ETrafficDensity():
    Low = 1
    Medium = 2
    High = 3
    
class EPosition():
    Back = 0
    Center = 1
    Front = 2

class TrackName():
    IstanbulPark = "Ist"
    DutchGrandPrix ="Dutch"
    HungaryGrandPrix = "Hungary"

class ChangeDirection():
    NoChange = 0
    Left = 1
    Right = 2

class DriveType():
    NotSet = -1
    Auto = 0
    Keyboard = 1
    API = 2
    Matlab = 3
    Teleport = 4

class QueryType():
    All = 0
    Dynamic = 1
    Static = 2
    
class DrivetrainType():
    MaxDegreeOfFreedom = 0
    ForwardOnly = 1

class ESpeedBump():
    Bump = 0
    Table = 1
    Dip = 2
    
class EDrawType():
    NoDrawing = 0
    Straight = 1
    Cross = 2

class MsgpackMixin:
    def __repr__(self):
        from pprint import pformat
        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4, width=1)

    def to_msgpack(self, *args, **kwargs):
        return self.__dict__

    @classmethod
    def from_msgpack(cls, encoded):
        obj = cls()
        #obj.__dict__ = {k.decode('utf-8'): (from_msgpack(v.__class__, v) if hasattr(v, "__dict__") else v) for k, v in encoded.items()}
        obj.__dict__ = { k : (v if not isinstance(v, dict) else getattr(getattr(obj, k).__class__, "from_msgpack")(v)) for k, v in encoded.items()}
        #return cls(**msgpack.unpack(encoded))
        return obj

class VehicleControl(MsgpackMixin):
    throttle = 0.0
    steer    = 0.0
    brake    = 0.0

    def __init__(self, throttle = 0.0, steer = 0.0, brake = 0.0):
        self.throttle = throttle
        self.steer = steer
        self.brake = brake

class WayPoint(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 0.0

    def __init__(self, X = 0.0, Y = 0.0, Z = 0.0):
        self.X = X
        self.Y = Y
        self.Z = Z

class PositionRPC(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 0.0

    def __init__(self, X = 0.0, Y = 0.0, Z = 0.0):
        self.X = X
        self.Y = Y
        self.Z = Z


class OrientationRPC(MsgpackMixin):
    Roll  = 0.0
    Pitch = 0.0
    Yaw   = 0.0

    def __init__(self, Roll = 0.0, Pitch = 0.0, Yaw = 0.0):
        self.Roll  = Roll
        self.Pitch = Pitch
        self.Yaw   = Yaw

class WayPointOrientation(MsgpackMixin):
    Roll  = 0.0
    Pitch = 0.0
    Yaw   = 0.0

    def __init__(self, Roll = 0.0, Pitch = 0.0, Yaw = 0.0):
        self.Roll  = Roll
        self.Pitch = Pitch
        self.Yaw   = Yaw

class RadarDetection(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 0.0
    X_v = 0.0
    Y_v = 0.0
    Z_v = 0.0

    def __init__(self, X = 0.0, Y = 0.0, Z = 0.0,X_v=0.0,Y_v=0.0,Z_v=0.0):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.X_v = X_v
        self.Y_v = Y_v
        self.Z_v = Z_v


class VisionDetection(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 0.0

    def __init__(self, X = 0.0, Y = 0.0, Z = 0.0):
        self.X = X
        self.Y = Y
        self.Z = Z

class WayPointList():
    waypoint_list = []

class ActorVelocity():
    actor_velocity = []

class Actor(MsgpackMixin):
    actor_id = 1
    actor_class = 1
    actor_pos = WayPoint()
    def __init__(self, actor_id = 0.0, actor_class = 0.0, actor_pos = 0.0):
        self.actor_id = actor_id
        self.actor_class = actor_class
        self.actor_pos = actor_pos

class ActorSpecifications(MsgpackMixin):
    actor_id = 1
    actor_class = 1
    actor_pos = WayPoint()
    actor_waypoints = WayPointList()
    actor_velocity = ActorVelocity()

    def __init__(self, actor_id = 0.0, actor_class = 0.0, actor_pos = 0.0, actor_waypoints = 0.0, actor_velocity = 0.0):
        self.actor_id = actor_id
        self.actor_class = actor_class
        self.actor_pos = actor_pos
        self.actor_waypoints = actor_waypoints
        self.actor_velocity = actor_velocity

class RoadSpecifications(MsgpackMixin):
    road_waypoints = WayPointList()

    def __init__(self, road_waypoints = 0.0):
        self.road_waypoints = road_waypoints

class ESensorType(MsgpackMixin):
    Undefined = 0
    Camera = 1
    Radar = 2
    Lidar = 3
    Vision = 4
    Distance = 5
    Imu = 6
    Gnss = 7
    Encoder = 8
    RoadSurface = 9
    SemanticCamera = 10



class GeoLocation(MsgpackMixin):

    def __init__(self,latitude,
        longitude,
        altitude = 0):
        self.latitude =latitude
        self.longitude =longitude
        self.altitude =altitude

class SettingsGNSS(MsgpackMixin):

    def __init__(self,sensor_name="gps",bias_lat=0.0,bias_long=0.0,bias_alt=0.0,
        deviation_lat=0.0,deviation_long=0.0,
        deviation_alt=0.0):
        self.sensor_name = sensor_name
        self.bias_lat = bias_lat
        self.bias_long = bias_long
        self.bias_alt = bias_alt
        self.deviation_lat = deviation_lat
        self.deviation_long = deviation_long
        self.deviation_alt = deviation_alt


class CameraSensorSettings(MsgpackMixin):
    def __init__(self,
        sensor_name="",
        width=800,
        height=600,
        FOV_degree=90.0,
        target_gamma = 1.5,
        is_post_processing_enabled = False,
        position=PositionRPC(0,0,100),
        orientation=OrientationRPC(0,0,0)):
        self.sensor_name = sensor_name
        self.width = width
        self.height = height
        self.target_gamma = target_gamma
        self.is_post_processing_enabled = is_post_processing_enabled
        self.FOV_degree = FOV_degree
        self.position = position
        self.orientation = orientation

class PostProcessSettings(MsgpackMixin):
        depth_of_field_focal_distance = 20e2
        lens_flare_intensity = 0.1
        bloom_intensity=0.675
        def __init__(self,depth_of_field_focal_distance=20e2 ,
            lens_flare_intensity=0.1,
            bloom_intensity=0.675):
            self.depth_of_field_focal_distance = depth_of_field_focal_distance
            self.lens_flare_intensity = lens_flare_intensity
            self.bloom_intensity = bloom_intensity



class SensorSettingsBase(MsgpackMixin):
    enable = True;
    sensor_name = "";
    position = PositionRPC(0,0,0)
    orientation = OrientationRPC(0,0,0)
    frame = ECoordinateFrame.World;

class LaserSensorSettingsBase(SensorSettingsBase):
    range = 100
    hor_fov_lower = 0
    hor_fov_upper = 359
    ver_fov_lower = 10
    ver_fov_upper = -10
    add_noise = False
    draw_lines = False
    draw_points = False
    ignore_ground = False

class RadarSettingRPC(LaserSensorSettingsBase):
    sensor_name = "Radar00"
    detection_prob = 1.0
    radar_res_cm = 180

    def __init__(self, enable=True,range = 50.0, draw_lines = False, draw_points = True,
        position=PositionRPC(0,0,0.5), orientation=OrientationRPC(0,0,0),
        hor_fov_lower = -20, hor_fov_upper = 20,
        ver_fov_lower = 0, ver_fov_upper=20,
        add_noise = False,ignore_ground=True,sensor_name="Radar00",detection_prob=0.99,radar_res_cm=180,
        frame=ECoordinateFrame.Vehicle,enable_weather_effect=False):
            self.enable = enable
            self.range = range
            self.hor_fov_lower = hor_fov_lower
            self.hor_fov_upper = hor_fov_upper
            self.ver_fov_lower = ver_fov_lower
            self.ver_fov_upper = ver_fov_upper
            self.add_noise = add_noise
            self.draw_lines = draw_lines
            self.draw_points = draw_points
            self.ignore_ground = ignore_ground
            self.sensor_name = sensor_name
            self.position= position
            self.orientation= orientation
            self.detection_prob = detection_prob
            self.radar_res_cm = radar_res_cm
            self.frame = frame
            self.enable_weather_effect = enable_weather_effect

class VisionSettingRPC(LaserSensorSettingsBase):
    sensor_name = "Vision00"
    detection_prob = 1.0
    def __init__(self, enable=True,range = 50.0, draw_lines = False, draw_points = True,
        hor_fov_lower = -20, hor_fov_upper = 20,
        ver_fov_lower = 0, ver_fov_upper=20,
        add_noise = False,ignore_ground=True,sensor_name="Vision00",
        position=PositionRPC(0,0,0.5), orientation=OrientationRPC(0,0,0),
        detection_prob=0.99,frame=ECoordinateFrame.Vehicle):
            self.enable = enable
            self.range = range
            self.hor_fov_lower = hor_fov_lower
            self.hor_fov_upper = hor_fov_upper
            self.ver_fov_lower = ver_fov_lower
            self.ver_fov_upper = ver_fov_upper
            self.add_noise = add_noise
            self.draw_lines = draw_lines
            self.draw_points = draw_points
            self.ignore_ground = ignore_ground
            self.position= position
            self.orientation= orientation
            self.sensor_name = sensor_name
            self.detection_prob = detection_prob
            self.frame = frame

class RpcFVector(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 0.0
    def __init__(self,X=0.0,Y=0.0,Z=0.0):
        self.X = X
        self.Y = Y
        self.Z = Z

class LaneData(MsgpackMixin):
    Coordinates = RpcFVector()
    Curvature = [0.0]
    CurvatureDerivative = [0.0]
    HeadingAngle = 0.0
    LateralOffset = 0.0
    Strength = 0.0
    Width = 0.0
    Length = 0.0
    def __init__(self,Coordinates=0.0 , Curvature=0.0, CurvatureDerivative=0.0, HeadingAngle=0.0, LateralOffset=0.0, Strength=0.0, Width=0.0, Length=0.0):
            self.Coordinates = Coordinates
            self.Curvature =  Curvature
            self.CurvatureDerivative =  CurvatureDerivative
            self.HeadingAngle = HeadingAngle
            self.LateralOffset = LateralOffset
            self.Strength = Strength
            self.Width = Width
            self.Length = Length

class ControllerTestParamsRPC(MsgpackMixin):
    let_ego_control_steer = False
    let_ego_control_speed = False
    def __init__(self,let_ego_control_steer=False,let_ego_control_speed=False):
        self.let_ego_control_steer = let_ego_control_steer
        self.let_ego_control_speed = let_ego_control_speed


class PoseData(MsgpackMixin):
    location = PositionRPC(0,0,0)
    rotation = OrientationRPC(0,0,0)
    def __init__(self, x = 0.0, y = 0.0, z = 0.0, roll = 0.0, pitch = 0.0, yaw = 0.0):
        self.location = PositionRPC(x, y, z)
        self.rotation = OrientationRPC(roll, pitch, yaw)

class TrajectoryEvent(MsgpackMixin):
    def __init__(self, event_type = EEventType.NoVal, target_object = -1) -> None:
        self.event_type = event_type
        self.target_object = target_object

class TrajectoryCondition(MsgpackMixin):
    def __init__(self, relative_actor_id = -1, distance_to_actor = 0.0,
        duration_in_seconds=0, duration_in_meters=0,
        target_speed_in_ms=-1, event=TrajectoryEvent()):
        self.relative_actor_id = relative_actor_id
        self.distance_to_actor = distance_to_actor
        self.duration_in_seconds = duration_in_seconds
        self.duration_in_meters = duration_in_meters
        self.target_speed_in_ms = target_speed_in_ms
        self.event = event

## TrajectoryInfo in RPCActor corresponds to TrajectoryData here
class TrajectoryData(MsgpackMixin):
    def __init__(self, start_condition = TrajectoryCondition(), set_accel_in_mss = 0.0,
    end_condition = TrajectoryCondition(),
    lane_change_direction = 0, lane_change_duration = 0.0,
    trajectory_type = ETrajectoryType.Longitudinal):
        self.start_condition = start_condition
        self.set_accel_in_mss = set_accel_in_mss
        self.end_condition = end_condition
        self.lane_change_direction = lane_change_direction
        self.lane_change_duration = lane_change_duration
        self.trajectory_type = trajectory_type

class IntersectionTrajectory(MsgpackMixin):
    direction = EDirection.Straight
    angle = 45.0

    def __init__(self,
                 direction,
                 angle=45.0):
        self.direction = direction
        self.angle = angle


class EngineSetup(MsgpackMixin):
    torque_curve = []
    max_rpm = 0.0
    moi = 0.0
    damping_rate_full_throttle = 0.0
    damping_rate_zero_throttle_clutch_engaged = 0.0
    damping_rate_zero_throttle_clutch_disengaged = 0.0

    def __init__(self,
                 setup):
        self.torque_curve = setup['torque_curve']
        self.max_rpm = setup['max_rpm']
        self.moi = setup['moi']
        self.damping_rate_full_throttle = setup['damping_rate_full_throttle']
        self.damping_rate_zero_throttle_clutch_engaged = setup['damping_rate_zero_throttle_clutch_engaged']
        self.damping_rate_zero_throttle_clutch_disengaged = setup['damping_rate_zero_throttle_clutch_disengaged']

class DifferentialType():
    LimitedSlip_4W = 0
    LimitedSlip_FrontDrive = 1
    LimitedSlip_RearDrive = 2
    Open_4W = 3
    Open_FrontDrive = 4
    Open_RearDrive = 5

class DifferentialSetup(MsgpackMixin):
    differential_type = DifferentialType.LimitedSlip_4W
    front_rear_split = 0.0
    front_left_right_split = 0.0
    rear_left_right_split = 0.0
    centre_bias = 0.0
    front_bias = 0.0
    rear_bias = 0.0

    def __init__(self,
                 setup):
        self.differential_type = setup['differential_type']
        self.front_rear_split = setup['front_rear_split']
        self.front_left_right_split = setup['front_left_right_split']
        self.rear_left_right_split = setup['rear_left_right_split']
        self.centre_bias = setup['centre_bias']
        self.front_bias = setup['front_bias']
        self.rear_bias = setup['rear_bias']

class TransmissionSetup(MsgpackMixin):
    b_use_gear_auto_box = True
    gear_switch_time = 0.0
    gear_autoBox_latency = 0.0
    final_ratio = 0.0
    reverse_gear_ratio = 0.0
    neutral_gear_up_Ratio = 0.0
    clutch_strength = 0.0

    def __init__(self,
                 setup):
        self.b_use_gear_auto_box = setup['b_use_gear_auto_box']
        self.gear_switch_time = setup['gear_switch_time']
        self.gear_autoBox_latency = setup['gear_autoBox_latency']
        self.final_ratio = setup['final_ratio']
        self.reverse_gear_ratio = setup['reverse_gear_ratio']
        self.neutral_gear_up_Ratio = setup['neutral_gear_up_Ratio']
        self.clutch_strength = setup['clutch_strength']

class AvoidanceSetup(MsgpackMixin):
    b_use_rvo_avoidance = 0
    rvo_avoidance_radius = 0.0
    rvo_avoidance_height = 0.0
    avoidance_consideration_radius = 0.0
    rvo_steering_step = 0.0
    rvo_throttle_step = 0.0
    avoidance_weight = 0.0

    def __init__(self,
                 setup):
        self.b_use_rvo_avoidance = setup['b_use_rvo_avoidance']
        self.rvo_avoidance_radius = setup['rvo_avoidance_radius']
        self.rvo_avoidance_height = setup['rvo_avoidance_height']
        self.avoidance_consideration_radius = setup['avoidance_consideration_radius']
        self.rvo_steering_step = setup['rvo_steering_step']
        self.rvo_throttle_step = setup['rvo_throttle_step']
        self.avoidance_weight = setup['avoidance_weight']

class VehicleSetup(MsgpackMixin):
    steering_curve = []
    mass = 0.0
    chassis_width = 0.0
    chassis_height = 0.0
    min_normalized_tire_load = 0.0
    min_normalized_tire_load_filtered = 0.0
    max_normalized_tire_load = 0.0
    max_normalized_tire_load_filtered = 0.0
    threshold_longitudinal_speed = 0.0
    low_forward_speed_sub_step_count = 0.0
    high_forward_speed_sub_step_count = 0.0

    def __init__(self,
                 setup):
        self.steering_curve = setup['steering_curve']
        self.mass = setup['mass']
        self.chassis_width = setup['chassis_width']
        self.chassis_height = setup['chassis_height']
        self.min_normalized_tire_load = setup['min_normalized_tire_load']
        self.min_normalized_tire_load_filtered = setup['min_normalized_tire_load_filtered']
        self.max_normalized_tire_load = setup['max_normalized_tire_load']
        self.max_normalized_tire_load_filtered = setup['max_normalized_tire_load_filtered']
        self.threshold_longitudinal_speed = setup['threshold_longitudinal_speed']
        self.low_forward_speed_sub_step_count = setup['low_forward_speed_sub_step_count']
        self.high_forward_speed_sub_step_count = setup['high_forward_speed_sub_step_count']

class VehiclePhysicalProperties(MsgpackMixin):
    def __init__(self,
                 setup):
        self.vehicle_setup = VehicleSetup(setup['vehicle_setup'])
        self.transmission_setup = TransmissionSetup(setup['transmission_setup'])
        self.differential_setup = DifferentialSetup(setup['differential_setup'])
        self.engine_setup = EngineSetup(setup['engine_setup'])
        self.avoidance_setup = AvoidanceSetup(setup['avoidance_setup'])


class WaypointTrajectory(MsgpackMixin):
    X = 0.0
    Y = 0.0
    time = 0.0
    def __init__(self, X = 0.0, Y = 0.0, time = 0.0):
        self.X = X
        self.Y = Y
        self.time = time
        
class TrafficLightData(MsgpackMixin):
    light_id = -1
    light_state = ETrafficLightState().red
    red_time = 0.0
    yellow_time = 0.0
    green_time = 0.0
    is_frozen = False
    
    def __init__(self, light_id = -1, light_state = ETrafficLightState().red, red_time = 0.0, yellow_time = 0.0, green_time = 0.0, is_frozen = False):
        self.light_id = light_id
        self.light_state = light_state
        self.red_time = red_time
        self.yellow_time = yellow_time
        self.green_time = green_time
        self.is_frozen = is_frozen
        
class YawMode(MsgpackMixin):
    is_rate = True
    yaw_or_rate = 0.0
    def __init__(self, is_rate = True, yaw_or_rate = 0.0):
        self.is_rate = is_rate
        self.yaw_or_rate = yaw_or_rate

class ComplexVehicleParameters(MsgpackMixin):
    speed_interface_accel_max = 5
    speed_interface_accel_min = -5
    jerk_upper_limit = 5
    jerk_lower_limit = -5
    accel_request_filter_coef = 0.00999999977
    accel_req_time_delay_F = 0.00999999977
    accel_req_upper_lim = 5
    accel_req_lower_lim = -8
    steer_fac = [16] * 7

    def __init__(self, speed_interface_accel_max = 5, speed_interface_accel_min = -5,
                    jerk_upper_limit = 5, jerk_lower_limit = -5, accel_request_filter_coef = 0.00999999977,
                    accel_req_time_delay_F = 0.00999999977, accel_req_upper_lim = 5, accel_req_lower_lim = -8, steer_fac = [16] * 7):

        self.speed_interface_accel_max = speed_interface_accel_max
        self.speed_interface_accel_min = speed_interface_accel_min
        self.jerk_upper_limit = jerk_upper_limit
        self.jerk_lower_limit = jerk_lower_limit
        self.accel_request_filter_coef = accel_request_filter_coef
        self.accel_req_time_delay_F = accel_req_time_delay_F
        self.accel_req_upper_lim = accel_req_upper_lim
        self.accel_req_lower_lim = accel_req_lower_lim
        self.steer_fac = steer_fac[:7]
        ## Make sure it's length is 7
        if (len(self.steer_fac) < 7):
            self.steer_fac += [16] * (7 - len(self.steer_fac))

class LidarSettings(LaserSensorSettingsBase):
    number_of_channels = 16
    points_per_second = 100000
    horizontal_rotation_frequency = 20
    def __init__(self,number_of_channels=16, range=100, points_per_second=100000,
             horizontal_rotation_frequency=20,
             hor_fov_lower=0, hor_fov_upper=359, ver_fov_lower=-10,
            ver_fov_upper=1,position=PositionRPC(0,0,2.0),
            orientation=OrientationRPC(0,0,0), draw_points=False, draw_lines=False,
            enable_weather_effect =False):
        self.number_of_channels= number_of_channels
        self.range= range
        self.points_per_second= points_per_second
        self.horizontal_rotation_frequency= horizontal_rotation_frequency
        self.hor_fov_lower= hor_fov_lower
        self.hor_fov_upper= hor_fov_upper
        self.ver_fov_lower= ver_fov_lower
        self.ver_fov_upper= ver_fov_upper
        self.position= position
        self.orientation= orientation
        self.draw_points= draw_points
        self.draw_lines= draw_lines
        self.enable_weather_effect = enable_weather_effect

class DistanceSensorParameters(SensorSettingsBase):
    def __init__(self, enable = True, sensor_name = "", draw_debug = False,
        add_noise = True,
      position=PositionRPC(0,0,0.0), orientation=OrientationRPC(0,0,0),
      minimum_distance = 0.0, maximum_distance = 200.0,
      fov = 0.0, update_latency_in_seconds = 0.0,
      update_frequency_in_hz = 1000.0, unnorrelated_noise_sigma = 0.0,
      number_of_returns=1,query_type=QueryType.All ,
      frame=ECoordinateFrame.Sensor):
        self.enable = enable
        self.sensor_name = sensor_name
        self.draw_debug = draw_debug
        self.add_noise = add_noise
        self.position= position
        self.orientation= orientation
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance
        self.fov = fov
        self.update_latency_in_seconds = update_latency_in_seconds
        self.update_frequency_in_hz = update_frequency_in_hz
        self.unnorrelated_noise_sigma = unnorrelated_noise_sigma
        self.number_of_returns = number_of_returns
        self.query_type = query_type
        self.frame = frame

class ParkingGridInfo(MsgpackMixin):
    name = "Z0"
    is_available = True
    grid_width_in_m = 0.0
    grid_height_in_m = 0.0
    grid_skew_angle_in_degrees = 0.0
    lane_thickness = 0.0
    front_corner1 = PositionRPC(0.0,0.0,0.0)
    front_corner2 = PositionRPC(0.0,0.0,0.0)
    back_corner1 = PositionRPC(0.0,0.0,0.0)
    back_corner2 = PositionRPC(0.0,0.0,0.0)
    
class BumpProfile(MsgpackMixin):
	start_distance = 0.0
	height = 0.0
	width = 0.0
	depth = 0.0
	start_offset = 0.0
	direction = "forward"
	bump_type = ESpeedBump().Bump
	def __init__(self, start_distance=0.0, height=0.0, width=0.0, depth=0.0, start_offset=0.0, direction="forward", bump_type=ESpeedBump().Bump):
		self.start_distance = start_distance
		self.height = height
		self.width = width
		self.depth = depth
		self.start_offset = start_offset
		self.direction = direction
		self.bump_type = bump_type
