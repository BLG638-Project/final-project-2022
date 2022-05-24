

import msgpackrpc
from .types import *
from .sensor import *

class Multirotor():
    def __init__(self,client,name,id,timeout=10):
        self._name = name
        self._ID = id
        self.local_client = client
        server_info = self.local_client.call("GetServerNetInfo")
        if server_info == "":
            self.client = client
        else:
            ip_port = server_info.split(":")
            self.client = msgpackrpc.Client(msgpackrpc.Address(ip_port[0], int(ip_port[1])), timeout = timeout, \
                pack_encoding = 'utf-8', unpack_encoding = 'utf-8')

    def get_name(self):
        return self._name
   
    def get_ID(self):
        return self._ID
   
    def takeoff(self, timeout=5):
        return self.client.call_async("TakeOff", self._name, timeout)
        
    def hover(self):
        return self.client.call_async("Hover", self._name)
        
    def get_position(self):
        return self.client.call("GetPosition", self._name)
        
    def move_by_veloctiy_z(self, vx, vy, z, duration, drivetrain=DrivetrainType().MaxDegreeOfFreedom, yawmode=YawMode()):
        return self.client.call_async("MoveByVelocityZ", self._name, vx, vy, z, duration, drivetrain, yawmode)
        
    def move_to_position(self, x, y, z, velocity, timeout_sec=3e+38, drivetrain=DrivetrainType().MaxDegreeOfFreedom, yaw_mode=YawMode(),
        lookahead=-1, adaptive_lookahead=1):
        return self.client.call_async("MoveToPosition", self._name, x, y, z, velocity, timeout_sec, drivetrain, yaw_mode, lookahead, adaptive_lookahead)
        
    def get_sensor_method(self, sensor_type):
        return {
            ESensorType().Camera: "AddCameraSensor",
            ESensorType().Radar: "AddRadarSensor",
            ESensorType().Lidar: "AddLidar",
            ESensorType().Vision: "AddSmartVisionSensor",
            ESensorType().Distance: "AddDistanceSensor",
            ESensorType().Imu: "AddIMUSensor",
            ESensorType().Gnss: "AddGNSSSensor",
        }[sensor_type]    
        
    def add_sensor(self, sensor_type, sensor_parameters=None):
        print(self.local_client)
        print(self.client)
        if(sensor_parameters):
            sensor_id = self.local_client.call(self.get_sensor_method(sensor_type), self._ID, sensor_parameters)
        else:
            sensor_id = self.local_client.call(self.get_sensor_method(sensor_type), self._ID, "")
        return Sensor(self.local_client, sensor_id, sensor_type)
        
    def get_attached_sensor(self, sensor_name):
        sensors = self.local_client.call("GetAttachedSensor", self._ID, sensor_name)
        attached_sensors = []
        for sensor in sensors:
            attached_sensors.append(Sensor(self.local_client, sensor['ID'], sensor['sensor_type']))
        return attached_sensors
        