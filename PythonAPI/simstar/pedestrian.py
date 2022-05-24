


import msgpackrpc

from .types import *
from .distance_sensor import *
from .radar_sensor import *
from .imu_sensor import *
from .smart_vision_sensor import *

class Pedestrian():
    def __init__(self,client,ID):
        self._ID = int(ID)
        self.client = client

    def get_ID(self):
        return self._ID

    def start_moving(self, use_path_planning=True):
        self.client.call("StartPedestrianMoving", self._ID, use_path_planning)

    def set_navigation(self, walking_speed,vision_range):
        self.client.call("SetPedestrianNavigation", self._ID, walking_speed,vision_range)

    def set_target_location(self, target_location):
        self.client.call("SetPedestrianTargetLocation", self._ID, target_location)

    def set_walking_speed(self, walking_speed):
        self.client.call("SetPedestrianWalkingSpeed", self._ID, walking_speed)

    def set_random_walkable_location(self, area_radius=50):
        return self.client.call("SetRandomWalkableLocation", self._ID, area_radius)
        
    def set_tag(self, tag):
        return self.client.call("SetPedestrianTag", self._ID, tag)
        
    def add_target_locations(self, targets):
        return self.client.call("AddPedestrianTargetLocations", self._ID, targets)
        
    def get_transform(self):
        return self.client.call("GetPedestrianTransform", self._ID)