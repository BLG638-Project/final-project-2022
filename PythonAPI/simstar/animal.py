


import msgpackrpc

from .types import *

class Animal():
    def __init__(self,client,ID):
        self._ID = int(ID)
        self.client = client

    def get_ID(self):
        return self._ID

    def move_to_target(self, target_location, movement_speed=5):
        return self.client.call("MoveAnimalToTarget", self._ID, target_location, movement_speed)
        
    def move_to_target_by_distance(self, target_location, movement_speed=5, relative_actor=None, relative_distance=10):
        if(relative_actor != None):
            return self.client.call("MoveAnimalToTargetByDistance", self._ID, target_location, movement_speed, relative_actor.get_ID(), relative_distance)
        else:
            return False
        
    def move_to_target_by_time(self, target_location, movement_speed=5, start_in=10):
        return self.client.call("MoveAnimalToTargetByTime", self._ID, target_location, movement_speed, start_in)