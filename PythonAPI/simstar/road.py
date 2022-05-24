

import msgpackrpc

from .types import *



class RoadGenerator():
    def __init__(self,client,ID):
        self._ID = int(ID)
        self.client = client
        # damage, dirt, tar
        self._road_structure = [0.0, 0.0, 0.0]

    def get_ID(self):
        return self._ID
    
    def add_road(self,radius=100,angle=0,slope_angle=0,damage=None,dirt=None,tar=None):
        road_structure = [damage or 0, dirt or 0, tar or 0]
        if(self._road_structure != road_structure):
            self._road_structure = road_structure
            self.client.call("UpdateRoadStructure",self._ID,road_structure[0],road_structure[1],road_structure[2])
        if(slope_angle == 0):
            return self.client.call("AddRoadTo",self._ID,radius,angle)
        else:
            return self.client.call("AddSlopedRoadTo",self._ID,radius,angle,slope_angle)

    def update_entire_road_structure(self,damage=0,dirt=0,tar=0):
        self.client.call("UpdateEntireRoadStructure",self._ID,damage,dirt,tar)
    
    def add_road_bump(self,distance):
        self.client.call("SpawnRoadBump",self._ID,distance)

    def add_intersection(self,
                         add_forward_road=True,
                         add_left_road=True,
                         add_right_road=True,
                         green_light_duration=[10, 10, 10, 10],
                         yellow_light_duration=[2, 2, 2, 2],
                         is_traffic_lights_enabled=True):

        return self.client.call("AddIntersection",
                                self._ID,
                                add_forward_road,
                                add_left_road,
                                add_right_road,
                                green_light_duration,
                                yellow_light_duration,
                                is_traffic_lights_enabled)

