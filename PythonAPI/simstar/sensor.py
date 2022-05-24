
import msgpackrpc

from .types import *

class Sensor(object):
    def __init__(self, client, ID, type):
        self._ID = int(ID)
        self.client = client
        
        self.type = type

    def get_ID(self):
        return self._ID
        
    def get_detections(self, delta_time=0.05):
        method_name = self.get_method()
        if(method_name == "GetPointCloud"):
            return self.client.call(method_name, self._ID, delta_time)
        elif(method_name != "Undefined"):
            return self.client.call(method_name, self._ID)

    def get_lane_detections(self):
        if (self.type == ESensorType.Vision):
            return self.client.call("GetVisionLanePoints", self._ID)
        else:
            return None
    
    def set_post_process(self,post_process_settings):
        if self.type == ESensorType.Camera:
            return self.client.call("SetPostProcess", self._ID,post_process_settings)
        else:
            return None
         
    def get_detections_async(self):
        if self.type == ESensorType.Imu:
            return self.client.call("GetIMUMeasurementsAsync",self._ID)
        else:
            return None

    def get_annotations(self):
        if (self.type == ESensorType.Vision):
            return self.client.call("GetVisionAnnotations",self._ID)
        else:
            return None
    

    def get_highway_lanes(self):
        """
        return is HighwayLaneInfo: left_lane, next_left_lane, right_lane, next_right_lane
        """
        if (self.type == ESensorType.Vision):
            return self.client.call("GetHighwayLanePoints",self._ID)
        else:
            return None
        
    def get_method(self):
        return {
            ESensorType.Camera: "GetCameraImage",
            ESensorType.Radar: "GetRadarSensorDetections",
            ESensorType.Lidar: "GetPointCloud",
            ESensorType.Vision: "GetSmartVisionSensorDetections",
            ESensorType.Distance: "GetDistanceSensorDetections",
            ESensorType.Imu: "GetIMUMeasurements",
            ESensorType.Gnss: "GetGNSSOutput",
            ESensorType.Encoder: "GetEncoderOutput",
            ESensorType.RoadSurface:"GetRoadSurfaceOutput",
            ESensorType.SemanticCamera:"GetSegmentationImage",
        }.get(self.type, "Undefined")

    def remove_sensor(self):
        return self.client.call("RemoveSensor", self._ID)
        
    def set_sensor_visibility(self, is_visible):
        self.client.call("SetSensorVisibility", self._ID, is_visible)