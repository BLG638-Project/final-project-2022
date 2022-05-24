 
import msgpackrpc

from .types import *
from .sensor import *

class ImuSensor(Sensor):
    def __init__(self, client, ID):
        super(ImuSensor, self).__init__(client, ID, ESensorType().Imu)
    
    def get_sensor_detections(self):
        print("ImuSensor::GetIMUMeasurements is deprecated, will be removed in the next release. Please use Sensor::get_detections!")
        return self.client.call("GetIMUMeasurements", self._ID)
