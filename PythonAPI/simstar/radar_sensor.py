from .types import *
from .sensor import *


class RadarSensor(Sensor):
    def __init__(self, client, ID):
        super(RadarSensor, self).__init__(client, ID, ESensorType().Radar)
