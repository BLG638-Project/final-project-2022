from .types import *
from .sensor import *

class DistanceSensor(Sensor):
    def __init__(self, client, ID):
        super(DistanceSensor, self).__init__(client, ID, ESensorType().Distance)
