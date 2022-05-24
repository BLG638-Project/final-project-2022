from .types import *
from .sensor import *


class SmartVisionSensor(Sensor):
    def __init__(self, client, ID):
        super(SmartVisionSensor, self).__init__(client, ID, ESensorType().Vision)

    def get_annotations(self):
        return self.client.call("GetVisionAnnotations", self._ID)
