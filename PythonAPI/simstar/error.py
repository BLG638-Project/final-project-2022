
from msgpackrpc.error import TransportError 
from msgpackrpc.error import TimeoutError

class VehicleError(Exception):
     def __init__(self, value):
         self.vehicle_id = value
     def __str__(self):
         return repr(self.vehicle_id)

class SensorError(Exception):
     def __init__(self, value):
         self.sensor_id = value
     def __str__(self):
         return repr(self.sensor_id)

class EgoError(Exception):
     def __init__(self, value):
         self.ego_id = value
     def __str__(self):
         return repr(self.ego_id)

