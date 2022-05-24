
import msgpackrpc

from .types import *
from .traffic_sign import *

class SpeedLimit(TrafficSign):
    def __init__(self, client, ID):
        super(SpeedLimit, self).__init__(client, ID)
        
