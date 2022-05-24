

import msgpackrpc

class RoadWork():
    def __init__(self,client,ID):
        self._ID = int(ID)
        self.client = client

    def get_ID(self):
        return self._ID
   
