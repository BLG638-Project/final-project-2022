

import msgpackrpc

class TrafficSign():
    def __init__(self,client,ID):
        self._ID = int(ID)
        self.client = client

    def get_ID(self):
        return self._ID
   
    def remove(self):
        return self.client.call("RemoveTrafficSign", self._ID)