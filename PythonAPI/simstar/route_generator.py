
from .types import *
from .road import *

def decode_polyline(polyline_str):
    '''Pass a Google Maps encoded polyline string; returns list of lat/lon pairs'''
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}

    # Coordinates have variable length when encoded, so just keep
    # track of whether we've hit the end of the string. In each
    # while loop iteration, a single coordinate is decoded.
    while index < len(polyline_str):
        # Gather lat/lon changes, store them in a dictionary to apply them later
        for unit in ['latitude', 'longitude']: 
            shift, result = 0, 0

            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20:
                    break

            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)

        lat += changes['latitude']
        lng += changes['longitude']

        coordinates.append((lat / 100000.0, lng / 100000.0))

    return coordinates

# This function requires Esri's arcpy module.
def convert_to_shapefile(steps, output_shapefile):
    '''Pass the steps object returned by the Maps API (should be response['routes'][0]['legs'][0]['steps'])
    and an output shapefile path; outputs a detailed shapefile of that route'''
    
    import arcpy
    import os

    # Decode each step of the route; add those coordinate pairs to a list
    total_route = []
    for step in steps:
        total_route += decode_polyline(step['polyline']['points'])

    # Create empty WGS84 shapefile.
    sr = arcpy.SpatialReference(4326)
    arcpy.CreateFeatureclass_management(os.path.dirname(output_shapefile), os.path.basename(output_shapefile), 
        "POLYLINE", spatial_reference=sr)

    # Add points to array, write array to shapefile as a polyline
    arr = arcpy.Array()
    for coord_pair in total_route:
        arr.add(arcpy.Point(coord_pair[1], coord_pair[0]))
    with arcpy.da.InsertCursor(output_shapefile, ['SHAPE@']) as rows:
        rows.insertRow([arcpy.Polyline(arr)])
    del rows

    return output_shapefile

def convert_route_to_x_y(way):
    try:
        import utm
    except ImportError:
        raise ImportError("pip install utm")  

    offset = utm.from_latlon(float(way[0][0]), float(way[0][1]))
    route = []
    for node in way:
        x = float(node[0])
        y = float(node[1])
        utm_value = utm.from_latlon(x, y)
        coordinate = (offset[0] - utm_value[0], offset[1] - utm_value[1])
        route.append(coordinate)
    return route

class RouteGenerator():
    def __init__(self, client, coordinates):
        try:
            import openrouteservice
        except ImportError:
            raise ImportError("pip install openrouteservice")
        self.coordinates = coordinates
        self.eatron_client = client
        self.ors_client = openrouteservice.Client(key='5b3ce3597851110001cf62488bf48e2f997f47fabe61396280929615')
        
    def generate_route(self, road,use_altitude=False):
        routes = self.ors_client.directions(self.coordinates)
        route = (decode_polyline(routes['routes'][0]['geometry']))
        converted_route = convert_route_to_x_y(route)
        
        # for each coordinate get height information
        if use_altitude:
            format_in = 'polyline'
            response = self.ors_client.elevation_line(format_in,route)
            coords = response['geometry']['coordinates']
            heights = [x[2] for x in coords]
            min_height = min(heights)
            max_height = max(heights)

        
        way_points = []
        for i in range(len(converted_route)):
            
            x = converted_route[i][1]
            y = converted_route[i][0]
            z = 0.0

            if use_altitude:
                z = heights[i] - min_height 

            way_point = WayPoint(x,y,z)
            way_points.append(way_point)

        self.eatron_client.call("SetRouteForGivenRoad", road.get_ID(), way_points, False, True, True, 1.0)
    