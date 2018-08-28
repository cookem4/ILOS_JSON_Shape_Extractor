from  shapely.geometry import Point, Polygon, LineString
import shapely.geometry as geometry
import math
import json
from pprint import pprint
import matplotlib.pyplot as plt
import os,sys,glob,math,GPy
import pickle
import numpy as np
import pandas as pd
import copy
from scipy.stats import norm
from collections import Counter
import matplotlib.pyplot as plt
from collections import namedtuple  
import itertools
import matplotlib.path as mplPath
from descartes.patch import PolygonPatch
from matplotlib.collections import LineCollection
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import numpy as np

#CHANGE BUILDING NAME HERE
###########################################################
DESIRED_BUILDING = "Information Technology Building"#######
###########################################################

def getBuildingPolyCoords():
    data = []
    #CHANGE FLOOR FILE HERE
    ##################################################################################################################
    with open("D:\Research\MacQuest\MacQuestX-master\MacQuestX-master\MapFiles\outline\outline.geojson") as f:########
    ##################################################################################################################
         data = json.load(f)
    desiredBuildingOutline = []
    for j in range(len(data["features"])):
        textString = str(data["features"][j])
        if(textString.find(DESIRED_BUILDING) != -1):
            desiredBuildingOutline.append(j)
    building_gps_polygon = []
    global xCoordsBuilding
    global yCoordsBuilding
    xCoordsBuilding = []
    yCoordsBuilding = []
    for i in desiredBuildingOutline:
        buildingInfo = str(data["features"][i])
        coordinates = buildingInfo[buildingInfo.index("[[")+2:buildingInfo.index("]]")+1]
        coordList = coordinates.split("]")
        del coordList[len(coordList)-1]
        for k in range(len(coordList)):
            coordList[k] = coordList[k].replace("[", "")
        for k in range(len(coordList)):
            tempCoord = coordList[k].split(",")
            if(tempCoord[0] == ""):
                del tempCoord[0]
            xCoordsBuilding.append(float(tempCoord[0]))
            yCoordsBuilding.append(float(tempCoord[1]))
            building_gps_polygon.append((float(tempCoord[1]), float(tempCoord[0])))
    return building_gps_polygon


building_local_polygon = getBuildingPolyCoords()
building_metre_polygon = []
lon0 = np.min(xCoordsBuilding)
lat0 = np.min(yCoordsBuilding)

#Functions from yongyong's code
def frange(start, end, step):
    ret = []
    tmp = start
    ret.append(tmp)
    while(tmp < end):
        tmp += step
        ret.append(tmp)
    return ret

def distance(lat1,lon1,lat2,lon2):
    radius = 6378.137 # km
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d*1000

def offset_coord(lat0,lon0,lat1,lon1):
    x_project = (lat0,lon1)
    y_project = (lat1,lon0)
    x_offset = distance(lat0,lon0,x_project[0],x_project[1])
    y_offset = distance(lat0,lon0,y_project[0],y_project[1])
    #if at the sough that the origin flip the sign
    if lat1 < lat0:
        y_offset = y_offset * (-1)
    #if at the west of the origin flip the sign
    if lon1 < lon0:
        x_offset = x_offset * (-1)
            
    #return x offset and y offset
    return x_offset, y_offset

def get_local_polygon(lat0,lon0,gpspo,grid_step=1,plot=True):
    #Convert the gps polygon to local polygon and generate candidate locations
    local_coords = [offset_coord(lat0,lon0,lat,lon) for lat,lon in gpspo]
    local_coords = np.array(local_coords)
    min_X, min_Y, max_X, max_Y = (np.amin(local_coords[:,0]), np.amin(local_coords[:,1]), np.amax(local_coords[:,0]), np.max(local_coords[:,1]))
    mesh = np.meshgrid(frange(min_X,max_X,grid_step),frange(min_Y,max_Y,grid_step))
    grid_coords = np.column_stack((np.ravel(mesh[0]),np.ravel(mesh[1])))
    shape_polygon = Polygon(local_coords)
    valid_coords = []
    valid_points = []
    for loc in grid_coords:
        if shape_polygon.contains(Point(loc[0],loc[1])):
            valid_coords.append(loc)
            valid_points.append(loc)
    valid_coords = np.array(valid_coords)
    
    if plot==True:
        plt.plot(*zip(*local_coords))
        plt.scatter(valid_coords[:,0],valid_coords[:,1],color='r',marker='*')
    return valid_points

COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }

def v_color(ob):
    return COLOR[ob.is_valid]

def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)
def get_room_list():
    roomList = []
    data = []
    with open("D:\Research\MacQuest\MacQuestX-master\MacQuestX-master\MapFiles\layer1rooms.geojson") as f:
        data = json.load(f)

    desiredBuildingIndices = []
    for j in range(len(data["features"])):
        textString = str(data["features"][j])
        if(textString.find(DESIRED_BUILDING)!=-1):
            desiredBuildingIndices.append(j)
    global xCoords
    xCoords = []
    global yCoords
    yCoords = []
    for i in desiredBuildingIndices:
            buildingInfo = str(data["features"][i])
            coordinates = buildingInfo[buildingInfo.index("[[[") + 3:buildingInfo.index("]]]")]
            coordinates = coordinates.replace("[", "")
            coordListStr = coordinates.split("]")
            coordListInt = []
            localRoomCoords = []
            for x in range(len(coordListStr)):
                for j in coordListStr:
                    if(j.index(",") == 0):
                            coordListStr[coordListStr.index(j)] = coordListStr[coordListStr.index(j)][1:]
                            j = j[1:]
                    if coordListStr.index(j) == x: #chooses what point to use of room
                        longLat = j.split(",")
                        coordListInt.append([float(longLat[0]), float(longLat[1])])
                    
                for k in coordListInt:
                    localRoomCoords.append((k[1],k[0]))
                    xCoords.append(k[0])
                    yCoords.append(k[1])
            roomList.append(localRoomCoords)
    
    return roomList
roomList = get_room_list()
def removeEntries(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def showPoly(threshold):
    #this outputs blue and white polygon
    polygons = []
    tempx = []
    tempy = []
    for i in roomListLocal:
        tempx.append(i[0])
        tempy.append(i[1])
        polygon = Polygon(i)
        polygon = polygon.buffer(threshold, resolution=16, cap_style=2, join_style=3, mitre_limit=0.1)
        polygons.append(polygon)
    u = cascaded_union(polygons)
    polygon = Polygon(building_metre_polygon)
    global newPoly
    newPoly = polygon.difference(u)
    newPoly = newPoly.buffer(threshold, resolution=16, cap_style=2, join_style=3, mitre_limit=0.1)
    fig = plt.figure(1, figsize=5000, dpi=1000)
    ax = fig.add_subplot(111)
    print(newPoly)
    #plot_coords(ax, newPoly.exterior)
    patch = PolygonPatch(newPoly, facecolor=v_color(newPoly), edgecolor=v_color(newPoly), alpha=0.5, zorder=2)
    xrange = [-10, np.max(tempx) + 10]
    yrange = [-10, np.max(tempy) + 10]
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_aspect(1)
    ax.add_patch(patch)
    ax.set_aspect(1)
    plt.show()

def output():
    plotX = []
    plotY = []
    outputILOSLOCS = open("C://Users//mitch//Desktop//iloslocs.txt",'w')
    finalCoordOptions = []
    for i in onlyTheHallways:
        finalCoordOptions.append((i.x,i.y))
        print(i.x,i.y,file = outputILOSLOCS)
        plotX.append(i.x)
        plotY.append(i.y)
    plt.plot(plotX,plotY, 'r*')
    plt.show()
    finalCoordOptions = np.array(finalCoordOptions)
    outputILOSLOCS.close()


#also functions to check if the neighbour exists or not
def doesPointExist(x,y):
    exists = False
    for i in listOfValidPoints:
        if i.x == x and i.y == y:
            del listOfValidPoints[listOfValidPoints.index(i)]
            exists = True
    return exists
def doesPointExist2(x,y):
    exists = False
    for i in onlyTheHallways:
        if i.x == x and i.y == y:
            exists = True
    return exists
def getYMin(x):
    tempy = []
    for i in onlyTheHallways:
        if i.x == x:
            tempy.append(i.y)
    if(len(tempy)>0):
        return np.min(tempy)
def numNeighbours(x,y):
    numNeighbours = 0
    for i in onlyTheHallways:
        if i.x == x-1 and i.y == y:
            numNeighbours+=1
        elif i.y == y+1 and i.x == x:
            numNeighbours+=1
        elif i.x == x+1 and i.y == y:
            numNeighbours+=1
        elif i.y == y-1 and i.x == x:
            numNeighbours+=1
    return numNeighbours
def trimCoords(onlyTheHallways):
    clusters = []
    xtemp = []
    for i in onlyTheHallways:
        xtemp.append(i.x)
    minx = np.min(xtemp)
    ytemp = []
    for i in onlyTheHallways:
        if i.x == minx:
            ytemp.append(i.y)
    miny = np.min(ytemp)
    x = minx
    y = miny
    finalList = []
    # here is building specific
    while(x + 1 < 70):
        if(y > 76):
            x = x+1
            y = getYMin(x)
        #Check right and add that neighbour to the cluster. If no item to the right check up
        #and add the up to the cluster
        if(doesPointExist2(x,y)):
            if numNeighbours(x,y)<4:
                finalList.append(Point(x,y))
        y = y+1
    return finalList
def notInHallwayList(x,y):
    exists = False
    for i in onlyTheHallways:
        if i.x == x and i.y == y:
            del onlyTheHallways[onlyTheHallways.index(i)]
            exists = True
    return not exists
#Called recursively to branch off and add all items within the same hallway to a final list
def keepHallways(x,y):
    onlyTheHallways.append(Point(x,y))
    #Will check up, right, down left and if the point exists in those pisitions will add it to the list them mvoe to that position
    if(doesPointExist(x,y+1) and notInHallwayList(x,y+1)):
        #recursion
        keepHallways(x,y+1)
    if(doesPointExist(x+1,y) and notInHallwayList(x+1,y)):
        #recursion
        keepHallways(x+1,y)
    if(doesPointExist(x,y-1) and notInHallwayList(x,y-1)):
        #recursion
        keepHallways(x,y-1)
    if(doesPointExist(x-1,y) and notInHallwayList(x-1,y)):
        #recursion
        keepHallways(x-1,y)
        
#For convex hull computation
class ConvexHull(object):  
    _points = []
    _hull_points = []

    def __init__(self):
        pass

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):
        '''
        Returns the orientation of the Point p1 with regards to Point p2 using origin.
        Negative if p1 is clockwise of p2.
        :param p1:
        :param p2:
        :return: integer
        '''
        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference

    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points

        # get leftmost point
        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:

            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2

            self._hull_points.append(far_point)
            point = far_point

    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points

    def display(self):
        # all points
        plt.clf()
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, marker='D', linestyle='None')

        # hull points
        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy)
        plt.title('Convex Hull')
        plt.show()
#Used for concave hull
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
    coords = np.array([point.coords[0]
                       for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points
#Used for concave hull
def plot_polygon(polygon):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return fig
#Used for concave hull
def concaveHull():
    for i in range(1,10):
        alpha = i/10
        new_points = onlyTheHallways
        concave_hull, edge_points = alpha_shape(new_points,
                                                alpha=alpha)
        #print concave_hull
        lines = LineCollection(edge_points)
        plt.figure(figsize=(10,10))
        plt.title('Alpha={0} Delaunay triangulation'.format(
            alpha))
        plt.gca().add_collection(lines)
        delaunay_points = np.array([point.coords[0]
                                    for point in new_points])
        if(i==7):
            print(concave_hull)
        plt.plot(delaunay_points[:,0], delaunay_points[:,1],
                'o', hold=1, color='#f16824')
        plot_polygon(concave_hull)
        plt.plot(x,y,'o', color='#f16824')


for k in range(len(roomList)):
    roomList[k] = removeEntries(roomList[k])

validPredictionCoords = get_local_polygon(lat0,lon0, building_local_polygon,grid_step=1,plot=True)
roomListLocal = []
for i in roomList:
    roomListLocal.append([offset_coord(lat0,lon0,lat,lon) for lat,lon in i])
plt.clf()
building_metre_polygon = [offset_coord(lat0,lon0,lat,lon) for lat,lon in building_local_polygon]

#Modify this value between 0 and 1 to change fill threshold for polygon generated
#Outputs an image of the multipolygon as well as printing the multipolygon to the log
#####################
showPoly(0.9)########
#####################


##########################################################
#BELOW HERE ARE OTHER METHODS OF EXTRACTING HALLWAYS######
##########################################################

#CAN UNCOMMENT THEM TO TEST, HOWEVER SHOWPOLY IS THE BEST

'''
#Makes convex hull polygon wrapping rooms
hullExteriorPoints = []
point = namedtuple('Point', 'x y')
ch = ConvexHull()
for k in building_metre_polygon:
    ch.add(point(k[0], k[1]))
ch.get_hull_points()
ch.display()
xList = []
yList = []
for k in ch._hull_points:
    hullExteriorPoints.append((k.x,k.y))
    xList.append(k.x)
    yList.append(k.y)
#limits weather to shrink or expand x & y coords. Totally relative to building size however!!!!
xMax = np.max(xList)
yMax = np.max(yList)
xMin = np.min(xList)
yMin = np.min(yList)
centreX = (xMax + xMin)/2
centreY = (yMax + yMin)/2
xlim = centreX
ylim = centreY
#Contract exterior points in slightly
for i in range(len(hullExteriorPoints)):
    if hullExteriorPoints[i][0] > xlim and hullExteriorPoints[i][1]>ylim:
        hullExteriorPoints[i] = (hullExteriorPoints[i][0]-3,hullExteriorPoints[i][1]-3)
    elif hullExteriorPoints[i][0] < xlim and hullExteriorPoints[i][1]>ylim:
        hullExteriorPoints[i] = (hullExteriorPoints[i][0]+3,hullExteriorPoints[i][1]-3)
    elif hullExteriorPoints[i][0] > xlim and hullExteriorPoints[i][1]<ylim:
        hullExteriorPoints[i] = (hullExteriorPoints[i][0]-3,hullExteriorPoints[i][1]+3)
    elif hullExteriorPoints[i][0] < xlim and hullExteriorPoints[i][1]<ylim:
        hullExteriorPoints[i] = (hullExteriorPoints[i][0]+3,hullExteriorPoints[i][1]+3)
convexHullPolygon = Polygon(hullExteriorPoints)
    

listOfPredictionPoints = []
for i in validPredictionCoords:
    listOfPredictionPoints.append(Point(i[0],i[1]))
listOfValidPoints = []
for i in listOfPredictionPoints:
    if i.within(newPoly) and i.within(convexHullPolygon):
        listOfValidPoints.append(i)
plt.clf()
#check over this if it erodes the hallways too much
#listOfValidPoints = trimCoords(listOfValidPoints)
xtemp = []
for i in listOfValidPoints:
    xtemp.append(i.x)
minx = np.min(xtemp)
ytemp = []
for i in listOfValidPoints:
    if i.x == minx:
        ytemp.append(i.y)
miny = np.min(ytemp)
x = minx
y = miny
sys.setrecursionlimit(2000)
onlyTheHallways = []
keepHallways(x,y)

concaveHull()
output()
'''
