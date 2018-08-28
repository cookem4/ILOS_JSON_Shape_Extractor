from  shapely.geometry import Point, Polygon, LineString
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
from shapely.ops import cascaded_union

##############################
#Change these
##############################
DESIRED_BUILDING = "Information Technology Building"
BUILDING_FLOOR = "1"


def getBuildingPolyCoords():
    data = []
    #######################################
    #Please note that in MacQuest/MapFiles/outline there are several different files with a variety of names not related to floor number. I am not sure what these do yet
    #######################################
    
    if(BUILDING_FLOOR == "1"):
        with open("D:\Research\MacQuest\MacQuestX-master\MacQuestX-master\MapFiles\outline\outline.geojson") as f:
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
        coordinates = buildingInfo[buildingInfo.index("[[[")+2:buildingInfo.index("]]]")+1]
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

#gives the building exterior coordinates in degrees
building_local_polygon = getBuildingPolyCoords()
#origin from bounding box of building
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

def get_room_list():
    roomList = []
    data = []
    with open("D:\Research\MacQuest\MacQuestX-master\MacQuestX-master\MapFiles\layer" + BUILDING_FLOOR + "rooms.geojson") as f:
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
        if(str(data["features"][i]).find("stair")!=-1):
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

building_local_polygon_metre = [offset_coord(lat0,lon0,lat,lon) for lat,lon in building_local_polygon]
roomCoords = get_room_list()
roomListLocal = []
for i in roomCoords:
    roomListLocal.append([offset_coord(lat0,lon0,lat,lon) for lat,lon in i])
print("BUILDING POLY COORDS:")
print(building_local_polygon_metre)
print("ORIGIN OF BUILDING:")
print(lat0,lon0)
print("ALL STAIR COORDS:")
print(roomListLocal)


