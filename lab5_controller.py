"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

# range = robot.getDevice('range-finder')
# range.enable(timestep)
# camera = robot.getDevice('camera')
# camera.enable(timestep)
# camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 2.58
pose_y     = 8.9
pose_theta = 0

vL = 0
vR = 0

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
#mode = 'manual' # Part 1.1: manual mode
mode = 'planner'
# mode = 'autonomous'

lidar_sensor_readings = []
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # remove blocked sensor rays

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

###################
#
# Planner
#
###################
if mode == 'planner':
# Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    map = np.load('/Applications/Robotics/Lab/CSCI3302_Lab5/controllers/map.npy')
    #plt.imshow(map, cmap='gray')
    #plt.show()
    kernel = np.ones((13,13))
    intermediate_config = convolve2d(map, kernel, mode='same')

    proper_map = np.zeros((360,360))

    for i in range(0,360):
        for j in range(0,360):
            if(intermediate_config[i][j] >= 28):
                proper_map[i][j] = 1
            
    # np.save('/Applications/Robotics/Lab/CSCI3302_Lab5/controllers/proper_map.npy', proper_map)
    plt.imshow(proper_map, cmap='gray')
    plt.show()

    start_w = (pose_x, pose_y) # (Pose_X, Pose_Z) in meters
    end_w = (10.0, 7.0) # (Pose_X, Pose_Z) in meters

    # Convert the start_w and end_W from webot's coordinate frame to map's
    start = (360 - int(start_w[1]*30), int((start_w[0]*30))) # (x, y) in 360x360 map
    end = (360 - int(end_w[1]*30), int((end_w[0])*30)) # (x, y) in 360x360 map
    print ("start: " , start)
    print ("end:", end)
# Part 2.3: Implement A* or Dijkstra's
    def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
    
        def reconstruct_path(cameFrom, ending, start):
            # print ("calling function")
            path = []
            value = ending
            while value != start:
                # if value == cameFrom[value]:
                # del cameFrom[value]
                # else:
                value = cameFrom[value]
                path.append(value)
            return path
    
        def heuristic_cost(neighbor, end):
            return np.linalg.norm(np.array(neighbor)- np.array(end)) # SHIV : Changed this because this is the correct way to calculate the distance
    
        closed = []  # nodes we've already looked at (explored)
        open = []  # frontier
        gscore = {}
        for i in range(0, 360):
            for j in range(0, 360):
                gscore[(i, j)] = float('inf')
    
        cameFrom = {start: None}
    
        hscore = {}
        for i in range(0, 360):
            for j in range(0, 360):
                hscore[(i, j)] = float('inf')
    
        fscore = {}
        for i in range(0, 360):
            for j in range(0, 360):
                fscore[(i, j)] = hscore[(i, j)] + gscore[(i, j)]
    
        gscore[start] = 0  # g
        hscore[start] = heuristic_cost(start, end)
        fscore[start] = heuristic_cost(start, end)
    
        open.append(start)
        # start = (30,253)
        current = start
        count = 0
    
        while len(open) != 0:
            # current = min(fscore)#min(fscore, key=fscore.get)
            min = float('inf')
            for key in open:
                if fscore[key] != float('inf') and fscore[key] < min:
                    current = key
                    min = fscore[key]
    
            # print ("new current node is: ", current)
            if current == end:
                print("just in case")
                print("path is: ", reconstruct_path(cameFrom, current, start))
                return reconstruct_path(cameFrom, current, start)
    
            open.remove(current)
            closed.append(current)
    
            neighbors = []
            for i in range(current[0] - 1, current[0] + 2):
                for j in range(current[1] - 1, current[1] + 2):
                    if (i >= 0 and i < 360 and j >= 0 and j < 360 and (i, j) != current):
                        neighbors.append((i, j))
    
            # print("neighbors we are looking at is: ", neighbors)
    
            for neighbor in neighbors:
                # (29, 252), (29, 253), (29, 254), (30, 252), (30, 254), (31, 252), (31, 253), (31, 254)]
                if neighbor in closed or map[neighbor[0]][neighbor[1]] == 1:
                    continue
    
                if neighbor not in open:
                    open.append(neighbor)
    
                # print ("current neighbor is: ", neighbor)
    
                # SHIV : Changed np.linalg.norm([neighbor, current]) TO np.linalg.norm(np.array(neighbor)- np.array(current))
                tentative_gscore = gscore[current] + np.linalg.norm(np.array(neighbor)- np.array(current))
                if tentative_gscore >= gscore[neighbor]:  # this is not a better path
                    # print ("entering here")
                    continue
    
                # SHIV : I was wrong about this. Sorry. This shoulkd be outside else because otherwise we won't have these
                # values for the new childen. It's only when the current g value is worse that we skip to update the value.
                gscore[neighbor] = tentative_gscore
                cameFrom[neighbor] = current
                fscore[neighbor] = gscore[neighbor] + heuristic_cost(neighbor, end)
    
                # print ("finish with ", neighbor)
            # print ("open is ", open)
        # print ("end")
        # print ("came from is: ",cameFrom)
        # print ("path is: ", reconstruct_path(cameFrom, current, start))
    
        return [] # SHIV : If it comes to this, this means no path was found because of obstacles
    
    
    # Part 2.1: Load map (map.npy) from disk and visualize it
    
    
    # Part 2.3 continuation: Call path_planner
    
    path = path_planner(proper_map, start, end)
    print("path is", path)
    
    
    # SHIV : overlaying path on the 'proper_map' via just changing the cell value
    for p in path: proper_map[p[0]][p[1]] = 2
    fig = plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(proper_map)
    plt.show()
    

# Part 2.4: Turn paths into goal points and save on disk as path.npy and visualize it

    for point in path:
        point = (point[1], point[0])
        # print ("here the point is: ", point)
    
    
    np.save('/Applications/Robotics/Lab/CSCI3302_Lab5/controllers/path.npy', path)

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
if mode == 'manual':
    map = np.zeros(shape=(360,360)) # Replace None by a numpy 2D floating point array


if mode == 'autonomous':
# Part 3.1: Load path from disk and visualize it (Make sure its properly indented)
    path = np.load('/Applications/Robotics/Lab/CSCI3302_Lab5/controllers/path.npy')
    print ("path is: ", path)
    map = np.load('/Applications/Robotics/Lab/CSCI3302_Lab5/controllers/map.npy')
    proper_map = np.load('/Applications/Robotics/Lab/CSCI3302_Lab5/controllers/proper_map.npy')
    
    for point in path:
        proper_map[point[0]][point[1]] = 2
    # plt.imshow(proper_map, cmap = 'Reds')         
    # plt.show()          
       

state = 0 # use this to iterate through your path


current_waypoint = path[len(path)-2]
# print ("here the path: ", path)
path = path[:-2]
# print ("now the path is: ", path)

        
while robot.step(timestep) != -1 and mode != 'planner':

###################
#
# Sensing
#
###################
    # Ground truth pose
    pose_y = gps.getValues()[2]
    pose_x = gps.getValues()[0]

    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            rho = LIDAR_SENSOR_MAX_RANGE

        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        wx =  math.cos(pose_theta)*rx - math.sin(pose_theta)*ry + pose_x
        wy =  -(math.sin(pose_theta)*rx + math.cos(pose_theta)*ry) + pose_y

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))

        if rho < 0.5*LIDAR_SENSOR_MAX_RANGE:
# Part 1.3: visualize map gray values.

            # You will eventually REPLACE the following 2 lines with a more robust version of map
            # and gray drawing that has more levels than just 0 and 1.
            (pixel_x, pixel_y) = (int(wx*30), int(wy*30))
            if(pixel_x >= 0 and pixel_x < 360 and pixel_y >= 0 and pixel_y < 360):
                map[pixel_x][pixel_y] += 0.005
                if(map[pixel_x][pixel_y] >= 1.0):
                    map[pixel_x][pixel_y] = 1.0
                g = int(map[pixel_x][pixel_y]*255)
                g = int((g*256**2+g*256+g))
                display.setColor(g)
                display.drawPixel(360-int(wy*30),int(wx*30))

    display.setColor(int(0xFF0000))
    display.drawPixel(360-int(pose_y*30),int(pose_x*30))



###################
#
# Controller
#
###################
    
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
# Part 1.4: Save map to disc
            np.save('/Users/Owner/Documents/Robotics/CSCI3302_Lab5/controllers/map', map)
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    else: # not manual mode
        pass
# Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        
        # print ("current waypoint: ", current_waypoint)    
        # current_waypoint =np.array( [current_waypoint[0]/360, current_waypoint[1]/360])
        print ("posex, posey: ", pose_x, pose_y)
        cur_x = (float(current_waypoint[1])/360.0)*30.0
        cur_y = (float(current_waypoint[0])/360.0)*30.0
        print ("way_point x, way_point y:", cur_x, cur_y)
        # print ("new current waypoint: ", current_waypoint)
        dist_err = math.sqrt(math.pow(pose_x - cur_x, 2) + math.pow(pose_y - cur_y, 2))
        bearing_err = pose_theta + math.atan2(cur_y - pose_y, cur_x - pose_x)
        
        if(len(path) != 0 and dist_err < 0.2):
        
            current_waypoint = path[len(path)-1]
            cur_x = (float(current_waypoint[1])/360.0)*30.0
            cur_y = (float(current_waypoint[0])/360.0)*30.0 
            # print ("current waypoint: ", cur_x, cur_y)
            path = path[:-1]
        # else:
            # break
        print ("distance is:", dist_err)
        print ("bearing is:", bearing_err)
        
        
        
        
        #STEP 2: Controller
        x_gain = 1.5
        theta_gain = 3.1
        x_prime = dist_err * x_gain
        theta_prime = (theta_gain * bearing_err)
        
        

        #STEP 3: Compute wheelspeeds
        if(bearing_err > 0.25):
            vL = theta_prime + theta_gain
            vR = -theta_prime - theta_gain
        elif(bearing_err < -0.25):
            vL = -theta_prime - theta_gain
            vR = theta_prime + theta_gain
        elif (len(path) == 0 and dist_err < 0.2):
            vL = vR = 0
        else:
            vR = x_prime + x_gain
            vL = x_prime + x_gain
        

    # Normalize wheelspeed
    # Keep the max speed a bit less to minimize the jerk in motion
        if (vR > MAX_SPEED/2):
            vR = MAX_SPEED/2
        elif(vR < -MAX_SPEED/2):
            vR = -MAX_SPEED/2
        if (vL > MAX_SPEED/2):
            vL = MAX_SPEED/2
        elif(vL < -MAX_SPEED/2):
            vL = -MAX_SPEED/2

    # Odometry code. Don't change speeds after this
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta)) #/3.1415*180))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
