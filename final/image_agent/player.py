import pystk
import numpy as np
from . import planner
import torch
from os import path 
from torch import load
from torchvision.transforms import ToTensor
import random

def seek_point(location,direction,destination,forward=None):
    # take in current location and destination and normed direction vector
    # return steer, accel, brake
    # optional: set forward or backward. Default will choose automatically
    new_vector = torch.tensor(destination-location)
    new_vector_magnitude = torch.norm(new_vector).numpy()
    if new_vector_magnitude < 4.0: 
        # I think I'm already close, inject some randomness
        return np.clip((np.random.rand()-0.5)*10,-1,1), np.clip(np.random.rand()*5,0,1), bool(random.getrandbits(1))
    new_vector = new_vector.numpy()
    new_vector_normed = new_vector/new_vector_magnitude
    if forward==None:
        if np.sign(new_vector[1]) == np.sign(direction[1]): forward = True
        else: forward = False
    if forward==True:
        x_diff = new_vector_normed[0] - direction[0]
        y_diff = new_vector_normed[1] - direction[1]
        accel = 1.0
        brake = False
    elif forward==False:
        x_diff = new_vector_normed[0] + direction[0]
        y_diff = new_vector_normed[1] + direction[1]
        accel = 0.0
        brake = True
    steer = np.sign(direction[1]) * x_diff
    return steer, accel, brake
class Team:
    agent_type = 'image'

    def __init__(self):
        self.team = None
        self.num_players = None
        self.transform = ToTensor()
        r = planner.Planner()
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
        self.model = r
        self.p0_prev_actions = []
        self.p1_prev_actions = []
        self.p0_prev_locations = []
        self.p1_prev_locations = []
        self.p0_prev_vel_magnitudes = []
        self.p1_prev_vel_magnitudes = []
        self.p0_prev_vels = []
        self.p1_prev_vels = []
        self.p0_prev_puck_onscreens = []
        self.p1_prev_puck_onscreens = []
        self.p0_prev_puck_locs = []
        self.p0_prev_puck_locs = []
        self.p0_prev_stucks = []
        self.p1_prev_stucks = []
        self.p0_prev_steers = []
        self.p1_prev_steers = []
        self.p1_box_centerpoint = [0.0,0.0]
        self.p0_corner_point = [-15.0, 0.0]
        self.corner_points = [[-15.0, 0.0],[15.0,0.0]]
        self.corner_point_counter = 0
        self.own_goal = None
        self.target_goal = None
        self.goal_width = 10.
        self.global_step = 0
        
    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        self.team, self.num_players = team, num_players
        if self.team == 1:
            self.own_goal = [0., 64.]
            self.target_goal = [0., -64.]
        else: 
            self.own_goal = [0., -64.]
            self.target_goal = [0., 64.]
        self.p1_box_centerpoint[1] = self.own_goal[1] - np.sign(self.own_goal[1])
        self.p0_corner_point[1] = self.p1_box_centerpoint[1] - np.sign(self.p1_box_centerpoint[1])
        for i in range(0,len(self.corner_points)):
            self.corner_points[i][1] = self.p0_corner_point[1]
            if self.p0_corner_point == self.corner_points[i]:
                self.corner_point_counter = i
        players = 'tux'
        #if self.team == 0:
        #    players = 'sara_the_racer'
        return [players] * num_players

    def transform_images(self, image_0, image_1):
        i0 = self.transform(image_0)
        i1 = self.transform(image_1)
        return torch.stack([i0,i1],0)

    def calc_yaw(self, q):
        #take in quaternion, return yaw angle in [-1 to 1]
        yaw = np.arctan2(2.0*(q[2]*q[3] + q[0]*q[1]), q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3])
        return yaw / np.pi

    def angle_between_vectors(self,current,desired):
        a = np.array(current)
        b = np.array(desired)
        ang1 = np.arctan2(*b[::-1])
        ang2 = np.arctan2(*a[::-1])
        res = - (ang1 - ang2) % (2 * np.pi)
        res = res / np.pi
        if res < -1:
            return res + 2
        elif res > 1:
            return res - 2
        else: return res
        
    def avoid_crash(self,location, direction):
        # honestly not really sure why I wrote this one
        xloc = location[0]
        yloc = location[1]
        xdir = direction[0]
        ydir = direction[1]
        if (np.sign(xloc) != np.sign(xdir)) and (np.sign(yloc) == np.sign(ydir)):
            return np.sign(xdir * yloc)
        if (np.sign(xloc) == np.sign(xdir)) and (np.sign(yloc) == np.sign(ydir)):
            return -np.sign(xdir * yloc)
        if (np.sign(xloc) == np.sign(xdir)) and (np.sign(yloc) != np.sign(ydir)):
            return np.sign(xdir * yloc)
        else:
            return 0.0

    def check_defensive_box(self, current_location):
        #if defensive player in his box, return True
        center_vector = ((torch.tensor(self.p1_box_centerpoint) - torch.tensor(current_location)) / torch.norm(torch.tensor(self.p1_box_centerpoint) - torch.tensor(current_location))).numpy()
        if np.abs(current_location[0]) <= 8:
            if np.sign(current_location[1]) == np.sign(self.own_goal[1]) and np.abs(current_location[1]) > 55:
                return True, center_vector
        return False, center_vector
    def check_offensive_box(self, current_location,current_direction):
        # take in current loc and direction
        # called when targeting puck
        # if perpendicular in goal box, attempt to steer outside and bank in
        # return steer addition sign
        #print('sign x-value',np.sign(current_location[1]),'sign target',np.sign(self.target_goal[1]))
        if np.sign(current_location[1]) == np.sign(self.target_goal[1]) and np.abs(current_location[1]) > 45:
            #print('in offensive box, x-dir',current_direction[0])
            if np.abs(current_direction[0]) > 0.7:
                return np.sign(current_location[1])*np.sign(current_direction[0])
        return 0.0
    '''
    def calc_angle(self, direction, location):
        #calculate turn direction (positive = right, negative = left) from current
        #direction vector to a specified point (usually origin?)
        target = -location
        dir_angle = np.arctan2(direction[1],direction[0])
        tar_angle = np.arctan2(target[1],target[0])
        return -np.sign(tar_angle - dir_angle)
    '''
    def check_for_wall(self,kart_loc, kart_dir):
        # check if close to wall
        # return booleans as well as steering addition to get into target goal
        on_x_edge = np.abs(kart_loc[0]) > 18
        on_y_edge = np.abs(kart_loc[1]) > 54
        in_corner = np.abs(kart_loc[0]) > 15 and np.abs(kart_loc[1]) > 50
        if (on_x_edge and np.sign(kart_dir[0]) == np.sign(kart_loc[0])) or (on_y_edge and np.sign(kart_dir[1]) == np.sign(kart_loc[1])) or (in_corner and np.sign(kart_dir[0]) == np.sign(kart_loc[0]) and np.sign(kart_dir[1]) == np.sign(kart_loc[1])):
            puck_steer_add = np.sign(self.target_goal[1]) / np.cbrt(kart_loc[0].detach().numpy())
        else: puck_steer_add = 0.0
        return on_x_edge, on_y_edge, in_corner, puck_steer_add

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # set constants for calculations
        steer_gain=2.5
        skid_thresh=0.8
        target_vel=30
        p0_brake = False
        p1_brake = False
        p0_nitro = False
        p1_nitro = False
        p0_rescue = False
        p1_rescue = False
        p0_puck_onscreen_threshold = 3
        p1_puck_onscreen_threshold = 3
        
        #print('global step:',self.global_step,'team',self.team)
        if self.global_step % 300 == 0:
            if self.corner_point_counter == 1:
                self.corner_point_counter = 0
            else:
                self.corner_point_counter = 1
            self.p0_corner_point = self.corner_points[self.corner_point_counter]

        p0_quaternion = player_state[0].get('kart').get('rotation')
        p1_quaternion = player_state[1].get('kart').get('rotation')

        p0_kart_front = torch.tensor(player_state[0].get('kart').get('front'), dtype=torch.float32)[[0, 2]]
        p1_kart_front = torch.tensor(player_state[1].get('kart').get('front'), dtype=torch.float32)[[0, 2]]
        p0_kart_center = torch.tensor(player_state[0].get('kart').get('location'), dtype=torch.float32)[[0, 2]]
        p1_kart_center = torch.tensor(player_state[1].get('kart').get('location'), dtype=torch.float32)[[0, 2]]
        p0_kart_direction = ((p0_kart_front-p0_kart_center) / torch.norm(p0_kart_front-p0_kart_center)).numpy()
        p1_kart_direction = ((p1_kart_front-p1_kart_center) / torch.norm(p1_kart_front-p1_kart_center)).numpy()
        p0_kart_angle = np.arctan2(p0_kart_direction[1], p0_kart_direction[0])
        p1_kart_angle = np.arctan2(p1_kart_direction[1], p1_kart_direction[0])
        p0_kart_to_goal_angle = (torch.tensor(self.target_goal) - p0_kart_front) / torch.norm(torch.tensor(self.target_goal) - p0_kart_front).numpy()
        p1_kart_to_goal_angle = (torch.tensor(self.target_goal) - p1_kart_front) / torch.norm(torch.tensor(self.target_goal) - p1_kart_front).numpy()
        p0_goal_direction = ((torch.tensor(self.target_goal) - p0_kart_front) / torch.norm(torch.tensor(self.target_goal) - p0_kart_front)).numpy()
        p1_goal_direction = ((torch.tensor(self.target_goal) - p1_kart_front) / torch.norm(torch.tensor(self.target_goal) - p1_kart_front)).numpy()
        p0_own_goal_direction = ((torch.tensor(self.own_goal) - p0_kart_front) / torch.norm(torch.tensor(self.own_goal) - p0_kart_front)).numpy()
        p1_own_goal_direction = ((torch.tensor(self.own_goal) - p1_kart_front) / torch.norm(torch.tensor(self.own_goal) - p1_kart_front)).numpy()
        p0_goal_distance = np.linalg.norm(self.target_goal-p0_kart_front.numpy()) 
        p1_goal_distance = np.linalg.norm(self.target_goal-p1_kart_front.numpy())
        p0_goal_angle = self.angle_between_vectors(p0_kart_direction,p0_goal_direction)
        p1_goal_angle = self.angle_between_vectors(p1_kart_direction,p1_goal_direction)

        # transform images to tensor for input to model
        img_stack = self.transform_images(player_image[0],player_image[1])
        pred_puck, loc = self.model(img_stack)

        # get current values for puck, dist, location, velocity
        p0_puck_onscreen = pred_puck[0,0].detach().numpy()
        p1_puck_onscreen = pred_puck[1,0].detach().numpy()

        p0_puck_dist = pred_puck[0,1].detach().numpy()
        p1_puck_dist = pred_puck[1,1].detach().numpy()

        p0_puck_loc = loc[0,:].detach().numpy()
        p1_puck_loc = loc[1,:].detach().numpy()

        p0_current_vel = player_state[0].get('kart').get('velocity')
        p1_current_vel = player_state[1].get('kart').get('velocity')
        p0_current_vel_magnitude = np.linalg.norm([p0_current_vel[0],p0_current_vel[2]])
        p1_current_vel_magnitude = np.linalg.norm([p1_current_vel[0],p1_current_vel[2]])

        p0_current_location = player_state[0].get('kart').get('front')
        p1_current_location = player_state[1].get('kart').get('front')

        # check for "edge" cases
        p0_on_x_edge, p0_on_y_edge, p0_in_corner, p0_puck_wall_steer = self.check_for_wall(p0_kart_front,p0_kart_direction)
        p0_x_perpendicular = np.abs(p0_kart_direction[0]) > 0.85
        p0_y_perpendicular = np.abs(p0_kart_direction[1]) > 0.85 
        p0_diagonal = np.abs(p0_kart_direction[0]) > 0.65 and np.abs(p0_kart_direction[1]) > 0.65           
            
        p0_stuck_corner = p0_x_perpendicular and p0_in_corner
        p0_stuck_x = p0_on_x_edge and p0_x_perpendicular
        p0_stuck_y = p0_on_y_edge and p0_y_perpendicular
        
        p0_stuck_x_forward = np.sign(p0_kart_front[0]) == np.sign(p0_kart_direction[0])
        p0_stuck_y_forward = np.sign(p0_kart_front[1]) == np.sign(p0_kart_direction[1])
        p0_stuck_diag_forward = p0_stuck_x_forward and p0_stuck_y_forward
                    
        # default controller values
        p0_steer_angle = steer_gain * p0_puck_loc[0]
        p0_steer_addition = 0.0
        p0_steer = p0_steer_angle * steer_gain + p0_steer_addition
        p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0


        
        if (p0_puck_onscreen<p0_puck_onscreen_threshold or p0_puck_dist < -0.2) and self.global_step>0:
            #print('p0 puck out of frame')
            # idea: check if it was previously in frame, then turn toward that direction?
            if ((p0_stuck_x and p0_stuck_x_forward) or (p0_stuck_y and p0_stuck_y_forward) or (p0_stuck_corner and p0_stuck_diag_forward)):
                p0_accel = 0
                p0_brake = True
                p0_steer = 0.1 * np.random.rand()
                #print('p0 backing straight up')
            elif(p0_stuck_x or p0_stuck_y or p0_stuck_corner):
                p0_accel = 1
                p0_brake = False
                p0_steer = 0.5 * np.random.rand()
            else:
                p0_steer, p0_accel, p0_brake = seek_point(p0_kart_front.numpy(),p0_kart_direction,self.p0_corner_point,False)
                #print('p0 steering backward to corner')
                '''
                #print('backing zigzag up')
                p0_accel = 0
                p0_brake = True
                #if self.global_step >=1: p0_steer = self.p0_prev_steers[self.global_step - 1]
                p0_steer = -1 * self.p0_prev_steers[self.global_step-1]
                '''
        else:
            #print('p0 puck in frame')
            if p0_puck_dist < .225 and p0_puck_dist >= -0.1 and np.abs(p0_puck_loc[0]) <0.5:
                #attempt to adjust slightly to get behind puck
                #p0_accel = .8
                p0_nitro = True
                #print('p0 getting close to puck, goal angle',p0_goal_angle)
                if np.sign(p0_goal_direction[1]) == np.sign(p0_kart_direction[1]) and np.abs(p0_goal_angle) < 0.5:
                    #print('p0 heading toward target goal angle',p0_goal_angle)
                    if np.abs(p0_goal_angle) > 0.01:
                        #print('p0 kart and goal angle are not aligned yet')
                        p0_offensive_box = self.check_offensive_box(p0_kart_front,p0_kart_direction)
                        if p0_offensive_box != 0.0:
                            #print('p0 offensive box steer',p0_offensive_box)
                            #print('p0 adding extra steer in offensive box to bank in')
                            p0_steer_addition = - 1.5 * p0_offensive_box
                        elif p0_on_x_edge or p0_on_y_edge or p0_in_corner:
                            p0_steer_addition =  p0_puck_wall_steer
                            #print('p0 I think puck near wall, add higher steer',p0_steer_addition)
                        else: 
                            p0_steer_addition = -20*np.square(p0_puck_dist)*np.sqrt(p0_goal_distance)*p0_goal_angle
                            #print('p0 facing target, adding goal_angle value',  p0_steer_addition)
                    
                elif np.sign(p0_kart_front[1]) == np.sign(self.own_goal[1]) and np.sign(p0_kart_front[1]) == np.sign(p0_kart_direction[1])and np.abs(p0_kart_front[1]) > 15:
                    #heading toward own goal, go toward outside of puck instead
                    p0_goal_alignment_value = np.sign(p0_kart_direction[1])*(p0_kart_direction[0] - p0_own_goal_direction[0])
                    p0_steer_addition = - 50 * p0_goal_alignment_value / np.abs(p0_puck_dist)
                    #print('p0 steering away from own goal adding',p0_steer_addition)
                    p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
            else:
                if np.abs(p0_puck_loc[0]) > 0.1 and p0_puck_dist <= 0.2 and p0_puck_loc[1] > -0.1:
                    p0_accel = 0
                    p0_steer_addition = 0.0
                    p0_brake = True
                    p0_steer = -np.sign(p0_puck_loc[0])
                    #print('p0 puck off to side, steering',p0_steer)
            if (np.abs(p0_puck_loc[0])<0.15 and (np.abs(p0_puck_loc[1]))<0.075 and np.abs(p0_puck_dist) < 0.15):
                #print('p0 I think I have the puck')
                p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
                if np.sign(p0_goal_direction[1]) == np.sign(p0_kart_direction[1]) and np.abs(p0_goal_angle) < 0.2:
                    #print('p0 heading toward target goal')
                    #p0_goal_alignment_value = np.sign(p0_kart_direction[1])*(p0_kart_direction[0] - p0_goal_direction[0])
                    
                    if np.abs(p0_goal_angle) > 0.05:
                        p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
                        #p0_steer_addition = -0.25 * p0_goal_alignment_value #/ np.abs(p0_puck_dist)
                        p0_steer_addition = 5*np.square(p0_puck_dist)*np.sqrt(p0_goal_distance)*p0_goal_angle
                        #print('p0 kart and goal x-value are not aligned yet, adding value', p0_steer_addition)
                    else: 
                        #print('p0 facing goal, let\'s go')
                        p0_steer = p0_goal_angle
                        p0_nitro = True
                        p0_steer_addition = 0
                elif np.sign(p0_kart_front[1]) == np.sign(self.own_goal[1]) and np.sign(p0_kart_front[1]) == np.sign(p0_kart_direction[1]) and np.abs(p0_kart_front[1]) > 15:
                    #heading toward own goal, go toward outside of puck instead
                    p0_goal_alignment_value = np.sign(p0_kart_direction[1])*(p0_kart_direction[0] - p0_own_goal_direction[0])
                    p0_steer_addition = - 2 * p0_goal_alignment_value / np.abs(p0_puck_dist)
                    #print('p0 steering away from own goal adding',p0_steer_addition)
                    p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
            #print('p0 final steer',p0_steer,'p0 final addition',p0_steer_addition)
            p0_steer = p0_steer + p0_steer_addition
        
        #print('p0_direction',p0_kart_direction,'p0_goal_dir',p0_goal_direction)
        #print('p0 location',p0_kart_front.detach().numpy(),'p0_steer',p0_steer,'p0_goal_distance',p0_goal_distance)
        #print('p0 puck loc',p0_puck_loc,'p0 puck distance',p0_puck_dist,'p0 puck onscreen',p0_puck_onscreen)
        # what to do if kart is stuck (velocity ~0, location ~ same as prev)
        if self.global_step > 15 and ((np.abs(self.p0_prev_locations[self.global_step-1][0] - p0_current_location[0]) < 0.05 and np.abs(self.p0_prev_locations[self.global_step-1][1] - p0_current_location[1]) < 0.05) and (np.abs(self.p0_prev_vel_magnitudes[self.global_step-1]) < .5 and np.abs(p0_current_vel_magnitude)   < .5)) or np.abs(p0_kart_front[1]) > 69.0:
            #print('i\'m stuck')
            self.p0_prev_stucks.append(True)
            if ((p0_stuck_x and np.sign(p0_kart_front[0]) == np.sign(p0_kart_direction[0])) or (p0_stuck_y and np.sign(p0_kart_front[1]) == np.sign(p0_kart_direction[1])) or p0_stuck_corner):
                p0_accel = 0
                p0_brake = True
                p0_steer = 0 
                #print('p0 backing straight up')
            else:
                p0_steer, p0_accel, p0_brake = seek_point(p0_kart_front.numpy(),p0_kart_direction,self.p0_corner_point)
                #print('p0 steering to corner')
            '''
            if self.global_step > 5:
                p0_rescue = True
                for i in range(0,5):
                    if not self.p0_prev_stucks[self.global_step-i]: p0_rescue = False
                if(p0_rescue): #print('p0 rescuing at location',p0_kart_front)
            '''
        else: self.p0_prev_stucks.append(False)

        #clip steer value to [-1,1]
        p0_steer = np.clip(p0_steer,-1,1)
        self.p0_prev_steers.append(p0_steer)
        # store current state values

        # default controller values
        p1_steer_angle = steer_gain * p1_puck_loc[0]*1.5
        p1_steer_addition = 0.0
        p1_steer = p1_steer_angle * steer_gain + p1_steer_addition
        p1_accel = 1.0 if p1_current_vel_magnitude < target_vel else 0.0

        #print('p1 kart loc',p1_kart_front,'p1 kart dir',p1_kart_direction,'p1 goal dir',p1_goal_direction)
        #print('p1 puck loc',p1_puck_loc,'p1 puck distance',p1_puck_dist,'p1 puck onscreen',p1_puck_onscreen)
        
        # idea: make one player defensive?
        # e.g. sit in front of own goal (determine using coordinates)
        # only act if puck gets close
        # hit it away then back up to desired location
        p1_in_box, p1_center_vector = self.check_defensive_box(p1_kart_front)
        if p1_puck_onscreen < p1_puck_onscreen_threshold:
            #print('p1 puck not onscreen')
            #if not p1_in_box:
            p1_steer, p1_accel, p1_brake = seek_point(p1_kart_front.numpy(),p1_kart_direction,self.p1_box_centerpoint,forward=False)
            #else: #rotate toward target goal
            #    p1_accel = 0.25
            #    p1_brake = False
            #    p1_steer = np.sign(p1_kart_front[1])*(p1_goal_direction[0]-p1_kart_direction[0])
        else: # I think puck onscreen p1
            #print('p1_goal_angle',p1_goal_angle)
            if p1_puck_dist < .225 and p1_puck_dist >= -0.1 and np.abs(p1_puck_loc[0]) <0.5:
                p1_steer_addition = - 50 * np.square(p1_puck_dist)*np.sqrt(p1_goal_distance)*p1_goal_angle
                #print('p1 facing target, adding goal_angle value', p1_steer_addition)
                if (np.abs(p1_puck_loc[0])<0.15 and (np.abs(p1_puck_loc[1]))<0.075 and np.abs(p0_puck_dist) < 0.15):
                    if np.sign(p1_goal_direction[1]) == np.sign(p1_kart_direction[1]):
                        #print('p1 I think I have the puck')
                        if np.abs(p1_goal_angle) > 0.05:
                            p1_steer_addition = 5*np.square(p1_puck_dist)*np.sqrt(p1_goal_distance)*p1_goal_angle
                            #print('p1 kart and goal x-value are not aligned yet, adding value', p1_steer_addition)
                        else: 
                            #print('p1 facing goal, let\'s go')
                            p1_nitro = True
                            p1_steer = p1_goal_angle
                            p1_steer_addition = 0
                p1_offensive_box = self.check_offensive_box(p1_kart_front,p1_kart_direction)
                if p1_offensive_box != 0.0:
                    p1_steer_addition = -0.5 * p1_goal_angle
                    #print('in offensive box, adding',p1_steer_addition)
            elif np.abs(p1_puck_loc[0]) > 0.1 and p1_puck_dist <= 0.2 and p1_puck_loc[1] > -0.15:
                p1_accel = 0
                p1_steer_addition = 0.0
                p1_brake = True
                p1_steer = -np.sign(p1_puck_loc[0])
                #print('p1 puck off to side, steering',p1_steer)
            elif np.sign(p1_kart_front[1]) == np.sign(self.own_goal[1]) and np.sign(p1_kart_front[1]) == np.sign(p1_kart_direction[1])and np.abs(p1_kart_front[1]) > 15:
                #heading toward own goal, go toward outside of puck instead
                p1_goal_alignment_value = np.sign(p1_kart_direction[1])*(p1_kart_direction[0] - p1_own_goal_direction[0])
                p1_steer_addition = - 2 * p1_goal_alignment_value / np.abs(p1_puck_dist)
                #print('p1 steering away from own goal adding',p1_steer_addition)
                p1_accel = 1.0 if p1_current_vel_magnitude < target_vel else 0.0
        #print('p1_steer',p1_steer,'p1_addition',p1_steer_addition)
        #clip steer value to [-1,1]
        p1_steer = p1_steer + p1_steer_addition
        p1_steer = np.clip(p1_steer,-1,1)
        self.p1_prev_steers.append(p1_steer)
        '''
        p0_accel = 0.0
        p0_brake = False
        p0_steer = torch.tensor(0.0)
        p0_nitro = False
        '''
        actions = [dict(acceleration=p0_accel, brake=p0_brake, steer=p0_steer, nitro=p0_nitro,rescue=p0_rescue),
                   dict(acceleration=p1_accel, brake=p1_brake, steer=p1_steer, nitro=p1_nitro,rescue=p1_rescue)]
        # update memory values
        self.p0_prev_locations.append(p0_current_location)
        self.p1_prev_locations.append(p1_current_location)
        self.p0_prev_vel_magnitudes.append(p0_current_vel_magnitude)
        self.p1_prev_vel_magnitudes.append(p1_current_vel_magnitude)
        self.p0_prev_vels.append(p0_current_vel)
        self.p0_prev_actions.append(actions[0])
        self.p1_prev_actions.append(actions[1])
        self.p0_prev_vels.append(p0_current_vel)
        self.p1_prev_vels.append(p1_current_vel)
        self.p0_prev_puck_onscreens.append(p0_puck_onscreen)
        self.p1_prev_puck_onscreens.append(p1_puck_onscreen)
        self.p0_prev_puck_locs.append(p0_puck_loc)
        self.p0_prev_puck_locs.append(p1_puck_loc)
        
        self.global_step+=1
        return actions
