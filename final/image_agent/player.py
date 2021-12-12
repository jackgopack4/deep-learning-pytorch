import pystk
import numpy as np
from . import planner
import torch
from os import path 
from torch import load
from torchvision.transforms import ToTensor


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
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
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        if self.team == 1:
            self.own_goal = [0., 64.]
            self.target_goal = [0., -64.]
        else: 
            self.own_goal = [0., -64.]
            self.target_goal = [0., 64.]
        self.p1_box_centerpoint[1] = self.own_goal[1] - np.sign(self.own_goal[1]) * 2
        #print('p1 box centerpoint',self.p1_box_centerpoint)
        players = 'tux'
        #if self.team == 0:
        #    players = 'sara_the_racer'
        return [players] * num_players

    def transform_images(self, image_0, image_1):
        i0 = self.transform(image_0)
        i1 = self.transform(image_1)
        return torch.stack([i0,i1],0)

    def calc_yaw(self, q):
        yaw = np.arctan2(2.0*(q[2]*q[3] + q[0]*q[1]), q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3])
        return np.degrees(yaw)

    def avoid_crash(self,location, direction):
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
    '''
    def check_quadrant_vincent_16(self, location):
        # based on location, determine if safe to drive straight to goal
        # if True, go straight to goal when I have puck
        if np.abs(location[0]) > 10:
            #outer sides
            if np.sign(location[1]) == np.sign(self.target_goal[1]):
                # in opponent's half
                if np.abs(location[1]) > 45:
                    #outer quadrant
                    return False
        return True
    '''

    def check_defensive_box(self, current_location):
        #if defensive player in his box, return True
        center_vector = ((torch.tensor(self.p1_box_centerpoint) - torch.tensor(current_location)) / torch.norm(torch.tensor(self.p1_box_centerpoint) - torch.tensor(current_location))).numpy()
        if np.abs(current_location[0]) <= 8:
            if np.sign(current_location[1]) == np.sign(self.own_goal[1]) and np.abs(current_location[1]) > 55:
                return True, center_vector
        return False, center_vector
    def calc_angle(self, direction, location):
        #calculate turn direction (positive = right, negative = left) from current
        #direction vector to a specified point (usually origin?)
        target = -location
        dir_angle = np.arctan2(direction[1],direction[0])
        tar_angle = np.arctan2(target[1],target[0])
        return -np.sign(tar_angle - dir_angle)

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
        puck_onscreen_threshold = 4.9
        
        #print('global step:',self.global_step,'team',self.team)

        p0_quaternion = player_state[0].get('kart').get('rotation')
        p1_quaternion = player_state[1].get('kart').get('rotation')
        #print('got quaternion')
        #p0_yaw = self.calc_yaw(p0_quaternion)
        #p1_yaw = self.calc_yaw(p1_quaternion)
        #print('yaw angle',yaw, 'degrees')

        p0_kart_front = torch.tensor(player_state[0].get('kart').get('front'), dtype=torch.float32)[[0, 2]]
        p1_kart_front = torch.tensor(player_state[1].get('kart').get('front'), dtype=torch.float32)[[0, 2]]
        p0_kart_center = torch.tensor(player_state[0].get('kart').get('location'), dtype=torch.float32)[[0, 2]]
        p1_kart_center = torch.tensor(player_state[1].get('kart').get('location'), dtype=torch.float32)[[0, 2]]
        p0_kart_direction = ((p0_kart_front-p0_kart_center) / torch.norm(p0_kart_front-p0_kart_center)).numpy()
        p1_kart_direction = ((p1_kart_front-p1_kart_center) / torch.norm(p1_kart_front-p1_kart_center)).numpy()
        p0_goal_direction = ((torch.tensor(self.target_goal) - p0_kart_front) / torch.norm(torch.tensor(self.target_goal) - p0_kart_front)).numpy()
        p1_goal_direction = ((torch.tensor(self.target_goal) - p1_kart_front) / torch.norm(torch.tensor(self.target_goal) - p1_kart_front)).numpy()
        p0_own_goal_direction = ((torch.tensor(self.own_goal) - p0_kart_front) / torch.norm(torch.tensor(self.own_goal) - p0_kart_front)).numpy()
        p1_own_goal_direction = ((torch.tensor(self.own_goal) - p1_kart_front) / torch.norm(torch.tensor(self.own_goal) - p1_kart_front)).numpy()
        p0_goal_distance = np.linalg.norm(self.target_goal-p0_kart_front.numpy()) 
        p1_goal_distance = np.linalg.norm(self.target_goal-p1_kart_front.numpy())
        
        #print('p0 kart loc',p0_kart_front,'p0 kart dir',p0_kart_direction,'p0 goal dir',p0_goal_direction)
        #print('yaw',yaw)

        #p1_in_defense_quadrant = self.check_defensive_box(p1_kart_front)

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

        
        # default controller values
        p0_steer_angle = steer_gain * p0_puck_loc[0]
        p0_steer_addition = 0.0
        p0_steer = p0_steer_angle * steer_gain + p0_steer_addition
        p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
        
        # idea: make one player defensive?
        # e.g. sit in front of own goal (determine using coordinates)
        # only act if puck gets close
        # hit it away then back up to desired location

        if p0_puck_onscreen<puck_onscreen_threshold or p0_puck_dist < -0.2:
            #print('puck out of frame')
            # idea: check if it was previously in frame, then turn toward that direction?
            
            p0_x_perpendicular = np.abs(p0_kart_direction[0]) > 0.85
            p0_y_perpendicular = np.abs(p0_kart_direction[1]) > 0.85            
            p0_on_x_edge = (np.sign(p0_kart_front[0]) == np.sign(p0_kart_direction[0]) and (np.abs(p0_kart_front[0]) > 38))
            p0_on_y_edge = (np.sign(p0_kart_front[1]) == np.sign(p0_kart_direction[1]) and (np.abs(p0_kart_front[1]) > 54))
            p0_stuck_x = p0_on_x_edge and p0_x_perpendicular
            p0_stuck_y = p0_on_y_edge and p0_y_perpendicular

            if (p0_stuck_x or p0_stuck_y):
                p0_accel = 0
                p0_brake = True
                p0_steer = 1
                #print('backing up')
            else:
                p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
                p0_brake = False
                #if self.global_step >=1: p0_steer = self.p0_prev_steers[self.global_step - 1]
                p0_steer = - 1
                #print('going forward')
        else:
            #print('puck in frame')
            if p0_puck_dist < .225 and p0_puck_dist >= -0.1 and np.abs(p0_puck_loc[0]) <0.2:
                #attempt to adjust slightly to get behind puck
                p0_accel = .8
                #print('p0 getting close to puck')
                if np.sign(p0_goal_direction[1]) == np.sign(p0_kart_direction[1]) and np.abs(p0_goal_direction[0] - p0_kart_direction[0]) < 0.5:
                    #print('p0 heading toward target goal')
                    p0_goal_alignment_value = np.sign(p0_kart_direction[1])*(p0_kart_direction[0] - p0_goal_direction[0])
                    if np.abs(p0_goal_alignment_value) > 0.02:
                        #print('kart and goal x-value are not aligned yet')
                        p0_steer_addition = 0.1 * p0_goal_alignment_value
                        #print('p0 facing target, adding value',  p0_steer_addition)
                    
                elif np.sign(p0_kart_front[1]) == np.sign(self.own_goal[1]):
                    #heading toward own goal, go toward outside of puck instead
                    p0_goal_alignment_value = np.sign(p0_kart_direction[1])*(p0_kart_direction[0] - p0_own_goal_direction[0])
                    p0_steer_addition = - 0.01 * p0_goal_alignment_value / np.abs(p0_puck_dist)
                    #print('p0 steering away from own goal adding',p0_goal_alignment_value)
                    p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
            else:
                if np.abs(p0_puck_loc[0]) > 0.02 and p0_puck_dist <= 0.2 and p0_puck_loc[1] > -0.075:
                    #print('p0 puck back to the side')
                    p0_accel = 0
                    p0_steer_addition = 0.0
                    p0_brake = True
                    p0_steer = -.25 *np.sign(p0_puck_loc[0])
                    #print('puck off to side, steering',p0_steer)
            if (np.abs(p0_puck_loc[0])<0.1 and (np.abs(p0_puck_loc[1]+0.05))<0.1 and p0_puck_dist < 0.1 and p0_puck_dist > -0.05):
                #print('p0 I think I have the puck')
                p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
                if np.sign(p0_goal_direction[1]) == np.sign(p0_kart_direction[1]) and np.abs(p0_goal_direction[0] - p0_kart_direction[0]) < 0.2:
                    #print('heading toward target goal')
                    p0_goal_alignment_value = np.sign(p0_kart_direction[1])*(p0_kart_direction[0] - p0_goal_direction[0])
                    if np.abs(p0_goal_alignment_value) > 0.05:
                        p0_accel = 1.0 if p0_current_vel_magnitude < 15 else 0.0
                        p0_steer_addition = -0.1 * p0_goal_alignment_value #/ np.abs(p0_puck_dist)
                        #print('p0 kart and goal x-value are not aligned yet, adding value', p0_steer_addition)
                    else: 
                        #print('p0 facing goal, let\'s go')
                        p0_nitro = True
                        p0_steer_addition = 0
                elif np.sign(p0_kart_front[1]) == np.sign(self.own_goal[1]):
                    #heading toward own goal, go toward outside of puck instead
                    p0_goal_alignment_value = np.sign(p0_kart_direction[1])*(p0_kart_direction[0] - p0_own_goal_direction[0])
                    p0_steer_addition = - 0.01 * p0_goal_alignment_value / np.abs(p0_puck_dist)
                    #print('p0 steering away from own goal adding',p0_steer_addition)
                    p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
            #print('p0 final steer',p0_steer,'p0 final addition',p0_steer_addition)
            p0_steer = p0_steer + p0_steer_addition
        
        #print('p0_direction',p0_kart_direction,'p0_goal_dir',p0_goal_direction,'p0_steer',p0_steer,'p0_goal_distance',p0_goal_distance)
        #print('p0 location',p0_kart_front,'p0 puck loc',p0_puck_loc,'p0 puck distance',p0_puck_dist,'p0 puck onscreen',p0_puck_onscreen)
        # what to do if kart is stuck (velocity ~0, location ~ same as prev)
        if self.global_step > 15 and ((np.abs(self.p0_prev_locations[self.global_step-1][0] - p0_current_location[0]) < 0.02 and np.abs(self.p0_prev_locations[self.global_step-1][1] - p0_current_location[1]) < 0.02) and (np.abs(self.p0_prev_vel_magnitudes[self.global_step-1]) < 0.5 and np.abs(p0_current_vel_magnitude)   < 0.5)) or np.abs(p0_kart_front[1]) > 64.0:
            #print('i\'m stuck')
            self.p0_prev_stucks.append(True)

            # if up against wall and facing away from wall, move forward?
            if np.sign(p0_kart_front[1]) == np.sign(p0_kart_direction[1] or np.sign(p0_kart_front[0]) == np.sign(p0_kart_direction[0])):
                #print('kart facing direction of location, back up')
                p0_accel = 0
                p0_brake = True
                p0_steer = np.sign(p0_current_location[0])
            else:
                #print('kart facing opposite of location, go forward')
                p0_accel = 1
                p0_brake = False
                p0_steer = -np.sign(p0_current_location[0])
            if self.global_step > 5:
                p0_rescue = False
                for i in range(0,5):
                    if self.p0_prev_stucks[i]: p0_rescue = True
                #if(p0_rescue): print('p0 rescuing at location',p0_kart_front)
        else: self.p0_prev_stucks.append(False)

        #clip steer value to [-1,1]
        p0_steer = np.clip(p0_steer,-1,1)
        self.p0_prev_steers.append(p0_steer)
        # store current state values

        # default controller values
        p1_steer_angle = steer_gain * p1_puck_loc[0]*2
        p1_steer_addition = 0.0
        p1_steer = p1_steer_angle * steer_gain + p1_steer_addition
        p1_accel = 1.0 if p1_current_vel_magnitude < target_vel else 0.0

        #print('p1 kart loc',p1_kart_front,'p1 kart dir',p1_kart_direction,'p1 goal dir',p1_goal_direction)
        #print('p1 puck loc',p1_puck_loc,'p1 puck distance',p1_puck_dist,'p1 puck onscreen',p1_puck_onscreen)
        
        # idea: make one player defensive?
        # e.g. sit in front of own goal (determine using coordinates)
        # only act if puck gets close
        # hit it away then back up to desired location

        if p1_puck_onscreen<puck_onscreen_threshold or p1_puck_dist < -0.2:
            #print('puck out of frame')
            # idea: check if it was previously in frame, then turn toward that direction?
            
            p1_x_perpendicular = np.abs(p1_kart_direction[0]) > 0.85
            p1_y_perpendicular = np.abs(p1_kart_direction[1]) > 0.85            
            p1_on_x_edge = (np.sign(p1_kart_front[0]) == np.sign(p1_kart_direction[0]) and (np.abs(p1_kart_front[0]) > 38))
            p1_on_y_edge = (np.sign(p1_kart_front[1]) == np.sign(p1_kart_direction[1]) and (np.abs(p1_kart_front[1]) > 54))
            p1_stuck_x = p1_on_x_edge and p1_x_perpendicular
            p1_stuck_y = p1_on_y_edge and p1_y_perpendicular

            if (p1_stuck_x or p1_stuck_y):
                p1_accel = 0
                p1_brake = True
                p1_steer = 1
                #print('backing up')
            else:
                p1_accel = 1.0 if p1_current_vel_magnitude < target_vel else 0.0
                p1_brake = False
                #if self.global_step >=1: p1_steer = self.p1_prev_steers[self.global_step - 1]
                p1_steer = - 1
                #print('going forward')
        else:
            #print('puck in frame')
            if p1_puck_dist < .225 and p1_puck_dist >= -0.1 and np.abs(p1_puck_loc[0]) <0.2:
                #attempt to adjust slightly to get behind puck
                p1_accel = .8
                #print('p1 getting close to puck')
                if np.sign(p1_goal_direction[1]) == np.sign(p1_kart_direction[1]) and np.abs(p1_goal_direction[0] - p1_kart_direction[0]) < 0.5:
                    #print('p1 heading toward target goal')
                    p1_goal_alignment_value = np.sign(p1_kart_direction[1])*(p1_kart_direction[0] - p1_goal_direction[0])
                    if np.abs(p1_goal_alignment_value) > 0.02:
                        #print('kart and goal x-value are not aligned yet')
                        p1_steer_addition = 0.1 * p1_goal_alignment_value
                        #print('p1 facing target, adding value',  p1_steer_addition)
                    
                elif np.sign(p1_kart_front[1]) == np.sign(self.own_goal[1]):
                    #heading toward own goal, go toward outside of puck instead
                    p1_goal_alignment_value = np.sign(p1_kart_direction[1])*(p1_kart_direction[0] - p1_own_goal_direction[0])
                    p1_steer_addition = - 0.01 * p1_goal_alignment_value / np.abs(p1_puck_dist)
                    #print('p1 steering away from own goal adding',p1_goal_alignment_value)
                    p1_accel = 1.0 if p1_current_vel_magnitude < target_vel else 0.0
            else:
                if np.abs(p1_puck_loc[0]) > 0.02 and p1_puck_dist <= 0.2 and p1_puck_loc[1] > -0.075:
                    #print('p1 puck back to the side')
                    p1_accel = 0
                    p1_steer_addition = 0.0
                    p1_brake = True
                    p1_steer = -.25 *np.sign(p1_puck_loc[0])
                    #print('puck off to side, steering',p1_steer)
            if (np.abs(p1_puck_loc[0])<0.1 and (np.abs(p1_puck_loc[1]+0.05))<0.1 and p1_puck_dist < 0.1 and p1_puck_dist > -0.05):
                #print('p1 I think I have the puck')
                p1_accel = 1.0 if p1_current_vel_magnitude < target_vel else 0.0
                if np.sign(p1_goal_direction[1]) == np.sign(p1_kart_direction[1]) and np.abs(p1_goal_direction[0] - p1_kart_direction[0]) < 0.2:
                    #print('heading toward target goal')
                    p1_goal_alignment_value = np.sign(p1_kart_direction[1])*(p1_kart_direction[0] - p1_goal_direction[0])
                    if np.abs(p1_goal_alignment_value) > 0.05:
                        p1_accel = 1.0 if p1_current_vel_magnitude < 15 else 0.0
                        p1_steer_addition = -0.1 * p1_goal_alignment_value #/ np.abs(p1_puck_dist)
                        #print('p1 kart and goal x-value are not aligned yet, adding value', p1_steer_addition)
                    else: 
                        #print('p1 facing goal, let\'s go')
                        p1_nitro = True
                        p1_steer_addition = 0
                elif np.sign(p1_kart_front[1]) == np.sign(self.own_goal[1]):
                    #heading toward own goal, go toward outside of puck instead
                    p1_goal_alignment_value = np.sign(p1_kart_direction[1])*(p1_kart_direction[0] - p1_own_goal_direction[0])
                    p1_steer_addition = - 0.01 * p1_goal_alignment_value / np.abs(p1_puck_dist)
                    #print('p1 steering away from own goal adding',p1_steer_addition)
                    p1_accel = 1.0 if p1_current_vel_magnitude < target_vel else 0.0
            #print('p1 final steer',p1_steer,'p1 final addition',p1_steer_addition)
            p1_steer = p1_steer + p1_steer_addition
        
        #print('p1_direction',p1_kart_direction,'p1_goal_dir',p1_goal_direction,'p1_steer',p1_steer,'p1_goal_distance',p1_goal_distance)
        #print('p1 location',p1_kart_front,'p1 puck loc',p1_puck_loc,'p1 puck distance',p1_puck_dist,'p1 puck onscreen',p1_puck_onscreen)
        # what to do if kart is stuck (velocity ~0, location ~ same as prev)
        if self.global_step > 15 and ((np.abs(self.p1_prev_locations[self.global_step-1][0] - p1_current_location[0]) < 0.02 and np.abs(self.p1_prev_locations[self.global_step-1][1] - p1_current_location[1]) < 0.02) and (np.abs(self.p1_prev_vel_magnitudes[self.global_step-1]) < 0.5 and np.abs(p1_current_vel_magnitude)   < 0.5)) or np.abs(p1_kart_front[1]) > 64.0:
            #print('i\'m stuck')
            self.p1_prev_stucks.append(True)

            # if up against wall and facing away from wall, move forward?
            if np.sign(p1_kart_front[1]) == np.sign(p1_kart_direction[1] or np.sign(p1_kart_front[0]) == np.sign(p1_kart_direction[0])):
                #print('kart facing direction of location, back up')
                p1_accel = 0
                p1_brake = True
                p1_steer = np.sign(p1_current_location[0])
            else:
                #print('kart facing opposite of location, go forward')
                p1_accel = 1
                p1_brake = False
                p1_steer = -np.sign(p1_current_location[0])
            if self.global_step > 5:
                p1_rescue = False
                for i in range(0,5):
                    if self.p1_prev_stucks[i]: p1_rescue = True
                #if(p1_rescue): print('p1 rescuing at location',p1_kart_front)
        else: self.p1_prev_stucks.append(False)

        #clip steer value to [-1,1]
        p1_steer = np.clip(p1_steer,-1,1)
        self.p1_prev_steers.append(p1_steer)


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
        
        #print('passing action p0',actions[0])
        self.global_step+=1
        #print('actions to output',actions)
        return actions
