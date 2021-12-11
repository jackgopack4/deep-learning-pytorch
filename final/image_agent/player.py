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

    def check_defensive_box(self, location):
        #if defensive player in his box, return True
        if np.abs(location[0]) <= 10:
            if np.sign(location[1]) == np.sign(self.own_goal[1]) and np.abs(location[1]) > 45:
                return True
        return False
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
        steer_gain=2.25
        skid_thresh=0.8
        target_vel=25
        p0_brake = False
        p0_nitro = False
        p1_nitro = False
        p0_rescue = False
        puck_onscreen_threshold = 6.2
        
        print('global step:',self.global_step,'team',self.team)

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
        
        print('p0 kart loc',p0_kart_front,'p0 kart dir',p0_kart_direction,'p0 goal dir',p0_goal_direction)
        #print('yaw',yaw)

        p0_in_attack_quadrants = self.check_quadrant_vincent_16(p0_kart_front)
        p1_in_defense_quadrant = self.check_defensive_box(p1_kart_front)

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
        p0_steer = p0_steer_angle * steer_gain
        p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
                    
        print('p0 puck loc',p0_puck_loc,'p0 puck distance',p0_puck_dist,'p0 puck onscreen',p0_puck_onscreen)

        # make one player defensive
        # e.g. sit in front of own goal (determine using coordinates)
        # only act if puck gets close
        # hit it away then back up to desired location
        
        #print('puck_onscreen',p0_puck_onscreen,'loc',p0_puck_loc,'dist',p0_puck_dist)
        #print('kart location',p0_current_location)
        #print('kart velocity',p0_current_vel_magnitude)
        if p0_puck_onscreen<puck_onscreen_threshold:
            
            print('puck out of frame')
            # idea: check if it was previously in frame, then turn toward that direction?
            if np.sign(p0_kart_front[1]) == np.sign(p0_kart_direction[1]): #or np.sign(p0_kart_front[0]) == np.sign(p0_kart_direction[0]):
                print('kart facing direction of location, back up')
                p0_accel = 0
                p0_brake = True
                p0_steer = np.sign(p0_current_location[0])
            else:
                print('kart facing opposite of location, go forward')
                p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
                p0_brake = False
                p0_steer = -np.sign(p0_current_location[0])
            
        else:
            print('puck in frame')
            p0_steer = p0_steer_angle * steer_gain
            p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
            if p0_puck_dist < .1 and p0_puck_dist >= 0.04:
                # attempt to adjust slightly to get behind puck
                #p0_steer = 1.2 * p0_steer
                p0_accel = 0
                print('adding small amount to steer value close to puck')
                if np.sign(p0_goal_direction[1]) == np.sign(p0_kart_direction[1]):
                    #heading toward target goal
                    print('heading toward target goal?')
                    if np.abs(p0_goal_direction[0] - p0_kart_direction[0]) > 0.01:
                        # kart and goal x-value are not aligned yet
                        print('facing target, adding value',  (p0_goal_direction[0] - p0_kart_direction[0]))
                        p0_steer = p0_steer + (p0_goal_direction[0] - p0_kart_direction[0]) 
                else:
                    #heading toward own goal, go toward outside of puck instead
                    print('steering away from own goal adding',(0.3 * (np.sign(self.own_goal[1]) * np.sign(p0_kart_front[0]))))
                    p0_steer = p0_steer + 0.3 * (np.sign(self.own_goal[1]) * np.sign(p0_kart_front[0]))
                    p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
            elif (np.abs(p0_puck_loc[0])<0.1 and (np.abs(p0_puck_loc[1])+0.035)<0.1 and p0_puck_dist < 0.05):
                #The puck is on screen and in front of me?
                print('I think I have the puck')
                if np.sign(p0_goal_direction[1]) == np.sign(p0_kart_direction[1]):
                    #heading toward target goal
                    print('heading toward target goal?')
                    if np.abs(p0_goal_direction[0] - p0_kart_direction[0]) > 0.05:
                        # kart and goal x-value are not aligned yet
                        print('facing target, adding value', (p0_goal_direction[0] - p0_kart_direction[0]))
                        p0_steer = p0_steer + (p0_goal_direction[0] - p0_kart_direction[0]) 
                    else: 
                        print('facing goal, let\'s go')
                        p0_nitro = True
                else:
                    #heading toward own goal, go toward outside of puck instead
                    print('steering away from own goal adding',(0.25 * (np.sign(self.own_goal[1]) * np.sign(p0_kart_front[0]))))
                    p0_steer = p0_steer + 0.69 * (np.sign(self.own_goal[1]) * np.sign(p0_kart_front[0]))
                    p0_accel = 1.0 if p0_current_vel_magnitude < target_vel else 0.0
            else:
                if np.abs(p0_puck_loc[0]) > 0.02 and p0_puck_dist <= 0.02 and p0_puck_loc[1] > -0.075:
                    print('puck off to the side')
                    p0_accel = 0
                    p0_brake = True
                    p0_steer = - np.sign(p0_puck_loc[0])
            

        # what to do if kart is stuck (velocity ~0, location ~ same as prev)
        if self.global_step > 15 and ((np.abs(self.p0_prev_locations[self.global_step-1][0] - p0_current_location[0]) < 0.02 and np.abs(self.p0_prev_locations[self.global_step-1][1] - p0_current_location[1]) < 0.02) and (np.abs(self.p0_prev_vel_magnitudes[self.global_step-1]) < 0.5 and np.abs(p0_current_vel_magnitude)   < 0.5)) or np.abs(p0_kart_front[1]) > 64.0:
            print('i\'m stuck')
            # if up against wall and facing away from wall, move forward?
            if np.sign(p0_kart_front[1]) == np.sign(p0_kart_direction[1]):
                print('kart facing direction of location, back up')
                p0_accel = 0
                p0_brake = True
                p0_steer = np.sign(p0_current_location[0])
            else:
                print('kart facing opposite of location, go forward')
                p0_accel = 1
                p0_brake = False
                p0_steer = -np.sign(p0_current_location[0])
        '''
        #maybe let's avoid crashing into wall altogether
        # check if |x| > 38, |y| > 55
        # if so, compare x_loc to x_dir and y_loc to y_dir
        
        if (np.abs(p0_kart_front[0]) > 38.) or (np.abs(p0_kart_front[1]) > 55.):
            print('executing avoid crash function')
            p0_steer_addition = self.avoid_crash(p0_kart_front,p0_kart_direction)
            p0_steer = p0_steer_addition
            p0_accel = 1
        
        if np.abs(p0_kart_front[1]) > 64.0:
            print('stuck in goal')
            if np.sign(p0_kart_front[1]) == np.sign(p0_kart_direction[1]):
                print('kart facing direction of location, back up')
                p0_accel = 0
                p0_brake = True
                p0_steer = - np.sign(p0_current_location[0])
            else:
                print('kart facing opposite of location, go forward')
                p0_accel = 1
                p0_brake = False
                p0_steer = np.sign(p0_current_location[0])
        '''
        #clip steer value to [-1,1]
        p0_steer = np.clip(p0_steer,-1,1)
        # store current state values
        actions = [dict(acceleration=p0_accel, brake=p0_brake, steer=p0_steer, nitro=p0_nitro,rescue=p0_rescue),
                   dict(acceleration=0, steer=0)]
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
