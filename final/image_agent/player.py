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
        self.prev_location = None
        self.prev_velocity = None
        self.own_goal = None
        self.target_goal = None
        self.goal_width = 10.
        
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
        if self.team == 0:
            self.own_goal = [0., 64.]
            self.target_goal = [0., -64.]
        else: 
            self.own_goal = [0., -64.]
            self.target_goal = [0., 64]
        return ['tux'] * num_players

    def transform_images(self, image_0, image_1):
        i0 = self.transform(image_0)
        i1 = self.transform(image_1)
        return torch.stack([i0,i1],0)

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
        '''
        img0 = self.transform(player_image[0])
        img1 = self.transform(player_image[1])
        img0 = img0[None,:]
        img1 = img1[None,:]
        '''
        # transform images to tensor for input to model
        img_stack = self.transform_images(player_image[0],player_image[1])
        pred_puck, loc = self.model(img_stack)

        # get current values for puck, dist, location, velocity
        p0_puck_onscreen = pred_puck[0,0].detach()
        p1_puck_onscreen = pred_puck[1,0].detach()

        p0_puck_dist = pred_puck[0,1].detach()
        p1_puck_dist = pred_puck[1,1].detach()

        p0_puck_loc = loc[0,:].detach()
        p1_puck_loc = loc[1,:].detach()

        p0_current_vel = player_state[0].get('kart').get('velocity')
        p1_current_vel = player_state[1].get('kart').get('velocity')
        p0_velocity_magnitude = np.linalg.norm([p0_current_vel[0],p0_current_vel[2]])
        p1_velocity_magnitude = np.linalg.norm([p1_current_vel[0],p1_current_vel[2]])

        p0_current_location = player_state[0].get('kart').get('front')
        p1_current_location = player_state[1].get('kart').get('front')

        # set constants for calculations
        steer_gain=2
        skid_thresh=0.5
        target_vel=25
        brake = False
        nitro = False
        loc_0 = loc[0].detach().numpy()
        steer_angle = steer_gain * loc_0[0]

        acceleration = 1.0 if velocity < target_vel else 0.0

        steer = np.clip(steer_angle * steer_gain, -1, 1)
        # Compute skidding
        if abs(steer_angle) > skid_thresh:
            drift = True
        else:
            drift = False

        # what to do if kart is stuck (velocity ~0, location ~ same as prev)

        # determine which goal is mine (based on team)

        # make one player defensive
        # e.g. sit in front of own goal (determine using coordinates)
        # only act if puck gets close
        # hit it away then back up to desired location

        print('puck',puck.detach(),'at distance',dist.detach())
        print('kart location',current_location, 'velocity',current_vel)
        if puck.detach()<0.5:
            print('puck out of frame')
            # I think the puck is out of frame
            brake = True
            acceleration = 0
            #steer = 0.420
        else:
            print('puck in frame')
            if np.absolute(loc_0[0])<0.1 and np.absolute(loc_0[1])<0.1:
                nitro = True
        self.prev_location = current_location
        self.prev_velocity = current_vel
        return [dict(acceleration=acceleration, brake=brake, drift=drift,
                     nitro=nitro, steer=steer)] * self.num_players
