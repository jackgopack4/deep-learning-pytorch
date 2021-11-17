import pystk
import numpy as np

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    #print(aim_point)
    accel_val = 1.
    if current_vel >= 10:
        accel_val = 0.5
    elif current_vel >= 15:
        accel_val = 0.25
    elif current_vel >= 20:
        accel_val = 0.
    
    '''
    if aim_point[1] > 0 and current_vel < 20:
        accel_val = aim_point[1]
    '''
    action.acceleration = accel_val
    action.drift = False
    steer_val = []
    steer_val.append(aim_point[0] * 2.5)
    #print('steer value =',steer_val[0])
    same_sign = abs(aim_point[0]) + abs(steer_val[0]) == abs(aim_point[0] + steer_val[0])
    if abs(aim_point[0]) > 0.6 and same_sign and abs(aim_point[0] < 0.9):
        action.drift = True
    '''
    if aim_point[0] != 0.:
        steer_val = aim_point[0] / 2
        if abs(aim_point[0]) > 0.5:
          action.drift = True
    '''
    steer_val = np.clip(steer_val,-1,1)
    action.steer = steer_val[0]

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
