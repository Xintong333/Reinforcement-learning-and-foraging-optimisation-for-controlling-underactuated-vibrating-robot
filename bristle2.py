import numpy as np
from os import path
import rlschool
import sim
import time
import math
import random

class Bristle(object):
    def __init__(self):
        sim.simxFinish(-1)
        self.clientID = None

        self.forward_distance = 0 #
        self.back_distance = 0 #
        self.left_distance = 0 #
        self.right_distance = 0 #

        self.max_sensor_distance = 20 #
        
        self.train_flag = 1 #1=train,0=eval or test
        self.test_goal_num = 9#9targets
        

        self.target_position_handle = None #
        self.step_size = 0.1

        self.position = None
        self.robot_angle = None
        self.target_position = None
        self.Bristle_target = None
        self.goal_name = None
        self.start_connection()#Connections
        self.laser_sensors_init()
        self.object_init()
        self.get_position()
        #self.close()

        self.first_dis1 = cau_dis2(self.target_position, self.position)
        #print(f"Initial distance is{self.first_dis1}")
        self.lastdis = self.first_dis1

        #self.yaw,self.pitch,self.roll = Bristle.get_angle(self.robot_angle)#取欧拉角

    def start_connection(self):
        self.clientID = sim.simxStart('127.0.0.1',19997,True,True,5000,5) # start a connection
        if self.clientID != -1:
            print ('Connected to remote API server')
        else:
            print ('Failed connecting to remote API server')

    def laser_sensors_init(self):
        
        [returnCode, signalValue1] = sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime1',sim.simx_opmode_streaming)
        [returnCode, signalValue2] = sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime2',sim.simx_opmode_streaming)
        [returnCode, signalValue3] = sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime3',sim.simx_opmode_streaming)
        [returnCode, signalValue4] = sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime4',sim.simx_opmode_streaming)
        time.sleep(0.1)

    def read_laser_sensors(self):
        [returnCode_1, signalValue1] = sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime1',sim.simx_opmode_buffer)
        self.forward_distance = sim.simxUnpackFloats(signalValue1)[1] if (returnCode_1 == 0) else self.max_sensor_distance

        [returnCode_2, signalValue2] = sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime2',sim.simx_opmode_buffer)
        self.back_distance = sim.simxUnpackFloats(signalValue2)[1] if (returnCode_2 == 0) else self.max_sensor_distance

        [returnCode_3, signalValue3] = sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime3',sim.simx_opmode_buffer)
        self.left_distance = sim.simxUnpackFloats(signalValue3)[1] if (returnCode_3 == 0) else self.max_sensor_distance

        [returnCode_4, signalValue4] = sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime4',sim.simx_opmode_buffer)
        self.right_distance = sim.simxUnpackFloats(signalValue4)[1] if (returnCode_4 == 0) else self.max_sensor_distance



    def object_init(self):
        self.goal_name = 'Goal1'
        """"
        if(self.train_flag):
            gogogo = random.randint(1,8)
            goal_name = 'Goal' + str(gogogo)

        else:
            gogogo = self.test_goal_num
            goal_name = 'Goal' + str(gogogo)
            self.test_goal_num += 1
            if(self.test_goal_num == 13):
                self.test_goal_num = 9
        print(goal_name,end='')
        """
        err_code, self.target_position_handle = sim.simxGetObjectHandle(self.clientID, self.goal_name, sim.simx_opmode_blocking)
        if err_code != sim.simx_return_ok:
            print("Something is wrong!!!")
        ret, self.target_position = sim.simxGetObjectPosition(self.clientID, self.target_position_handle, -1, sim.simx_opmode_blocking)
        if ret != sim.simx_return_ok:
            print("Something is wrong!!")
        err_code, self.Bristle_target = sim.simxGetObjectHandle(self.clientID,"bristle", sim.simx_opmode_blocking)
        
    def step(self, action):
        #Determine if the game is over and give a reward
        u = np.argmax(action) + 1
        if(u == 1):
            self.go_left_bigR()
        elif(u == 2):
            self.go_right_bigR()
        elif(u == 3):
            self.go_left()
        elif (u == 4):
            self.go_right()
        #time.sleep(0.1)
        '''
        step1:Determine if you have reached your destination
        '''
        self.get_position()
        self.read_laser_sensors()
        dis = cau_dis2(self.position, self.target_position)
        if(dis < 0.28):
            costs = 250
            self.goal_name = 'Goal2'

            print({dis})
            err_code, self.target_position_handle = sim.simxGetObjectHandle(self.clientID, self.goal_name, sim.simx_opmode_blocking)
            ret, self.target_position = sim.simxGetObjectPosition(self.clientID, self.target_position_handle, -1, sim.simx_opmode_blocking)
            dis = cau_dis2(self.position, self.target_position)
            print({dis})
            print('got the first one')
            if (dis < 0.28):
                costs = costs + 150

                self.goal_name = 'Goal3'
                err_code, self.target_position_handle = sim.simxGetObjectHandle(self.clientID, self.goal_name, sim.simx_opmode_blocking)
                ret, self.target_position = sim.simxGetObjectPosition(self.clientID, self.target_position_handle, -1, sim.simx_opmode_blocking)
                dis = cau_dis2(self.position, self.target_position)
                print({self.goal_name})
                print('got the second one')
                if (dis < 0.28):
                    costs = costs + 150
                    self.goal_name = 'Goal3'
                    err_code, self.target_position_handle = sim.simxGetObjectHandle(self.clientID, self.goal_name, sim.simx_opmode_blocking)
                    ret, self.target_position = sim.simxGetObjectPosition(self.clientID, self.target_position_handle, -1, sim.simx_opmode_blocking)
                    dis = cau_dis2(self.position, self.target_position)
                    print({self.goal_name})
                    print('got the Third one')
                    if (dis < 0.28):
                        costs = 250 + costs
                        print('got the forth one')
                        return self._get_obs(), costs, True, {}

        '''
        step2:Determine if it is very close to the wall, if it is close it is dangerous
        '''
        if( dis > 2.5 ):
            costs = -500
            print("too far!!!")
            return self._get_obs(), costs, True, {}
        
        #print(self.lastdis,dis)
        costs = (math.pow((20 - dis), 2) - math.pow((20 - self.lastdis), 2))
        #(self.lastdis - dis) * 100
        self.lastdis = dis
        #print(action, costs)
        return self._get_obs(), costs, False, {}

    def reset(self):
        self.stopflag = sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        time.sleep(4)
        self.startflag = sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)
        #time.sleep(1)
        self.laser_sensors_init()
        self.object_init()
        self.get_position()
        self.read_laser_sensors()

        self.first_dis1 = cau_dis2(self.target_position, self.position)
        self.lastdis = self.first_dis1
        print(f"  Initial distance is{self.first_dis1}")

        return self._get_obs()

    '''
    def _get_obs(self):
        return np.array([(self.target_position[0] - self.position[0]), (self.target_position[1] - self.position[1]), 
        self.forward_distance, self.back_distance, self.left_distance, self.right_distance
         ] )
    '''

    def _get_obs(self):
        return np.array([(self.target_position[0] - self.position[0]), (self.target_position[1] - self.position[1])])
        

    def get_position(self):
        ret, self.position = sim.simxGetObjectPosition(self.clientID, self.Bristle_target, -1, sim.simx_opmode_blocking)
        if ret != sim.simx_return_ok:
            print("Something is wrong!!!")
    """
    def get_angle(self):
        ret, self.robot_angle = sim.simxGetObjectOrientation(self.clientID, self.Bristle_target, -1, sim.simx_opmode_blocking)
        if ret != sim.simx_return_ok:
            print("Something is wrong!!!")
    """
    def go_left_bigR(self):
        errorCode, motor_handle = sim.simxGetObjectHandle(self.clientID, 'Revolute_joint', sim.simx_opmode_oneshot_wait)
        errorCode = sim.simxSetJointTargetVelocity(self.clientID, motor_handle, 15, sim.simx_opmode_streaming)

    def go_right_bigR(self):
        errorCode, motor_handle = sim.simxGetObjectHandle(self.clientID, 'Revolute_joint',
                                                              sim.simx_opmode_oneshot_wait)
        errorCode = sim.simxSetJointTargetVelocity(self.clientID, motor_handle, 15, sim.simx_opmode_streaming)
    """"
    def go_back(self):
        sim.simxSetObjectPosition(
            self.clientID, self.Quadricopter_target, -1, 
            [self.position[0] - self.step_size, self.position[1], self.position[2]],
            sim.simx_opmode_oneshot
        )
    """
    
    def go_left(self):
        errorCode, motor_handle = sim.simxGetObjectHandle(self.clientID, 'Revolute_joint', sim.simx_opmode_oneshot_wait)
        errorCode = sim.simxSetJointTargetVelocity(self.clientID, motor_handle, -18, sim.simx_opmode_streaming)

    def go_right(self):
        errorCode, motor_handle = sim.simxGetObjectHandle(self.clientID, 'Revolute_joint', sim.simx_opmode_oneshot_wait)
        errorCode = sim.simxSetJointTargetVelocity(self.clientID, motor_handle, 18, sim.simx_opmode_streaming)

    def close(self):
        sim.simxGetPingTime(self.clientID)
        self.stopflag = sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        #sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)
        sim.simxFinish(self.clientID)

            
def cau_dis1(pos1,pos2):
    return math.sqrt( math.pow(pos1[0] - pos2[0], 2) + math.pow(pos1[1] - pos2[1], 2))

def cau_dis2(pos1,pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])