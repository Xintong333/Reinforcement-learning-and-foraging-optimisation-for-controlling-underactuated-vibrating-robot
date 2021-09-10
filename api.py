# Make sure to have the server side running in CoppeliaSim:
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

try:
    import sim
except:
    print('--------------------------------------------------------------')
    print('"sim.py" could not be imported. This means very probably that')
    print('either "sim.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "sim.py"')
    print('--------------------------------------------------------------')
    print('')

import time
import math

print('Program started')
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
# print(clientID)#0

if clientID != -1:
    print('Connected to remote API server')
    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
    if res == sim.simx_return_ok:
        print('Number of objects in the scene: ', len(objs))
    else:
        print('Remote API function call returned with error code: ', res)

    ret, targetObj = sim.simxGetObjectHandle(clientID, 'Bristle_target', sim.simx_opmode_blocking)
    if ret != sim.simx_return_ok:
        print("!!!")

    ret, arr = sim.simxGetObjectPosition(clientID, targetObj, -1, sim.simx_opmode_blocking)
    if ret == sim.simx_return_ok:
        print(arr)
    time.sleep(1)

    # Now retrieve streaming data (i.e. in a non-blocking fashion):
    startTime = time.time()
    # sim.simxGetIntegerParameter(clientID,sim.sim_intparam_mouse_x,sim.simx_opmode_streaming) # Initialize streaming
    while time.time() - startTime < 5:
        # returnCode,data=sim.simxGetIntegerParameter(clientID,sim.sim_intparam_mouse_x,sim.simx_opmode_buffer) # Try to retrieve the streamed data

        ret, arr = sim.simxGetObjectPosition(clientID, targetObj, -1, sim.simx_opmode_blocking)
        if ret == sim.simx_return_ok:
            print(arr)

        if ret == sim.simx_return_ok:
            errorCode, motor_handle = sim.simxGetObjectHandle(clientID, 'Revolute_joint', sim.simx_opmode_oneshot_wait)
            errorCode = sim.simxSetJointTargetVelocity(clientID, motor_handle,0, sim.simx_opmode_streaming)
        else:
            errorCode, motor_handle = sim.simxGetObjectHandle(clientID, 'Revolute_joint', sim.simx_opmode_oneshot_wait)
            errorCode = sim.simxSetJointTargetVelocity(clientID, motor_handle,1000, sim.simx_opmode_streaming)

        time.sleep(0.5)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')