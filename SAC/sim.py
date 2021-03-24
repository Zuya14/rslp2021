import cv2
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

import math

import lidar_util

robot_name = "urdf/pla-robot.urdf"

class sim:

    def __init__(self, _id=0, mode=p.DIRECT, sec=0.01):
        self._id = _id
        self.mode = mode
        self.phisicsClient = bc.BulletClient(connection_mode=mode)
        self.reset(sec=sec)

    def reset(self, x=1.0, y=1.0, theta=0.0, vx=0.0, vy=0.0, w=0.0, sec=0.01, action=None, clientReset=False):
        if clientReset:
            self.phisicsClient = bc.BulletClient(connection_mode=self.mode)

        self.sec = sec

        self.vx = vx
        self.vy = vy
        self.w = w

        self.phisicsClient.resetSimulation()
        self.robotUniqueId = 0 
        self.bodyUniqueIds = []
        self.phisicsClient.setTimeStep(sec)

        self.action = action if action is not None else [0.0, 0.0, 0.0]
        
        self.done = False

        self.loadBodys(x, y, theta)

        x, y = self.getState()[:2]
        self.distance = math.sqrt((x - 10.0)**2 + (y - 10.0)**2)
        self.old_distance = self.distance

    def getId(self):
        return self._id

    def loadBodys(self, x, y, theta):
        self.robotPos = (x,y,0)
        self.robotOri = p.getQuaternionFromEuler([0, 0, theta])

        self.robotUniqueId = self.phisicsClient.loadURDF(
            robot_name,
            basePosition=self.robotPos,
            baseOrientation = self.robotOri
            )
        
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(-1, -1, 0))]
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(-1, 11, 0))]
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(11, 11, 0))]
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(11, -1, 0))]

        for i in range(11):
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( i, 11, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( i, -1, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(-1,  i, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(11,  i, 0))]

        for i in range(11):
            if i not in [2, 9]:
                self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( i, 5, 0))]
            if i not in [1, 5, 8]:
                if i < 5:
                    self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( 5, i, 0))]
                else:
                    self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( 6, i, 0))]

    def step(self, action):

        self.old_distance = self.distance

        if not self.done:
            self.action = action

            l = math.sqrt(action[1]**2 + action[2]**2)
            cos = action[1] / l
            sin = action[2] / l

            v  = (self.action[0] + 1.0) * 0.5

            self.vx = v * cos
            self.vy = v * sin

            self.w = 0

            self.updateRobotInfo()

            self.phisicsClient.stepSimulation()

            if self.isContacts():
                self.done = True

            if self.isArrive():
                self.done = True

        else:
            self.vx = 0
            self.vy = 0
            self.w = 0

        x, y = self.getState()[:2]
        self.distance = math.sqrt((x - 10.0)**2 + (y - 10.0)**2)

        return self.done

    def observe(self, bullet_lidar):
        pos, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        scanDist = bullet_lidar.scanDistance(self.phisicsClient, pos, yaw, height=0.1)
        self.scanDist = scanDist / bullet_lidar.maxLen
        self.scanDist = self.scanDist.astype(np.float32)

        return self.scanDist

    def render(self, bullet_lidar):
        pos, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        scanDist = bullet_lidar.scanDistance(self.phisicsClient, pos, yaw, height=0.1)
        self.scanDist = scanDist / bullet_lidar.maxLen
        self.scanDist = self.scanDist.astype(np.float32)

        img = lidar_util.imshowLocalDistance("render"+str(self.phisicsClient), 800, 800, bullet_lidar, self.scanDist, maxLen=1.0, show=False, line=True)
        print(self.scanDist.shape)

        return img

    def updateRobotInfo(self):

        self.phisicsClient.resetBaseVelocity(
            self.robotUniqueId,
            linearVelocity=[self.vx, self.vy, 0],
            angularVelocity=[0, 0, self.w]
            )

        self.robotPos, self.robotOri = self.phisicsClient.getBasePositionAndOrientation(self.robotUniqueId)

    def getRobotPosInfo(self):
        return self.robotPos, self.robotOri

    def getState(self):
        pos, ori = self.getRobotPosInfo()
        return pos[0], pos[1], p.getEulerFromQuaternion(ori)[2], self.vx, self.vy, self.w 

    def close(self):
        self.phisicsClient.disconnect()

    def contacts(self):
        contactList = []
        for i in self.bodyUniqueIds[0:]: # 接触判定
            contactList += self.phisicsClient.getContactPoints(self.robotUniqueId, i)
        return contactList 

    def isContacts(self):
        return len(self.contacts()) > 0

    def isArrive(self):
        return self.distance < 0.5

    def isDone(self):
        return self.done

if __name__ == '__main__':

    # sim = sim(0, mode=p.GUI, sec=0.001)
    sim = sim(0, mode=p.DIRECT)

    from bullet_lidar import bullet_lidar

    resolusion = 12
    deg_offset = 90.
    rad_offset = deg_offset*(math.pi/180.0)
    startDeg = -180. + deg_offset
    endDeg = 180. + deg_offset
    maxLen = 20.
    minLen = 0.
    lidar = bullet_lidar(startDeg, endDeg, resolusion, maxLen, minLen)

    while True:
        action = np.array([0.5, 0.5, 1])
        # action = np.random.rand(3)

        sim.step(action=action)

        # print(np.concatenate([action, sim.action]))

        cv2.imshow("sim", sim.render(lidar))
        if cv2.waitKey(1) >= 0:
            break