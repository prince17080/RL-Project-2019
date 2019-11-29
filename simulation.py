from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.disable_v2_behavior()
from sumolib import checkBinary
import datetime
import math
import timeit
import traci
import random
import matplotlib.patches as mpatches


import memory
from model import *
import traffic_generator

# add SUMO_HOME to the environment variable 'PATH'
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


SHOW_GUI = False
BATCH_FLAG = False
VERSION = 1
PATH = "./figures/version" + str(VERSION)
STATE_SPACE = 80
ACTION_SPACE = 4
MEMORY = 100000 # to store the data elements/samples
BATCH = 100
NO_OF_EPISODES = 10
NO_OF_CARS = 1000

PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

# below values are in seconds
GREEN_TIME = 10
YELLOW_TIME = 4
MAX_STEPS_PER_EPS = 5400


class Simulator:
    def __init__(self, _traffic_generator, _niterations, _gamma):
        self._traffic_generator = _traffic_generator
        self.niterations = _niterations
        self._gamma = _gamma
        self.epsilon = 1
        self.seed = 0
        self.steps = 0
        self._waiting_times = {}
        self._green_duration = GREEN_TIME
        self._yellow_duration = YELLOW_TIME
        self._sum_intersection_queue = 0
        self._old_waiting_time = 0
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_intersection_queue_store = []

    def _simulate(self, Session, memory, model1, model2, steps_todo):
        if (self.steps + steps_todo) >= MAX_STEPS_PER_EPS:  # do not do more steps than the maximum number of steps
            steps_todo = MAX_STEPS_PER_EPS - self.steps
        self.steps = self.steps + steps_todo  # update the step counter
        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._replay(memory, model1, model2, Session)  # training
            steps_todo -= 1
            intersection_queue = self._get_stats()
            self._sum_intersection_queue += intersection_queue
    
    def run(self, Session, memory, model1, model2, epsilon, port, sumo_cmd):
        # first, generate the route file for this simulation and set up sumo
        self._traffic_generator.generate_routefile(self.seed)
        self.seed = (self.seed + 1)%5
        traci.start(sumo_cmd, port)

        # inits
        self.steps = 0
        tot_neg_reward = 0
        old_total_wait = 0
        self._waiting_times = {}
        self._sum_intersection_queue = 0
        old_action = random.randint(0, ACTION_SPACE-1)
        while self.steps < MAX_STEPS_PER_EPS:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            
            current_total_wait = self._get_waiting_times()
            reward = old_total_wait - current_total_wait

            if model2 == None:
                action, reward = self.step_dqn(current_state, model1, epsilon, Session)
            else:
                action, reward = self.step_ddqn(current_state, model1, model2, epsilon, Session)


            # saving the data into the memory
            if self.steps != 0:
                memory.add_data((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self.steps != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(Session, memory, model1, model2, self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(Session, memory, model1, model2, self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            # old_total_wait = current_total_wait
            if reward < 0:
                tot_neg_reward += reward
            self.steps += 1

        self._save_stats(tot_neg_reward)
        print("Total reward: {}, Eps: {}".format(tot_neg_reward, epsilon))
        traci.close()

    def step_dqn(self, state, model, epsilon, Session):
        e = random.random()
        if e < epsilon:
            action = random.randint(0, ACTION_SPACE-1)
        else:
            action = np.argmax(model.predict_one(state, Session))

        current_waiting_time = self._get_waiting_times()
        reward =  self._old_waiting_time - current_waiting_time
        self._old_waiting_time = current_waiting_time

        return action, reward

    def step_ddqn(self, state, model1, model2, epsilon, Session):
        e = random.random()
        if e < epsilon:
            action = random.randint(0, ACTION_SPACE-1)
        else:
            if(random.random()<0.5):
                action = np.argmax(model1.predict_one(state, Session))
            else:
                action = np.argmax(model2.predict_one(state, Session))


        current_waiting_time = self._get_waiting_times()
        reward = self._old_waiting_time - current_waiting_time
        self._old_waiting_time = current_waiting_time

        return action, reward


    def _get_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        for veh_id in traci.vehicle.getIDList():
            wait_time_car = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            road_id = traci.vehicle.getRoadID(veh_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[veh_id] = wait_time_car
            else:
                if veh_id in self._waiting_times:
                    del self._waiting_times[veh_id]  # the car isnt in incoming roads anymore, delete his waiting time
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _replay(self, memory, model1, model2, Session):
        if(BATCH_FLAG == True):
            batch = memory.get_batch(BATCH)
            if len(batch) > 0:  # if there is at least 1 sample in the batch
                if(model2==None):
                    states = np.array([val[0] for val in batch])  # extract states from the batch
                    next_states = np.array([val[3] for val in batch])  # extract next states from the batch

                    # prediction
                    q_s_a = model1.predict_batch(states, Session)  # predict Q(state), for every sample
                    q_s_a_d = model1.predict_batch(next_states, Session)  # predict Q(next_state), for every sample

                    # setup training arrays
                    x = np.zeros((len(batch), STATE_SPACE))
                    y = np.zeros((len(batch), ACTION_SPACE))

                    for i, b in enumerate(batch):
                        state, action, reward, next_state = b[0], b[1], b[2], b[3]  # extract data from one sample
                        current_q = q_s_a[i]  # get the Q(state) predicted before


                        current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                        x[i] = state
                        y[i] = current_q  # Q(state) that includes the updated action value

                    model1.train_batch(Session, x, y)  # train the N
                
                else:
                    if(random.random()<=0.5):
                        states = np.array([val[0] for val in batch])  # extract states from the batch
                        next_states = np.array([val[3] for val in batch])  # extract next states from the batch

                        # prediction
                        q_s_a = model1.predict_batch(states, Session)  # predict Q(state), for every sample
                        q_s_a_d = model2.predict_batch(next_states, Session)  # predict Q(next_state), for every sample

                        # setup training arrays
                        x = np.zeros((len(batch),  STATE_SPACE))
                        y = np.zeros((len(batch), ACTION_SPACE))

                        for i, b in enumerate(batch):
                            state, action, reward, next_state = b[0], b[1], b[2], b[3]  # extract data from one sample
                            current_q = q_s_a[i]  # get the Q(state) predicted before


                            current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                            x[i] = state
                            y[i] = current_q  # Q(state) that includes the updated action value

                        model1.train_batch(Session, x, y)  # train the NN
                    else:
                        states = np.array([val[0] for val in batch])  # extract states from the batch
                        next_states = np.array([val[3] for val in batch])  # extract next states from the batch

                        # prediction
                        q_s_a = model2.predict_batch(states, Session)  # predict Q(state), for every sample
                        q_s_a_d = model1.predict_batch(next_states, Session)  # predict Q(next_state), for every sample

                        # setup training arrays
                        x = np.zeros((len(batch), STATE_SPACE))
                        y = np.zeros((len(batch), ACTION_SPACE))

                        for i, b in enumerate(batch):
                            state, action, reward, next_state = b[0], b[1], b[2], b[3]  # extract data from one sample
                            current_q = q_s_a[i]  # get the Q(state) predicted before


                            current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                            x[i] = state
                            y[i] = current_q  # Q(state) that includes the updated action value

                        model2.train_batch(Session, x, y)  # train the NN

        elif(len(memory.data)>0):
            if(model2==None):
                state,action,reward,next_state=memory.data[-1]
                q_s_a = model1.predict_one(state, Session)[0]
                q_s_a_d = model1.predict_one(next_state, Session)[0]
                current_q = q_s_a
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d)  # update Q(state, action)
                x=np.zeros((1, STATE_SPACE))
                y=np.zeros((1, ACTION_SPACE))
                x[0]=state
                y[0]=current_q
                model1.train_batch(Session, x, y)  # train the NN
            else:
                if(random.random()<=0.5):
                    state,action,reward,next_state=memory.data[-1]
                    q_s_a = model1.predict_one(state, Session)[0]
                    q_s_a_d = model2.predict_one(next_state, Session)[0]
                    current_q = q_s_a
                    current_q[action] = reward + self._gamma * np.amax(q_s_a_d)  # update Q(state, action)
                    x=np.zeros((1, STATE_SPACE))
                    y=np.zeros((1, ACTION_SPACE))
                    x[0]=state
                    y[0]=current_q
                    model1.train_batch(Session, x, y)  # train the NN
                else:
                    state,action,reward,next_state=memory.data[-1]
                    q_s_a = model2.predict_one(state, Session)[0]
                    q_s_a_d = model1.predict_one(next_state, Session)[0]
                    current_q = q_s_a
                    current_q[action] = reward + self._gamma * np.amax(q_s_a_d)  # update Q(state, action)
                    x=np.zeros((1, STATE_SPACE))
                    y=np.zeros((1, ACTION_SPACE))
                    x[0]=state
                    y[0]=current_q
                    model2.train_batch(Session, x, y)


    # NOT COMPLETE, prediction part #
    def _choose_action(self, state, epsilon, model, Session):
        if random.random() < epsilon:
            return random.randint(0, ACTION_SPACE - 1) # random action
        else:
            return np.argmax(model.predict_one(state, Session)) # the best action given the current state


    # SET IN SUMO THE CORRECT YELLOW PHASE
    def _set_yellow_phase(self, old_action):
        yellow_phase = old_action * 2 + 1 # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase("TL", yellow_phase)

    # SET IN SUMO A GREEN PHASE
    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    # RETRIEVE THE STATS OF THE SIMULATION FOR ONE SINGLE STEP
    def _get_stats(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        intersection_queue = halt_N + halt_S + halt_E + halt_W
        return intersection_queue

    # RETRIEVE THE STATE OF THE INTERSECTION FROM SUMO
    def _get_state(self):
        state = np.zeros(STATE_SPACE)

        for veh_id in traci.vehicle.getIDList():
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to TL, lane_pos = 0
            lane_group = -1  # just dummy initialization
            valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            # distance in meters from the TLS -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located - _3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7

            if lane_group >= 1 and lane_group <= 7:
                veh_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                veh_position = lane_cell
                valid_car = True

            if valid_car:
                state[veh_position] = 1  # write the position of the car veh_id in the state array

        return state

    # SAVE THE STATS OF THE EPISODE TO PLOT THE GRAPHS AT THE END OF THE SESSION
    def _save_stats(self, tot_neg_reward):
        self._reward_store.append(tot_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_intersection_queue)  # total number of seconds waited by cars in this episode
        self._avg_intersection_queue_store.append(self._sum_intersection_queue / MAX_STEPS_PER_EPS)  # average number of queued cars per step, in this episode

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_intersection_queue_store(self):
        return self._avg_intersection_queue_store

# PLOT AND SAVE THE STATS ABOUT THE SESSION
def graphs(sim_runner_dqn,sim_runner_ddqn,sim_runner_dqn_batch,sim_runner_ddqn_batch ,total_episodes):
    x=list()
    for i in range(total_episodes):
        x.append(i+1)

    plt.plot(x,sim_runner_dqn_batch.reward_store,'k')
    plt.plot(x,sim_runner_ddqn_batch.reward_store,'y')
    yellow_patch = mpatches.Patch(color='yellow', label='DDQN_Batch')
    black_patch = mpatches.Patch(color='black', label='DQN_Batch')
    plt.legend(handles=[yellow_patch,black_patch],loc=1)
    plt.xlabel('episodes')
    plt.ylabel('total_reward')
    plt.show()
    
    plt.plot(x,sim_runner_dqn_batch.cumulative_wait_store,'k')
    plt.plot(x,sim_runner_ddqn_batch.cumulative_wait_store,'y')
    yellow_patch = mpatches.Patch(color='yellow', label='DDQN_Batch')
    black_patch = mpatches.Patch(color='black', label='DQN_Batch')
    plt.legend(handles=[yellow_patch,black_patch],loc=1)
    plt.xlabel('episodes')
    plt.ylabel('total waiting time')
    plt.show()
    
    plt.plot(x,sim_runner_dqn_batch.avg_intersection_queue_store,'k')
    plt.plot(x,sim_runner_ddqn_batch.avg_intersection_queue_store,'y')
    yellow_patch = mpatches.Patch(color='yellow', label='DDQN_batch')
    black_patch = mpatches.Patch(color='black', label='DQN_batch')
    plt.legend(handles=[yellow_patch,black_patch],loc=1)
    plt.xlabel('episodes')
    plt.ylabel('average queue length')
    plt.show()
    
    plt.plot(x,sim_runner_dqn_batch.reward_store,'k')
    plt.plot(x,sim_runner_dqn.reward_store,'y')
    yellow_patch = mpatches.Patch(color='yellow', label='Simple DQN')
    black_patch = mpatches.Patch(color='black', label='DQN_Batch')
    plt.legend(handles=[yellow_patch,black_patch],loc=1)
    plt.xlabel('episodes')
    plt.ylabel('total_reward')
    plt.show()
    
    plt.plot(x,sim_runner_ddqn_batch.reward_store,'k')
    plt.plot(x,sim_runner_ddqn.reward_store,'y')
    yellow_patch = mpatches.Patch(color='yellow', label='Simple DDQN')
    black_patch = mpatches.Patch(color='black', label='DDQN_Batch')
    plt.legend(handles=[yellow_patch,black_patch],loc=1)
    plt.xlabel('episodes')
    plt.ylabel('total_reward')
    plt.show()




if SHOW_GUI:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')

_traffic_generator = traffic_generator.TrafficGenerator(MAX_STEPS_PER_EPS, NO_OF_CARS)
sumo_cmd = [sumoBinary, "-c", "intersection/tlcs_config_train.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(MAX_STEPS_PER_EPS)]

model1 = Model(STATE_SPACE, ACTION_SPACE, BATCH)
model2 = Model(STATE_SPACE, ACTION_SPACE, BATCH)
model3 = Model(STATE_SPACE, ACTION_SPACE, BATCH)

memory1 = memory.Memory(MEMORY)
memory2 = memory.Memory(MEMORY)

saver = tf.train.Saver()


#simple ddqn
with tf.Session() as Session:
    Session.run(model1.var_init)
    Session.run(model2.var_init)
    simulator_ddqn_simple = Simulator(_traffic_generator, MAX_STEPS_PER_EPS, 0.75)
    episode_count = 0
    port = 5000
    while episode_count < NO_OF_EPISODES:
        epsilon = 0.1
        print('----- Episode {} of {}'.format(episode_count + 1, NO_OF_EPISODES))
        start = timeit.default_timer()
        simulator_ddqn_simple.run(Session, memory1, model1, model2, epsilon, port, sumo_cmd)  # run the simulation
        stop = timeit.default_timer()
        print('Time: ', round(stop - start, 1))
        episode_count += 1
        port += 1

#simple dqn
with tf.Session() as Session:
    Session.run(model3.var_init)
    simulator_dqn_simple = Simulator(_traffic_generator, MAX_STEPS_PER_EPS, 0.75)
    episode_count = 0
    port = 5000
    while episode_count < NO_OF_EPISODES:
        epsilon = 0.1
        print('----- Episode {} of {}'.format(episode_count + 1, NO_OF_EPISODES))
        start = timeit.default_timer()
        simulator_dqn_simple.run(Session, memory2, model1, None, epsilon, port, sumo_cmd)  # run the simulation
        stop = timeit.default_timer()
        print('Time: ', round(stop - start, 1))
        episode_count += 1
        port += 1


model1 = Model(STATE_SPACE, ACTION_SPACE, BATCH)
model2 = Model(STATE_SPACE, ACTION_SPACE, BATCH)
model3 = Model(STATE_SPACE, ACTION_SPACE, BATCH)

memory1 = memory.Memory(MEMORY)
memory2 = memory.Memory(MEMORY)

saver = tf.train.Saver()

#batch DDQN
with tf.Session() as Session:
    Session.run(model1.var_init)
    Session.run(model2.var_init)
    simulator_ddqn = Simulator(_traffic_generator, MAX_STEPS_PER_EPS, 0.75)
    episode_count = 0
    port = 7000

    while episode_count < NO_OF_EPISODES:
        print("Episode " + str(episode_count) + " started")
        epsilon = 0.1
        start_time = timeit.default_timer()
        simulator_ddqn.run(Session, memory1, model1, model2, epsilon, port, sumo_cmd)
        end_time = timeit.default_timer()
        print("Time taken: ", str(end_time - start_time))
        episode_count += 1
        port += 1

    print("Simulation Done")

# batch DQN
with tf.Session() as Session:
    Session.run(model3.var_init)
    simulator_dqn = Simulator(_traffic_generator, MAX_STEPS_PER_EPS, 0.75)
    episode_count = 0
    port = 14000

    while episode_count < NO_OF_EPISODES:
        print("Episode " + str(episode_count) + " started")
        epsilon = 0.1
        start_time = timeit.default_timer()
        simulator_dqn.run(Session, memory2, model3, None, epsilon, port, sumo_cmd)
        end_time = timeit.default_timer()
        print("Time taken: ", str(end_time - start_time))
        episode_count += 1
        port += 1

    print("Simulation Done")


graphs(simulator_dqn_simple,simulator_ddqn_simple,simulator_dqn,simulator_ddqn,NO_OF_EPISODES)

