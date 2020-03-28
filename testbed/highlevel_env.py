"""
Environment
 (1) set the attribute of Node, App, BasicThroughput,
     Currently we fix the No. of App with 9, homogeneous cluster.
     The No. of Nodes could be set, No. of Containers in the batch is not required to know
 (2) Functions:
    1. _state_reset: clear state matrix
    2. step: allocate one container to a node, return new state
    3. get_tput_total_env: get the throughput of the entire cluster (after each episode)
"""
import numpy as np
from testbed.SubScheduler import NineNodeAPI
from testbed.simulator.simulator import Simulator


class LraClusterEnv():

    def __init__(self, num_nodes):
        #: Cluster configuration
        self.NUM_NODES = num_nodes # node_id: 0,1,2,...
        #: fixed 9 apps
        self.NUM_APPS = 7
        #: initialized state to zero matrix
        self._state_reset()
        # clustering
        self.baisc_oath_name = 'cpo_separate_level_sc_'
        path_surffix = "./checkpoint/"
        self.nine_node_api_0 = NineNodeAPI(path_name=self.baisc_oath_name + '0', surffix='0', path_surffix=path_surffix)
        self.nine_node_api_1 = NineNodeAPI(path_name=self.baisc_oath_name + '10', surffix='10', path_surffix=path_surffix)
        self.nine_node_api_2 = NineNodeAPI(path_name=self.baisc_oath_name + '20', surffix='20', path_surffix=path_surffix)
        self.nine_node_api_3 = NineNodeAPI(path_name=self.baisc_oath_name + '30', surffix='30', path_surffix=path_surffix)
        self.nine_node_api_4 = NineNodeAPI(path_name=self.baisc_oath_name + '40', surffix='40', path_surffix=path_surffix)
        self.nine_node_api_5 = NineNodeAPI(path_name=self.baisc_oath_name + '50', surffix='50', path_surffix=path_surffix)
        self.nine_node_api_6 = NineNodeAPI(path_name=self.baisc_oath_name + '60', surffix='60', path_surffix=path_surffix)
        self.nine_node_api_7 = NineNodeAPI(path_name=self.baisc_oath_name + '70', surffix='70', path_surffix=path_surffix)
        self.nine_node_api_8 = NineNodeAPI(path_name=self.baisc_oath_name + '80', surffix='80', path_surffix=path_surffix)
        self.nine_node_api_9 = NineNodeAPI(path_name=self.baisc_oath_name + '90', surffix='90', path_surffix=path_surffix)
        self.nine_node_api_10 = NineNodeAPI(path_name=self.baisc_oath_name + '100', surffix='100', path_surffix=path_surffix)

        self.nine_node_api_11 = NineNodeAPI(path_name=self.baisc_oath_name + '110', surffix='110', path_surffix=path_surffix)
        # self.nine_node_api_12 = NineNodeAPI(path_name=self.baisc_oath_name + '120', surffix='120', path_surffix=path_surffix)
        # self.nine_node_api_13 = NineNodeAPI(path_name=self.baisc_oath_name + '130', surffix='130', path_surffix=path_surffix)
        # self.nine_node_api_14 = NineNodeAPI(path_name=self.baisc_oath_name + '140', surffix='140', path_surffix=path_surffix)
        # self.nine_node_api_15 = NineNodeAPI(path_name=self.baisc_oath_name + '150', surffix='150', path_surffix=path_surffix)
        # self.nine_node_api_16 = NineNodeAPI(path_name=self.baisc_oath_name + '160', surffix='160', path_surffix=path_surffix)
        # self.nine_node_api_17 = NineNodeAPI(path_name=self.baisc_oath_name + '170', surffix='170', path_surffix=path_surffix)
        # self.nine_node_api_18 = NineNodeAPI(path_name=self.baisc_oath_name + '180', surffix='180', path_surffix=path_surffix)
        # self.nine_node_api_19 = NineNodeAPI(path_name=self.baisc_oath_name + '190', surffix='190', path_surffix=path_surffix)
        # self.nine_node_api_20 = NineNodeAPI(path_name=self.baisc_oath_name + '200', surffix='200', path_surffix=path_surffix)

        self.sim = Simulator()



    def _state_reset(self):
        self.state = np.zeros([self.NUM_NODES, self.NUM_APPS])

    def reset(self):
        self._state_reset()
        return self._get_state()

    def step(self, action, appid):
        """
        :param action: node chosen
        :param appid: current app_id of the container to be allocated
        :return: new state after allocation
        """
        curr_app = appid
        self.state[action][curr_app] += 1  # locate
        state = self._get_state()
        return state

    def _get_state(self):
        return self.state

    @property
    def _get_throughput(self):

        state_all = np.empty([0, self.NUM_APPS])

        for nid in range(self.NUM_NODES):
            container_list = self.state[nid]
            num_container = sum(container_list)
            predictor_class = int((num_container-1)/10)
            if predictor_class > 11:
                predictor_class = 11
            assert (predictor_class >= 0) & (predictor_class <= 11)
            if predictor_class == 0:
                state_this = self.nine_node_api_0.get_total_tput(container_list)
            elif predictor_class == 1:
                state_this = self.nine_node_api_1.get_total_tput(container_list)
            elif predictor_class == 2:
                state_this = self.nine_node_api_2.get_total_tput(container_list)
            elif predictor_class == 3:
                state_this = self.nine_node_api_3.get_total_tput(container_list)
            elif predictor_class == 4:
                state_this = self.nine_node_api_4.get_total_tput(container_list)
            elif predictor_class == 5:
                state_this = self.nine_node_api_5.get_total_tput(container_list)
            elif predictor_class == 6:
                state_this = self.nine_node_api_6.get_total_tput(container_list)
            elif predictor_class == 7:
                state_this = self.nine_node_api_7.get_total_tput(container_list)
            elif predictor_class == 8:
                state_this = self.nine_node_api_8.get_total_tput(container_list)
            elif predictor_class == 9:
                state_this = self.nine_node_api_9.get_total_tput(container_list)
            elif predictor_class == 10:
                state_this = self.nine_node_api_10.get_total_tput(container_list)
            elif predictor_class == 11:
                state_this = self.nine_node_api_11.get_total_tput(container_list)

            state_all = np.append(state_all, state_this, 0)
        if self. getTput:
            total_tput = (self.sim.predict(state_all.reshape(-1, self.NUM_APPS)) * state_all).sum()
        else:
            total_tput = 0
        state = state_all
        # list_check_per_app = (state > 1).sum()  # + max((env.state - 1).max(), 0)
        # list_check_sum = sum(state.sum(1) > 8)  # + max(max(env.state.sum(1) - params['container_limitation per node']), 0)
        # list_check_coex = sum((state[:, 1] > 0) * (state[:, 2] > 0))
        # list_check = list_check_sum + list_check_coex + list_check_per_app
        list_check = 0
        for node in range(self.NUM_NODES * 27):
            for app in range(self.NUM_APPS):
                if state[node, :].sum() > 8 or state[node, app] > 1 or (app == 1 and state[node, 2] > 0) or (app == 2 and state[node, 1] > 0):
                    list_check += state[node, app]

        return total_tput, 0, 0, 0, list_check

    def get_tput_total_env(self, getTput=True):
        self. getTput = getTput
        return self._get_throughput