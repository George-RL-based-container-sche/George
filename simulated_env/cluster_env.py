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
from simulated_env.util.Util_Node_App import Application
from simulated_env.util.Util_Node_App import Node

class LraClusterEnv():

    def __init__(self, num_nodes):
        #: Cluster configuration
        self.NUM_NODES = num_nodes # node_id: 0,1,2,...

        #: homogeneous cluster
        # TODO: heterogeneous cluster
        self.NODE_CAPACITY_NETWORK = [3600] * self.NUM_NODES
        self.NODE_CAPACITY_MEMBW = [2100] * self.NUM_NODES
        self.NODE_CAPACITY_CACHE = [2700] * self.NUM_NODES

        #: fixed 9 apps
        self.NUM_APPS = 20
        self.BasicThroughput = [100] * self.NUM_APPS  # normalized basic throughput

        #: Application Resource Usage Per Query
        self.NETWORK_BW_PER_QUERY = [1, 2, 3, 1, 2, 3, 1, 2, 3,1, 2, 3, 1, 2, 3, 1, 2, 3,1,2]  # network bandwidth
        self.MEM_BW_PER_QUERY = [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]  # memory bandwidth
        self.CACHE_PER_QUERY = [3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]  # cache footprint

        #: initialized state to zero matrix
        self._state_reset()

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
        """
       First create the node instance for each node, along with the application instances residing in it
           For each node, maintain the app list, node capacity of network bw, mem bw and so on
           For each app, maintain the container number, nw/mem bw consumption for each query
       Second calculate the throughput for each app and each node, based on interference analysis
       :return: total throughput for all the nodes and all the containers residing in them
        """
        node_list = []
        for nid in range(self.NUM_NODES):
            node = Node(nid,
                        self.NODE_CAPACITY_NETWORK[nid],
                        self.NODE_CAPACITY_MEMBW[nid],
                        self.NODE_CAPACITY_CACHE[nid],
                        self)
            for aid in range(self.NUM_APPS):
                num_container = self.state[nid][aid]
                if num_container > 0:
                    app = Application(aid,
                                      self.BasicThroughput[aid],
                                      self.NETWORK_BW_PER_QUERY[aid],
                                      self.MEM_BW_PER_QUERY[aid],
                                      self.CACHE_PER_QUERY[aid],
                                      num_container)
                    node.add_application(app)
            node.calculate_new_tput()
            node_list.append(node)

        total_tput = 0
        for node in node_list:
            total_tput += node.total_tput()
        return total_tput

    def get_SLA_violation(self, sla):
        node_list = []
        for nid in range(self.NUM_NODES):
            node = Node(nid,
                        self.NODE_CAPACITY_NETWORK[nid],
                        self.NODE_CAPACITY_MEMBW[nid],
                        self.NODE_CAPACITY_CACHE[nid],
                        self)
            for aid in range(self.NUM_APPS):
                num_container = self.state[nid][aid]
                if num_container > 0:
                    app = Application(aid,
                                      self.BasicThroughput[aid],
                                      self.NETWORK_BW_PER_QUERY[aid],
                                      self.MEM_BW_PER_QUERY[aid],
                                      self.CACHE_PER_QUERY[aid],
                                      num_container)
                    node.add_application(app)
            node.calculate_new_tput()
            node_list.append(node)

        violation = 0
        for node in node_list:
            violation += node.sla_violation(sla)

        return violation


    def get_tput_total_env(self):
        return self._get_throughput

    def _create_preference(self):
        from scipy.sparse import diags

        # cardinality of application itself
        # a b c d e f g h i
        # - 2 i i - 2 2 i -
        # -: cardinality == negative, i.e., anti-affinity
        # 2: cardinality == 2, if 1 < num <= cardinality, +50%, if num > cardinality, -50%
        # i: cardinatliy == infinity, i.e., affinity
        preference = diags(
            [-1, 2, np.inf, np.inf, -1, 2, 2, np.inf, -1,
             3, np.inf, 1,   -1, 4, 2, np.inf, -1, np.inf,
             4, 1]
        ).todense()

        # cardinality of application
        # -: abc, def, ghi
        # i: adg, beh, cfi
        # 2: af, di
        # 5: cd, fg
        # 0: others
        inf_group = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0,12,19],[ 5, 19, 16],[ 7, 11, 12],[ 6, 13,  9],[ 8, 17, 18],[15,  3,  4],[18, 16, 17],[ 6,  0,  5],[11, 10, 15],[ 2,  7,  8],[12, 13,  3],[18, 0, 2],[5, 3, 1],[7, 9, 11],[12, 14, 17],[8, 19, 6]]

        two_group = [[0, 5], [3, 8],[19,  8],[ 0, 11],[ 3,  2],[12,  7],[17,  4],[14,  1],[ 6, 13],[18,  5],[10, 16],[15,  9],[ 6, 18],[ 2, 10],[ 7, 17],[15, 14],[ 8, 16],[ 0,  1],[ 4, 12],[13, 19],[ 3,  5],[ 9, 11]]
        fiv_group = [[2, 3], [5, 6],[12,  2],[17, 10],[19, 14],[ 9, 13],[15,  3],[16,  6],[ 1,  8],[18,  5],[ 0,  4],[ 7, 11],[ 5, 14],[18, 11],[ 4,  3],[ 1,  7],[13, 10],[ 0, 12],[16, 15],[19,  8],[17,  2],[ 9,  6]]
        three_group = [[ 1, 15],[ 5,  9],[ 7, 18],[14, 19],[ 4, 12],[ 8, 13],[ 0,  3],[10, 17],[11,  6],[16,  2],[ 3,  8],[14, 19],[ 2, 12],[10, 18],[ 1, 11],[ 0,  6],[15,  5],[ 9, 16],[ 4, 17],[ 7, 13]]
        four_group = [[13,  0],[11, 18],[ 6, 17],[ 5, 10],[ 7,  2],[14, 19],[ 9, 15],[16,  8],[ 4,  1],[12,  3],[ 6, 16],[ 9,  8],[ 4, 12],[ 3,  5],[19, 18],[10, 11],[13, 15],[14,  2],[ 7,  0],[ 1, 17]]
        neg_group = [[0, 3, 6], [1, 4, 7], [2, 5, 8],[ 4,  7, 16], [ 5, 10,  6], [13, 12, 11],[ 0, 15,  8],[ 9, 19, 17],[12, 2, 11],[18, 17, 6],[5, 7, 8],[10, 13, 4],[16, 15, 1],[18,  0,  2],[ 5,  3,  1],[ 7,  9, 11],[12, 14, 17],[ 8, 19,  6],[3, 7, 16],[1, 6, 12],[14, 15, 9],[2, 18, 17],[13, 11, 19]]


        def assign_preference(pre, group, value):
            for g in group:
                length = len(g)
                for temp_i in range(length):
                    a = g[temp_i]
                    for temp_j in range(length):
                        if temp_i != temp_j:
                            b = g[temp_j]
                            pre[a, b] = value
                            pre[b, a] = value
            return pre

        preference = assign_preference(preference, inf_group, np.inf)
        preference = assign_preference(preference, two_group, 2)
        preference = assign_preference(preference, fiv_group, 5)
        preference = assign_preference(preference, three_group, 3)
        preference = assign_preference(preference, four_group, 4)
        preference = assign_preference(preference, neg_group, -1)

        preference = np.array(preference) # necessary.

        """
        Simulation: 9 nodes, 9 Apps, 81 containers
        Preference:
         [[-1. -1. -1. inf  0.  2. inf  0.  0.]
         [-1.  2. -1.  0. inf  0.  0. inf  0.]
         [-1. -1. inf  5.  0. inf  0.  0. inf]
         [inf  0.  5. inf -1. -1. inf  0.  2.]
         [ 0. inf  0. -1. -1. -1.  0. inf  0.]
         [ 2.  0. inf -1. -1.  2.  5.  0. inf]
         [inf  0.  0. inf  0.  5.  2. -1. -1.]
         [ 0. inf  0.  0. inf  0. -1. inf -1.]
         [ 0.  0. inf  2.  0. inf -1. -1. -1.]]

        # -: abc, def, ghi
        # i: adg, beh, cfi
        # 2: af, di
        # 5: cd, fg
        # 0: others
        """
        return preference


    def get_min_throughput(self):
        # TODO: minimum throughput, not used yet, for fairness
        node_list = []
        for nid in range(self.NUM_NODES):
            node = Node(nid,
                        self.NODE_CAPACITY_NETWORK[nid],
                        self.NODE_CAPACITY_MEMBW[nid],
                        self.NODE_CAPACITY_CACHE[nid],
                        self)
            for aid in range(self.NUM_APPS):
                container = self.state[nid][aid]
                if container > 0:
                    app =  Application(aid,
                                      100,
                                      self.NETWORK_BW_PER_QUERY[aid],
                                      self.MEM_BW_PER_QUERY[aid],
                                      self.CACHE_PER_QUERY[aid],
                                      container)
                    node.add_application(app)
            node.calculate_new_tput()
            node_list.append(node)

        minimum_tput = 100
        for node in node_list:
            if node.minimum() < minimum_tput:
                minimum_tput = node.minimum()
        return minimum_tput

    def get_throughput_single_node(self, nid):
        """
        calculate throughput value for a single node,
        used when test a final allocation performance, we need to show the detailed info on each node
        :return: throughput on node n_id
        """

        node = Node(nid,
                    self.NODE_CAPACITY_NETWORK[nid],
                    self.NODE_CAPACITY_MEMBW[nid],
                    self.NODE_CAPACITY_CACHE[nid],
                    self)
        for aid in range(self.NUM_APPS):
            num_container = self.state[nid][aid]
            if num_container > 0:
                app = Application(aid,
                                  self.BasicThroughput[aid],
                                  self.NETWORK_BW_PER_QUERY[aid],
                                  self.MEM_BW_PER_QUERY[aid],
                                  self.CACHE_PER_QUERY[aid],
                                  num_container)
                node.add_application(app)
        node.calculate_new_tput()
        return node.total_tput(),node.tput_breakdown()

    def get_throughput_given_state(self, container_list):
        nid = 0
        node = Node(nid,
                    self.NODE_CAPACITY_NETWORK[nid],
                    self.NODE_CAPACITY_MEMBW[nid],
                    self.NODE_CAPACITY_CACHE[nid],
                    self)
        for aid in range(self.NUM_APPS):
            num_container = container_list[0,aid]
            if num_container > 0:
                app = Application(aid,
                                  self.BasicThroughput[aid],
                                  self.NETWORK_BW_PER_QUERY[aid],
                                  self.MEM_BW_PER_QUERY[aid],
                                  self.CACHE_PER_QUERY[aid],
                                  num_container)
                node.add_application(app)

        self.state[0,:] = np.array(container_list).reshape(1,-1)
        node.calculate_new_tput()


        tput_this_node = node.total_tput()
        return tput_this_node, node.tput_breakdown()