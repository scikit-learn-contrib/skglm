import numpy as np
import bisect
from operator import itemgetter
from scipy.sparse import identity


def get_events_from(filepath):
    """ get the events list from a file"""
    event_list = []
    nodelist = set()
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                pass
            line = line.strip().split()
            u, v, t = int(line[0]), int(line[1]), float(line[2])
            event_list.append((u, v, t))
            nodelist.add(u)
            nodelist.add(v)
    return event_list, list(nodelist)


def write_events(filepath, events):
    """write the list of events in filepath"""
    with open(filepath, 'w') as f:
        for event in events:
            string = str(event[0]) + ' ' + str(event[1]) + ' ' + str(event[2]) + '\n'
            f.write(string)
    return 0


class TemporalNetwork:
    def __init__(self, events):
        self.__events = []
        for i, j, t in events:
            if i != j:  # no self-events
                self.__events.append((*sorted([i, j]), t))
        self.__events.sort()

        self.__incident_events = {}
        for i, j, t in self.__events:
            if i not in self.__incident_events:  # may be faster to do try: append except: initialize
                self.__incident_events[i] = []
            self.__incident_events[i].append((t, j))

            if j not in self.__incident_events:
                self.__incident_events[j] = []
            self.__incident_events[j].append((t, i))
        for node in self.__incident_events:
            self.__incident_events[node].sort()  # sorted list of successors of node
        print("Number of nodes: ", len(self.__incident_events))

    def events(self):
        return list(self.__events)

    def nodes(self):
        return list(self.__incident_events)

    def successors_vert(self, node, time, dt, first=False):
        if node not in self.__incident_events:
            return []
        inc = self.__incident_events[node]
        # list of all nodes in temporal network that interact with 'node' and the time at which they interact
        # [(time, node), (time, node), ...]

        res = []

        pos = bisect.bisect_left(inc, (time, -np.inf))
        # find the insertion point of (time, -np.inf) in inc such that order is conserved
        # find the location of the event happening at time "time"
        # for all event implying "node" after "time" and before "dt"
        while pos < len(inc) and inc[pos][0] - time < dt:
            # print(pos, res)
            if first and res and res[-1][0] < inc[pos][0]:
                # returns the first adjacent node and the time of interaction (relative to "node")
                return res
            if inc[pos][0] > time:  # dt-adjacent
                res.append(inc[pos])
            pos += 1
        # returns the list of adjacent nodes and their interaction times
        return res

    def successors(self, event, dt, first=False):
        i, j, t = event
        res = set()

        for ot, o in self.successors_vert(i, t, dt, first=first):
            res.add((*sorted([i, o]), ot))
        for ot, o in self.successors_vert(j, t, dt, first=first):
            res.add((*sorted([j, o]), ot))
        # if first=True, returns the list of the first two adjacent events after "event"
        return res


def poisson_temporal_network(static_network, mean_iet, max_t, seed=None):
    gen = np.random.default_rng(seed)
    events = []

    for i, j in static_network.edges():
        t = gen.exponential(mean_iet)  # residual
        while t <= max_t:
            events.append((i, j, t))
            t += gen.exponential(mean_iet)
    print("Number of events: ", len(events))
    events = sorted(events, key=itemgetter(2))
    return events


# @profile
def SI_process_parallel(events, link_table, output_dimension):
    C = np.eye(output_dimension, dtype=bool)
    for e in events:
        u, v, t = e
        u_prime, v_prime = link_table[u], link_table[v]
        union = C[u_prime] | C[v_prime]
        C[u_prime] = union
        C[v_prime] = union
    return C


# @profile
def SI_process_parallel_sparse(events, link_table, output_dimension):
    C = identity(output_dimension, dtype=bool, format='lil')
    for e in events:
        u, v, t = e
        u_prime, v_prime = link_table[u], link_table[v]
        union = C[u_prime] + C[v_prime]
        C[u_prime] = union
        C[v_prime] = union
    return C


def hashing_events(events, hashtable, dimension):
    """ relabel the events' nodes"""
    new_events = []
    if isinstance(hashtable, list):
        for u, v, t in events:
            new_event = hashtable[u], hashtable[v], t
            new_events.append(new_event)
    else:  # if hashtable is of class HashTable
        for u, v, t in events:
            new_event = hashtable.hash32(
                u) % dimension, hashtable.hash32(v) % dimension, t
            new_events.append(new_event)
    return new_events


def link_table(hashtable, input_dimension, output_dimension):
    """ return the table between input and output (labels and hashed labels)"""
    tables = [-1 for j in range(input_dimension)]
    for j in range(input_dimension):
        tables[j] = hashtable.hash32(j) % output_dimension
    return tables
