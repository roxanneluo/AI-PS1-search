# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from sets import Set
import util
from game import Actions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    visited = Set()
    # each element is <explored_idx, [[sibling_state, action_from_parent],...]> 
    # if cost is needed, its last element of each entry of the list can be the cost
    dfs_stack = util.Stack() 
    start_state = problem.getStartState()
    dfs_stack.push([-1, [[start_state,None]]])

    while not dfs_stack.isEmpty():
        prev_idx, cur_list = dfs_stack.pop()
        cur_idx = prev_idx + 1
        if cur_idx < len(cur_list) and cur_list[cur_idx][0] not in visited:
            cur_state, cur_action = cur_list[cur_idx]
            visited.add(cur_state)
            dfs_stack.push([cur_idx, cur_list])
            if problem.isGoalState(cur_state) :
                break

            next_idx, next_list = -1, []
            for next_state, next_action, act_cost in reversed(problem.getSuccessors(cur_state)):
                if next_state not in visited:
                    next_list.append([next_state, next_action])
            if len(next_list) > 0:
                dfs_stack.push([next_idx, next_list])

    actions = [] # return an empty list if goal is not found
    while not dfs_stack.isEmpty():
        idx, cur_list = dfs_stack.pop()
        action = cur_list[idx][1]
        if action is not None:
            actions.insert(0, action)
    return actions





def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    actions = []
    if problem.isGoalState(problem.getStartState()):
        return (actions)

    search_queue = util.Queue()
    visited = dict()
    init_state = problem.getStartState()
    visited[init_state] = [None, None]
    successors = problem.getSuccessors(problem.getStartState())
    for i in successors:
        search_queue.push([i,init_state])

    while not search_queue.isEmpty():
        queue_element = search_queue.pop()
        curr_state, prev_state = queue_element[0], queue_element[1]
        visited[curr_state[0]] = [prev_state,curr_state[1]]
        successors = problem.getSuccessors(curr_state[0])
        for i in successors:
            if i[0] not in visited and i not in search_queue.list:
                if problem.isGoalState(i[0]):
                    actions.insert(0,i[1])
                    j = curr_state[0]
                    while visited[j][1] is not None:
                        actions.insert(0,visited[j][1])
                        j = visited[j][0]

                search_queue.push([i,curr_state[0]])

    return actions
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    actions = []

    search_queue = util.PriorityQueue()
    visited = dict()
    init_state = problem.getStartState()
    if problem.isGoalState(problem.getStartState()):
        return (actions)
    #([(current_state,action,cost to get to this state),parent state,cost up to parent), cost up to parent)
    search_queue.push([(init_state,None,0),None,0],0)
    #visited[init_state] = [None, None]
    #successors = problem.getSuccessors(problem.getStartState())
    #for i in successors:
    #    search_queue.push([i,init_state])

    while not search_queue.isEmpty():
        queue_element = search_queue.pop()
        curr_node, prev_state, curr_cost = queue_element[0], queue_element[1], queue_element[2]
        curr_state = curr_node[0]
        #print curr_node, curr_cost
        #print curr_state, prev_state
        if curr_state in visited:
            continue
        #                     [parent,action from parent,current cost]
        visited[curr_state] = [prev_state,curr_node[1],curr_cost]
        #print visited
        if problem.isGoalState(curr_state):
            actions.insert(0,visited[curr_state][1])
            j = visited[curr_state][0]
            while visited[j][1] is not None:
                actions.insert(0,visited[j][1])
                j = visited[j][0]
            print actions
            break
        successors = problem.getSuccessors(curr_state)
        #print successors
        for i in successors:
            #print i
            new_cost = curr_cost + i[2]
            if i[0] not in visited:
                #print curr_cost, i[2]
                #new_cost = curr_cost + i[2]
                #print new_cost
                search_queue.push([i,curr_state,new_cost],new_cost)
            elif new_cost < visited[i[0]][2]:
                search_queue.push([i,curr_state,new_cost],new_cost)

    return actions
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    start_state = problem.getStartState()
    start_g_cost = 0
    start_f_cost = start_g_cost + heuristic(start_state, problem)
    q = util.PriorityQueue()
    q.push(start_state, start_f_cost)
    # another way is to put g_cost in the queue entry.
    # key: state, value: [action from parent, g_cost, f_cost]
    action_cost_dict = {start_state: [None, start_g_cost, start_f_cost]}
    visited = Set()
    
    while not q.isEmpty():
        cur_state = q.pop()
        if cur_state in visited:
            continue
        if problem.isGoalState(cur_state):
            goal_state = cur_state
            break
        visited.add(cur_state)
        cur_g_cost = action_cost_dict[cur_state][1]
        for next_state, action, act_cost in problem.getSuccessors(cur_state):
            if next_state not in visited:
                g_cost = cur_g_cost + act_cost 
                f_cost = g_cost + heuristic(next_state, problem)
                q.push(next_state, f_cost)
                if next_state not in action_cost_dict or \
                        f_cost < action_cost_dict[next_state][2]:
                    action_cost_dict[next_state] = [action, g_cost, f_cost]

    
    actions = []
    cur_state = goal_state
    while cur_state != start_state:
        action = action_cost_dict[cur_state][0]
        actions.insert(0, action)
        backward_action = Actions.reverseDirection(action)
        cur_state = Actions.getSuccessor(cur_state, backward_action)
    return actions





# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
