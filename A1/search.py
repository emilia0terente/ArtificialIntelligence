# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
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

import util
import time
import copy

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

def transformPathToDirections(path):
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
    open_stack = util.Stack()
    start_state = problem.getStartState()
    open_stack.push((start_state, [], 0))  # (state, actions, cost)

    visited = set()

    while not open_stack.isEmpty():
        state, actions, cost = open_stack.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)

            successors = problem.getSuccessors(state)
            for successor_state, action, step_cost in successors:
                new_actions = actions + [action]
                new_cost = cost + step_cost
                open_stack.push((successor_state, new_actions, new_cost))

    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    open_queue = util.Queue()
    open_queue.push([(problem.getStartState(), "")])
    visited = set()

    while not open_queue.isEmpty():
        path = open_queue.pop()
        state, action = path[-1]

        if problem.isGoalState(state):
            return [act for _, act in path[1:]]

        if state not in visited:
            visited.add(state)
            successors = problem.getSuccessors(state)
            for succ_state, succ_action, _ in successors:
                new_path = list(path)
                new_path.append((succ_state, succ_action))
                open_queue.push(new_path)

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue

    front = PriorityQueue()
    start = (problem.getStartState(), [], 0)
    front.push(start, 0)
    visited_state = set()

    while not front.isEmpty():
        state, actions, cost = front.pop()

        if state in visited_state:
            continue

        visited_state.add(state)

        if problem.isGoalState(state):
            return actions

        for succ_state, succ_action, succ_cost in problem.getSuccessors(state):
            if succ_state not in visited_state:
                new_actions = actions + [succ_action]
                new_cost = cost + succ_cost
                front.push((succ_state, new_actions, new_cost), new_cost)

    return []



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

import util

# DijkstraSearch
def dijkstraSearch(problem):
    open_ds = []  #list for priority queue
    start_state = problem.getStartState()
    open_ds.append((start_state, [], 0))

    visited_state = {}

    while open_ds:
        state, actions, cost = open_ds.pop(0)

        if state in visited_state:
            continue

        visited_state[state] = (actions, cost)

        if problem.isGoalState(state):
            return actions

        successors = problem.getSuccessors(state)
        for next_state, next_action, next_cost in successors:
            if next_state not in visited_state:
                new_actions = actions + [next_action]
                new_cost = cost + next_cost
                open_ds.append((next_state, new_actions, new_cost))
                open_ds.sort(key=lambda x: x[2])

    return []

def aStarSearch(problem, heuristic=nullHeuristic):
    front = util.PriorityQueue()
    start_state = problem.getStartState()
    start = (start_state, [], 0)
    front.push(start, 0)
    visited_state = {start_state: 0}
    
    while not front.isEmpty():
        current, actions, cost = front.pop()
        
        if problem.isGoalState(current):
            return actions
        
        successors = problem.getSuccessors(current)
        
        for next_state, action, step_cost in successors:
            gn_succ = cost + step_cost
            hn_succ = heuristic(next_state, problem) if heuristic else 0
            fn_succ = gn_succ + hn_succ
            
            if next_state not in visited_state or gn_succ < visited_state[next_state]:
                visited_state[next_state] = gn_succ
                new_actions = actions + [action]
                new_node = (next_state, new_actions, gn_succ)
                front.push(new_node, fn_succ)
    
    return []

#A*WeightedSearch
def aStarWeightedSearch(problem, heuristic, weight=10):
    front = util.PriorityQueue()
    start_state = problem.getStartState()
    start = (start_state, [], 0)
    front.push(start, 0)
    visited_state = {start_state: 0}
    
    while not front.isEmpty():
        current, actions, cost = front.pop()
        
        if problem.isGoalState(current):
            return actions
        
        successors = problem.getSuccessors(current)
        
        for next_state, action, step_cost in successors:
            gn_succ = cost + step_cost
            hn_succ = heuristic(next_state, problem) * weight if heuristic else 0
            fn_succ = (gn_succ + hn_succ) 
            
            if next_state not in visited_state or gn_succ < visited_state[next_state]:
                visited_state[next_state] = gn_succ
                new_actions = actions + [action]
                new_node = (next_state, new_actions, gn_succ)
                front.push(new_node, fn_succ)
    
    return []


def incrementalHeuristicSearch(problem, heuristic=nullHeuristic, timeLimit=10):
    closed = set()
    startState = problem.getStartState()
    startNode = (startState, [], 0, 0)
    priorityQueue = util.PriorityQueue()
    priorityQueue.push(startNode, startNode[3])
    start_time = time.time()

    while not priorityQueue.isEmpty():
        current, actions, cost, _ = priorityQueue.pop()
        if problem.isGoalState(current):
            return actions

        if current not in closed:
            closed.add(current)
            successors = problem.getSuccessors(current)
            for successor, action, stepCost in successors:
                newActions = actions + [action]
                newCost = cost + stepCost
                elapsed_time = time.time() - start_time
                remainingTime = timeLimit - elapsed_time
                newHeuristic = 0
                if heuristic is not None:
                    newHeuristic = heuristic(successor, problem)
                newPriority = newCost + newHeuristic
                priorityQueue.push((successor, newActions, newCost, newHeuristic), newPriority)

    return [] 


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
astar = aStarSearch
astarw = aStarWeightedSearch
djk = dijkstraSearch
inc = incrementalHeuristicSearch


