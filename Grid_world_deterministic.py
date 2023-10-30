import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# global variables
BOARD_ROWS = 10
BOARD_COLS = 10
WIN_STATE = (8, 8)
#LOSE_STATE = [(2, 4),(4, 2)]
LOSE_STATE = [(9, 9), (7, 7), (0, 4), (4, 0)]
START = (1, 1)
DETERMINISTIC = True
last_reward = 0

STATE_MAT = np.zeros([BOARD_COLS, BOARD_ROWS])
cmap = colors.ListedColormap(['white', 'black', 'green', 'red'])
bounds = [0, 1, 10, 11, 12]
norm = colors.BoundaryNorm(bounds, cmap.N)

class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        for f in LOSE_STATE:
            print(f)
            print(self.state)
            if self.state == WIN_STATE:
                return 1
            elif self.state == f:
                return -1
            

    def isEndFunc(self):
        for f in LOSE_STATE:
            if (self.state == WIN_STATE) or (self.state == f):
                self.isEnd = True

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS -1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS -1)):
                    if nxtState != (1, 1):
                        return nxtState
            return self.state

    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')
        

    


# Agent of player

class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "right", "left"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3 #exploration rate
        self.reward = 0

        #Draw the agent
        plt.ion()
        self.fig, self.ax = plt.subplots()
        #Draw gridlines
        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        self.ax.set_xticks(np.arange(-.5, BOARD_COLS, 1))
        self.ax.set_yticks(np.arange(-.5, BOARD_ROWS, 1))
        #self.ax.xaxis.set_ticklabels('none')
        #self.ax.yaxis.set_ticklabels('none')
        #show initial state
        self.im = self.ax.imshow(STATE_MAT, cmap=cmap, norm=norm)

        # initial state reward
        self.state_values = {}
        
        # initial value of 0 for stupid start
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                #self.state_values[(i, j)] = 0  # set initial value to 0
                self.state_values[(i, j)] = -np.abs((WIN_STATE[0] - i) + (WIN_STATE[1] - j))

    def draw_state(self):
        STATE_MAT[WIN_STATE] = 10
        for f in LOSE_STATE:
            STATE_MAT[f] = 11
        STATE_MAT[self.State.state] = 1
        self.im.set_data(STATE_MAT)
        self.im.autoscale()
        plt.pause(0.1)

        STATE_MAT[self.State.state] = 0

    def chooseAction(self):
        # choose action with most expected value
        #mx_nxt_reward = 0
        mx_nxt_reward = self.state_values[self.State.state]
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                self.reward = reward
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nxtPosition(action))
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                print("nxt state", self.State.state)
                print("Last given reward", self.reward)
                print("---------------------")
            self.draw_state()
           

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


if __name__ == "__main__":
    ag = Agent()
    #print(ag.showValues())
    ag.play(150)
    print(ag.showValues())
    
    plt.show()