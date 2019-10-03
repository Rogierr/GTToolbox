# here below, we import all the packages necessary
import time
import numpy as np
from osbrain import run_nameserver
from osbrain import run_agent
from osbrain import Agent

game = np.array([[7, 3], [2, 2]])   # the game, defined as a numpy array


class Alice(Agent):
    def on_init(self):
        """
        Initialize the agent including two messages that the agent is able to send
        Main and result messages are pushed.
        :return: Initialized agent
        """
        self.bind('PUSH', alias='main')
        self.bind('PUSH', alias='result')

    def hello(self, name):
        """
        A function that sends a hello message over the main push message.
        :param name: The name of the person that is greeted
        :return: Send a hello message
        """
        self.send('main', 'Hello, %s!' % name)

    def custom_log(self, message):
        self.log_info('Received a message: %s' % message)

    def store_result(self, result):
        """
        ALice stores the result of the other player
        :param result: The result of the other player
        :return: Empty
        """
        self.strategy_other = result

    def define_individual_rational_strategy(self):
        """
        This function defines the individual rational strategy for a player (the Nash strategy). It is only able to
        compute pure Nash equilibria
        :return: Does not return anything, but stores the pure Nash strategy for player 1
        """
        best_resp_p1 = game.argmax(0)  # look for maximal values in columns, return index value
        best_resp_p2 = game.argmin(1)  # look for minimal values in the rows, return index value

        best_resp_ind_p1 = []  # empty placeholders to store index values
        best_resp_ind_p2 = []

        # here below we convert the best responses to a readable array of index values
        for i in range(0, best_resp_p1.shape[0]):
            best_resp_ind_p1.append((best_resp_p1[i], i))

        for j in range(0, best_resp_p2.shape[0]):
            best_resp_ind_p2.append((j, best_resp_p2[j]))

        nash_eq_exists = False  # assume no Pure Nash exist first (for use if no pure is found)

        # check if a pure Nash Equilibrium exists by looping over index values
        for i in range(0, len(best_resp_ind_p1)):
            for j in range(0, len(best_resp_ind_p2)):
                if best_resp_ind_p1[i] == best_resp_ind_p2[j]:
                    nash_eq = best_resp_ind_p1[i]
                    nash_eq_exists = True  # pure Nash has been found, so change it's value to true

        if nash_eq_exists:  # if a pure Nash Equilibrium exists, do the following
            index_nash_eq = nash_eq  # get value of the Nash Equilibrium and print it on the screen

            value_nash = game[index_nash_eq]  # store the value of the Nash
            equal = np.equal(value_nash, game)  # look for equal values which are Nash candidates

            self.strategy = nash_eq[0]

            # this code here below is for if there are multiple pure Nash candidates
            if np.sum(equal) > 1:  # if multiple pure Nash candidates exist
                indexvalues = np.argwhere(equal)  # lookup all locations with pure Nash value
                length_index = round(indexvalues.size / 2)  # make a value for the number of found candidates

                for k in range(0, length_index):  # loop over all possible candidates
                    index_row_column = indexvalues[k]  # store index value of the candidate

                    value_found = game[index_row_column[0], index_row_column[1]]  # store found value
                    row_min = np.amin(game[index_row_column[0]],
                                      axis=1)  # look for minimal value in the row index
                    column_max = np.amax(game[:, index_row_column[1]],
                                         axis=0)  # look for maximal value in the column index

                    if row_min == value_found:  # only execute if value found is indeed row minimum
                        if column_max == value_found:  # and last but not least execute if value found is column maximum
                            self.strategy = index_row_column[0]

    def print_strategy(self):
        """
        This functions prints the strategy as a message within the MAS
        :return: Empty
        """
        row_number = str(int(self.strategy+1)) + str("th")  # set the rom to be played as a string message

        self.send('main', 'Alice will play the %s row' % row_number)    # send the message

    def play_strategy(self):
        row_number = self.strategy

        self.send('result', row_number)

    def print_result(self):
        self.result = game[self.strategy, self.strategy_other]

        self.log_info("I received a payoff of %s" % str(self.result))

class Bob(Agent):
    def on_init(self):
        """
        Initialize the agent including two messages that the agent is able to send
        Main and result messages are pushed.
        :return: Initialized agent
        """
        self.bind('PUSH', alias='main')
        self.bind('PUSH', alias='result')

    def hello(self, name):
        """
        A function that sends a hello message over the main push message.
        :param name: The name of the person that is greeted
        :return: Send a hello message
        """
        self.send('main', 'Hello, %s!' % name)

    def custom_log(self, message):
        """
        The log that prints the message to the screen and states that something has been received.
        :param message:
        :return:
        """
        self.log_info('Received: %s' % message)

    def store_result(self, result):
        self.strategy_other = result

    def define_individual_rational_strategy(self):
        best_resp_p1 = game.argmax(0)  # look for maximal values in columns, return index value
        best_resp_p2 = game.argmin(1)  # look for minimal values in the rows, return index value

        best_resp_ind_p1 = []  # empty placeholders to store index values
        best_resp_ind_p2 = []

        # here below we convert the best responses to a readable array of index values
        for i in range(0, best_resp_p1.shape[0]):
            best_resp_ind_p1.append((best_resp_p1[i], i))

        for j in range(0, best_resp_p2.shape[0]):
            best_resp_ind_p2.append((j, best_resp_p2[j]))

        nash_eq_exists = False  # assume no Pure Nash exist first (for use if no pure is found)

        # check if a pure Nash Equilibrium exists by looping over index values
        for i in range(0, len(best_resp_ind_p1)):
            for j in range(0, len(best_resp_ind_p2)):
                if best_resp_ind_p1[i] == best_resp_ind_p2[j]:
                    nash_eq = best_resp_ind_p1[i]
                    nash_eq_exists = True  # pure Nash has been found, so change it's value to true

        if nash_eq_exists:  # if a pure Nash Equilibrium exists, do the following
            index_nash_eq = nash_eq  # get value of the Nash Equilibrium and print it on the screen

            value_nash = game[index_nash_eq]  # store the value of the Nash
            equal = np.equal(value_nash, game)  # look for equal values which are Nash candidates

            self.strategy = nash_eq[1]

            # this code here below is for if there are multiple pure Nash candidates
            if np.sum(equal) > 1:  # if multiple pure Nash candidates exist
                indexvalues = np.argwhere(equal)  # lookup all locations with pure Nash value
                length_index = round(indexvalues.size / 2)  # make a value for the number of found candidates

                for k in range(0, length_index):  # loop over all possible candidates
                    index_row_column = indexvalues[k]  # store index value of the candidate

                    value_found = game[index_row_column[0], index_row_column[1]]  # store found value
                    row_min = np.amin(game[index_row_column[0]],
                                      axis=1)  # look for minimal value in the row index
                    column_max = np.amax(game[:, index_row_column[1]],
                                         axis=0)  # look for maximal value in the column index

                    if row_min == value_found:  # only execute if value found is indeed row minimum
                        if column_max == value_found:  # and last but not least execute if value found is column maximum
                            self.strategy = index_row_column[0]

    def print_strategy(self):
        """
        Print the strategy of the player Bob, sends this as a class linked to the player
        :return: Nothing
        """
        column_number = str(int(self.strategy+1)) + str("th")

        self.send('main', 'Bob will play the %s column' % column_number)

    def play_strategy(self):
        column_number = self.strategy

        self.send('result', column_number)

    def print_result(self):
        self.result = game[self.strategy_other, self.strategy]

        self.log_info("I received a payoff of %s" % str(-self.result))

# here below is the execution of the main function
if __name__ == '__main__':

    # we create a server with two agents
    ns = run_nameserver()
    alice = run_agent('Alice', base=Alice)
    bob = run_agent('Bob', base=Bob)

    # connect the agents to each other for regular messages
    alice.connect(bob.addr('main'), handler='custom_log')
    bob.connect(alice.addr('main'), handler='custom_log')

    # connect the agents in order to send the strategies they play
    alice.connect(bob.addr('result'), handler='store_result')
    bob.connect(alice.addr('result'), handler='store_result')

    # in the below loop we just say hi :)
    alice.hello('Bob')
    time.sleep(2)
    bob.hello('Alice')
    time.sleep(2)

    # define the nash strategy for alice and play it
    alice.define_individual_rational_strategy()
    alice.print_strategy()
    alice.play_strategy()

    # define the nash strategy for bob and play it
    bob.define_individual_rational_strategy()
    bob.print_strategy()
    bob.play_strategy()


    # print the results for alice and bob
    alice.print_result()
    bob.print_result()

    ns.shutdown()   # shutdown the whole MAS