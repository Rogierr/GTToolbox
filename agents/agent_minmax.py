import time
import numpy as np
# import random
from osbrain import run_nameserver
from osbrain import run_agent
from osbrain import Agent

game = np.array([[7, 3], [2, 2]])


class Alice(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def hello(self, name):
        self.send('main', 'Hello, %s!' % name)

    def custom_log(self, message):
        self.log_info('Received a message: %s' % message)

    def define_individual_rational_strategy(self):
        best_resp_p1 = game.argmax(0)  # look for maximal values in columns, return index value
        best_resp_p2 = game.argmin(1)  # look for minimal values in the rows, return index value

        print(best_resp_p1)

        best_resp_ind_p1 = []  # empty placeholders to store index values
        best_resp_ind_p2 = []

        # here below we convert the best responses to a readable array of index values
        for i in range(0, best_resp_p1.shape[1]):
            best_resp_ind_p1.append((best_resp_p1[0, i], i))

        for j in range(0, best_resp_p2.shape[0]):
            best_resp_ind_p2.append((j, best_resp_p2[j, 0]))

        NashEq_exists = False  # assume no Pure Nash exist first (for use if no pure is found)

        # check if a pure Nash Equilibrium exists by looping over index values
        for i in range(0, len(best_resp_ind_p1)):
            for j in range(0, len(best_resp_ind_p2)):
                if (best_resp_ind_p1[i] == best_resp_ind_p2[j]) == True:
                    NashEq = best_resp_ind_p1[i]
                    NashEq_exists = True  # pure Nash has been found, so change it's value to true

        if (NashEq_exists == True):  # if a pure Nash Equilibrium exists, do the following
            print("Pure Nash Equilibrium has been found")
            index_NashEq = NashEq  # get value of the Nash Equilibrium and print it on the screen
            print("Nash Equilibrium has value:", game[index_NashEq])

            valueNash = game[index_NashEq]  # store the value of the Nash
            equal = np.equal(valueNash, game)  # look for equal values which are Nash candidates
            print("")

            if np.sum(equal) > 1:  # if multiple pure Nash candidates exist
                print("Multiple Nash Equilibria candidates found")
                indexvalues = np.argwhere(equal)  # lookup all locations with pure Nash value
                length_index = round(indexvalues.size / 2)  # make a value for the number of found candidates

                counter = 0  # count number of nashes

                for k in range(0, length_index):  # loop over all possible candidates
                    index_row_column = indexvalues[k]  # store index value of the candidate

                    value_found = game[index_row_column[0], index_row_column[1]]  # store found value
                    row_min = np.amin(game[index_row_column[0]],
                                      axis=1)  # look for minimal value in the row index
                    column_max = np.amax(game[:, index_row_column[1]],
                                         axis=0)  # look for maximal value in the column index

                    if row_min == value_found:  # only execute if value found is indeed row minimum

                        if column_max == value_found:  # and last but not least execute if value found is column maximum
                            print("Player 1 can play row:",
                                  index_row_column[0])  # print the strategy pair if it is a pure Nash
                            print("Player 2 can play column:", index_row_column[1])
                            print("")
                            counter = counter + 1

                if counter == 1:  # if only one candidate is a Pure Nash, print this
                    print("Only one pure Nash Equilibrium found within the candidates")
                    print("")

                else:  # else print the total number of Pure Nash
                    print("Number of pure Nash Equilibria found:", counter)
                    print("")

            else:  # if only one pure Nash exists, print this on the screen
                print("Only one pure Nash Equilibrium found")
                print("Player 1 plays row", index_NashEq[0])
                print("Player 2 plays column", index_NashEq[1])
                print("")

        else:  # if no pure Nash exist, print this on the screen
            print("No Pure Nash Equilibrium has been found")
            print("")

    def play_strategy(self):
        row_number = str(int(self.strategy+1)) + str("th")

        self.send('main', 'Alice will play the %s row' % row_number)

class Bob(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def hello(self, name):
        self.send('main', 'Hello, %s!' % name)

    def custom_log(self, message):
        self.log_info('Received: %s' % message)

    def define_individual_rational_strategy(self):
        pure_strategy_exist = False
        save_indices = np.zeros(game.shape[0])

        for i in np.arange(0, game.shape[0]):
            print("first", game.argmin(1))
            print(np.argmin(game[i, :]))
            save_indices[i] = np.argmin(game[i, :])

        if np.max(save_indices) == np.min(save_indices):
            pure_strategy_exist = True
            self.strategy = save_indices[0]

        if not pure_strategy_exist:
            print("This code is emitted from the design (see also the other computation functions for inspiration)")

    def play_strategy(self):
        column_number = str(int(self.strategy+1)) + str("th")

        self.send('main', 'Bob will play the %s row' % column_number)

if __name__ == '__main__':

    ns = run_nameserver()
    alice = run_agent('Alice', base=Alice)
    bob = run_agent('Bob', base=Bob)

    alice.connect(bob.addr('main'), handler='custom_log')
    bob.connect(alice.addr('main'), handler='custom_log')

    alice.hello('Bob')
    time.sleep(2)
    bob.hello('Alice')
    time.sleep(2)

    alice.define_individual_rational_strategy()
    # alice.play_strategy()

    bob.define_individual_rational_strategy()
    # bob.play_strategy()

    ns.shutdown()