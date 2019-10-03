# necessary packages
import time
import numpy as np
from osbrain import run_nameserver
from osbrain import run_agent
from osbrain import Agent

# here below we define the payoffs for player 1 and 2 (prisoners game)
game_p1 = np.array([[-1, -10], [0, -9]])
game_p2 = np.array([[-1, 0], [-10, -9]])


class Alice(Agent):
    def on_init(self):
        """
        Initialize Alice as an agent with two bindings
        :return:
        """
        self.bind('PUSH', alias='main')
        self.bind('PUSH', alias='strategy')

    def hello(self, name):
        """The hello message from alice"""
        self.send('main', 'Hello to %s from Alice!' % name)

    def custom_log(self, message):
        """
        Standard log that prints a message
        :param message: message that needs to be stated
        :return:
        """
        self.log_info('Received a message: %s' % message)

    def nash_strategy(self):
        """
        Function that sets the strategy to a nash strategy
        :return:
        """
        self.strategy = 1

    def random_strategy(self):
        """
        Play a random strategy if this one is chosen
        :return:
        """
        upper_row = np.random.uniform() # draw a random number from uniform distribution

        # if the number is higher than 0.5, we play the upper row, else we play the bottom row
        if upper_row > 0.5:
            self.strategy = 0
        else:
            self.strategy = 1

    def cooperate_strategy(self):
        """
        Set the strategy to be played to cooperate
        :return:
        """
        self.strategy = 0

    def receive_strategy(self, strategy):
        """
        Function that receives the strategy of the other player and stores it
        :param strategy: opponents strategy
        :return:
        """
        self.opponent_strategy = strategy

    def tit_for_tat_strategy(self, round_number):
        """
        Play a tit for tat strategy
        :param round_number: The current round we are in with the game
        :return:
        """

        # if we are in the first round, we will always cooperate
        if round_number == 1:
            self.strategy = 0
        else:
            # if not, we base our strategy on the opponents last chosen strategy
            if self.opponent_strategy == 1:
                self.strategy = 1
            else:
                self.strategy = 0

    def play_strategy(self):
        """
        This function 'plays' the strategy by sending a string to the mediator in the game
        :return:
        """

        # message that converges the strategy into a string
        row_number = str(int(self.strategy+1)) + str("th")

        # sending the strategy to the mediator
        self.send('main', 'Alice will play the %s row' % row_number)
        self.send('strategy', self.strategy)

        # storing the strategy as last played
        self.last_strategy = self.strategy


class Bob(Agent):
    def on_init(self):
        """
        Initialize an agent (Bob) which commmunicates over two push messages
        :return:
        """
        self.bind('PUSH', alias='main')
        self.bind('PUSH', alias='strategy')

    def hello(self, name):
        """
        Hello from Bob!
        :param name: Say the name of the other agent
        :return:
        """
        self.send('main', 'Hello to %s from Bob!' % name)

    def custom_log(self, message):
        """
        Standard message viewer
        :param message: the input message that needs to be stated
        :return:
        """
        self.log_info('Received: %s' % message)

    def nash_strategy(self):
        """
        Set the strategy of Bob to Nash
        :return:
        """
        self.strategy = 1

    def random_strategy(self):
        """
        Bob will play a random strategy
        :return:
        """
        left_column = np.random.uniform()   # draw a random number from uniform distribution

        # if the value drawn is higher than 0.5, we pick the left column, else the right one
        if left_column > 0.5:
            self.strategy = 0
        else:
            self.strategy = 1

    def cooperate_strategy(self):
        """
        This function makes Bob cooperate in the game
        :return:
        """
        self.strategy = 0

    def tit_for_tat_strategy(self, round_number):
        """
        Play a tit for tat strategy
        :param round_number: The current round we are in with the game
        :return:
        """

        # if we are in the first round, we will always cooperate
        if round_number == 1:
            self.strategy = 0
        else:
            # if not, we base our strategy on the opponents last chosen strategy
            if self.opponent_strategy == 1:
                self.strategy = 1
            else:
                self.strategy = 0

    def receive_strategy(self, strategy):
        """
        This function stores the opponent strategy
        :param strategy: strategy played by the opponent
        :return:
        """
        self.opponent_strategy = strategy

    def play_strategy(self):
        """
        This function actually plays the strategy
        :return:
        """
        column_number = str(int(self.strategy+1)) + str("th")   # convert strategy into a string

        # send the message to the mediator
        self.send('main', 'Bob will play the %s column' % column_number)
        self.send('strategy', self.strategy)

        # store the last strategy as being played
        self.last_strategy = self.strategy


class Mediator(Agent):
    def on_init(self):
        """
        Initialize a mediator agent which binds with Alice and Bob
        :return:
        """
        self.bind('PUSH', alias='Alice_general')
        self.bind('PUSH', alias='Bob_general')

        self.bind('PUSH', alias='Alice_strategy_opponent')
        self.bind('PUSH', alias='Bob_strategy_opponent')

        # set cumulative payoffs equal to zero
        self.cumulative_payoffs_p1 = 0
        self.cumulative_payoffs_p2 = 0

    def custom_log(self, message):
        """
        Standard message log function
        :param message: The message that is being received
        :return:
        """
        self.log_info('Received: %s' % message)

    def receive_strategy_alice(self, strategy_alice):
        """
        Function that receives and stores the strategy played by Alice
        :param strategy_alice: strategy played by alice
        :return:
        """
        self.alice_played = strategy_alice

    def receive_strategy_bob(self, strategy_bob):
        """
        Function that receives and stores the strategy played by Bob
        :param strategy_bob: strategy played by Bob
        :return:
        """
        self.bob_played = strategy_bob

    def ask_action(self):
        """
        Function that asks Alice and Bob to take an action in the game
        :return:
        """
        self.send('Alice_general', 'Please state your action Alice')
        self.send('Bob_general', 'Please state your action Bob')

    def compute_result(self, round_number):
        """
        This function computes the result of the stage of the game after Alice and Bob have played a certain action
        :param round_number: The round of the game we are in
        :return:
        """
        # here we get the result of the game for both players
        result_p1 = game_p1[self.alice_played, self.bob_played]
        result_p2 = game_p2[self.alice_played, self.bob_played]

        # we send a message to both with the result
        self.send('Alice_general', 'You have received a payoff last round of: %s' % str(result_p1))
        self.send('Bob_general', 'You have received a payoff last round of: %s' % str(result_p2))

        # compute the cumulative payoff
        self.cumulative_payoffs_p1 += result_p1
        self.cumulative_payoffs_p2 += result_p2

        # compute the average result for both
        self.average_result_p1 = self.cumulative_payoffs_p1/round_number
        self.average_result_p2 = self.cumulative_payoffs_p2/round_number

        time.sleep(0.5)

        # send a message with the average reward
        self.send('Alice_general', 'You have an average reward of: %s' % str(self.average_result_p1))
        self.send('Bob_general', 'You have an average reward of: %s' % str(self.average_result_p2))

    def send_strategies(self):
        """
        Function that informs both players of the strategies of the opponent
        :return:
        """
        self.send('Alice_strategy_opponent', self.bob_played)
        self.send('Bob_strategy_opponent', self.alice_played)

if __name__ == '__main__':
    # this is the main game loop

    # we set up a MAS with Alice, Bob and a Mediator
    ns = run_nameserver()
    alice = run_agent('Alice', base=Alice)
    bob = run_agent('Bob', base=Bob)
    mediator = run_agent('Mediator', base=Mediator)

    #  Here below we do eight connection functions in order to stream messages
    alice.connect(mediator.addr('Alice_general'), handler='custom_log')
    bob.connect(mediator.addr('Bob_general'), handler='custom_log')

    alice.connect(mediator.addr('Alice_strategy_opponent'), handler='receive_strategy')
    bob.connect(mediator.addr('Bob_strategy_opponent'), handler='receive_strategy')

    mediator.connect(alice.addr('main'), handler='custom_log')
    mediator.connect(bob.addr('main'), handler='custom_log')

    mediator.connect(alice.addr('strategy'), handler='receive_strategy_alice')
    mediator.connect(bob.addr('strategy'), handler='receive_strategy_bob')

    print("We start with introducing both agents via the mediator")
    alice.hello('Bob')
    time.sleep(2)
    bob.hello('Alice')
    time.sleep(2)

    print("")

    # Here below we have the game loop, we loop from 0 to a total number of game rounds
    for i in np.arange(0, 100):
        print("")
        print("NEW ROUND, ROUND NUMBER:", i+1)
        mediator.ask_action()
        time.sleep(0.5)
        alice.random_strategy()
        bob.tit_for_tat_strategy(i+1)
        time.sleep(0.5)
        alice.play_strategy()
        bob.play_strategy()
        time.sleep(0.5)
        mediator.compute_result(i+1)
        mediator.send_strategies()

    print("")
    ns.shutdown()