import time
import numpy as np
from osbrain import run_nameserver
from osbrain import run_agent
from osbrain import Agent

game_p1 = np.array([[-1, 0], [0, -9]])
game_p2 = np.array([[-1, -10], [0, -9]])


class Alice(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def hello(self, name):
        self.send('main', 'Hello to %s from Alice!' % name)

    def custom_log(self, message):
        self.log_info('Received a message: %s' % message)

    def nash_strategy(self):
        self.strategy = 1

    def random_strategy(self):
        upper_row = np.random.uniform()

        if upper_row > 0.5:
            self.strategy = 0
        else:
            self.strategy = 1

        print(self.strategy)

    def cooperate_strategy(self):
        self.strategy = 0

    def tit_for_tat_strategy(self):
        print("Placeholder")


class Bob(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def hello(self, name):
        self.send('main', 'Hello to %s from Bob!' % name)

    def custom_log(self, message):
        self.log_info('Received: %s' % message)

    def nash_strategy(self):
        self.strategy = 1

    def random_strategy(self):
        upper_row = np.random.uniform()

        if upper_row > 0.5:
            self.strategy = 0
        else:
            self.strategy = 1

        print(self.strategy)

    def cooperate_strategy(self):
        self.strategy = 0

    def tit_for_tat_strategy(self):
        print("Placeholder")

    def play_strategy(self):
        column_number = str(int(self.strategy+1)) + str("th")

        self.send('main', 'Bob will play the %s row' % column_number)


class Mediator(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def custom_log(self, message):
        self.log_info('Received: %s' % message)

    def ask_strategy(self):
        print("Placeholder")

    def


if __name__ == '__main__':

    ns = run_nameserver()
    alice = run_agent('Alice', base=Alice)
    bob = run_agent('Bob', base=Bob)
    mediator = run_agent('Mediator', base=Mediator)

    mediator.connect(alice.addr('main'), handler='custom_log')
    mediator.connect(bob.addr('main'), handler='custom_log')



    alice.hello('Bob')
    time.sleep(2)
    bob.hello('Alice')
    time.sleep(2)

    # for i in np.arange(0, 20):
    #     alice.random_strategy()
    #
    # alice.define_individual_rational_strategy()
    # # alice.play_strategy()
    #
    # bob.define_individual_rational_strategy()
    # bob.play_strategy()

    ns.shutdown()