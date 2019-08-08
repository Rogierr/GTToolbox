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
        self.send('main', 'Hello to %s from Alice!' % name)

    def custom_log(self, message):
        self.log_info('Received a message: %s' % message)

    def nash_strategy(self):
        print("placeholder")

    def play_strategy(self):
        row_number = str(int(self.strategy+1)) + str("th")

        self.send('main', 'Alice will play the %s row' % row_number)

class Bob(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def hello(self, name):
        self.send('main', 'Hello to %s from Bob!' % name)

    def custom_log(self, message):
        self.log_info('Received: %s' % message)

    def play_strategy(self):
        column_number = str(int(self.strategy+1)) + str("th")

        self.send('main', 'Bob will play the %s row' % column_number)


class Mediator(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def custom_log(self, message):
        self.log_info('Received: %s' % message)

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
    #
    # alice.define_individual_rational_strategy()
    # # alice.play_strategy()
    #
    # bob.define_individual_rational_strategy()
    # bob.play_strategy()

    ns.shutdown()