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
        pure_strategy_exist = False
        save_indices = np.zeros(game.shape[1])

        for i in np.arange(0, game.shape[1]):
            save_indices[i] = np.argmax(game[:, i])

        if np.max(save_indices) == np.min(save_indices):
            pure_strategy_exist = True
            self.strategy = save_indices[0]

        if not pure_strategy_exist:
            print("Yolo in de polo")
            
class Bob(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def hello(self, name):
        self.send('main', 'Hello, %s!' % name)

    def custom_log(self, message):
        self.log_info('Received: %s' % message)

    def define_individual_rational_strategy(self):
        print("L")

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

    ns.shutdown()