import time
import numpy as np
from osbrain import run_nameserver
from osbrain import run_agent
from osbrain import Agent

game_p1 = np.array([[-1, -10], [0, -9]])
game_p2 = np.array([[-1, 0], [-10, -9]])


class Alice(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self.bind('PUSH', alias='strategy')

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

    def cooperate_strategy(self):
        self.strategy = 0

    def receive_strategy(self, strategy):
        self.opponent_strategy = strategy

    def tit_for_tat_strategy(self, round_number):
        if round_number == 1:
            self.strategy = 0
        else:
            if self.opponent_strategy == 1:
                self.strategy = 1
            else:
                self.strategy = 0

    def play_strategy(self):
        row_number = str(int(self.strategy+1)) + str("th")

        self.send('main', 'Alice will play the %s row' % row_number)
        self.send('strategy', self.strategy)

        self.last_strategy = self.strategy


class Bob(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self.bind('PUSH', alias='strategy')

    def hello(self, name):
        self.send('main', 'Hello to %s from Bob!' % name)

    def custom_log(self, message):
        self.log_info('Received: %s' % message)

    def nash_strategy(self):
        self.strategy = 1

    def random_strategy(self):
        left_column = np.random.uniform()

        if left_column > 0.5:
            self.strategy = 0
        else:
            self.strategy = 1

    def cooperate_strategy(self):
        self.strategy = 0

    def tit_for_tat_strategy(self, round_number):
        if round_number == 1:
            self.strategy = 0
        else:
            if self.opponent_strategy == 1:
                self.strategy = 1
            else:
                self.strategy = 0

    def receive_strategy(self, strategy):
        self.opponent_strategy = strategy

    def play_strategy(self):
        column_number = str(int(self.strategy+1)) + str("th")

        self.send('main', 'Bob will play the %s column' % column_number)
        self.send('strategy', self.strategy)

        self.last_strategy = self.strategy


class Mediator(Agent):
    def on_init(self):
        self.bind('PUSH', alias='Alice_general')
        self.bind('PUSH', alias='Bob_general')

        self.bind('PUSH', alias='Alice_strategy_opponent')
        self.bind('PUSH', alias='Bob_strategy_opponent')

        self.cumulative_payoffs_p1 = 0
        self.cumulative_payoffs_p2 = 0

    def custom_log(self, message):
        self.log_info('Received: %s' % message)

    def receive_strategy_alice(self, strategy_alice):
        self.alice_played = strategy_alice

    def receive_strategy_bob(self, strategy_bob):
        self.bob_played = strategy_bob

    def ask_strategy(self):
        self.send('Alice_general', 'Please state your strategy Alice')
        self.send('Bob_general', 'Please state your strategy Bob')

    def compute_result(self, round_number):
        result_p1 = game_p1[self.alice_played, self.bob_played]
        result_p2 = game_p2[self.alice_played, self.bob_played]

        self.send('Alice_general', 'You have received a payoff last round of: %s' % str(result_p1))
        self.send('Bob_general', 'You have received a payoff last round of: %s' % str(result_p2))

        self.cumulative_payoffs_p1 += result_p1
        self.cumulative_payoffs_p2 += result_p2

        self.average_result_p1 = self.cumulative_payoffs_p1/round_number
        self.average_result_p2 = self.cumulative_payoffs_p2/round_number

        time.sleep(0.5)

        self.send('Alice_general', 'You have an average reward of: %s' % str(self.average_result_p1))
        self.send('Bob_general', 'You have an average reward of: %s' % str(self.average_result_p2))

    def send_strategies(self):
        self.send('Alice_strategy_opponent', self.bob_played)
        self.send('Bob_strategy_opponent', self.alice_played)

if __name__ == '__main__':

    ns = run_nameserver()
    alice = run_agent('Alice', base=Alice)
    bob = run_agent('Bob', base=Bob)
    mediator = run_agent('Mediator', base=Mediator)

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

    for i in np.arange(0, 100):
        print("")
        print("NEW ROUND, ROUND NUMBER:", i+1)
        mediator.ask_strategy()
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