def profit_function(mu):

    profit = float(1/3.75) * ((4 + (0.75/mu**2)) - (12 + ((1-12*mu**2.5)/mu**1.5)))

    return profit
