import numpy as np
import time
import matplotlib.pyplot as plt


class ETPGame:
    """ In this code all the game types are combined , we have 6
    types . 3 Consisting of: 1. Regular Bertrand , 2. Stackelberg - Bertrand , 3.
    Collusion under Bertrand , 4. Regular Cournot , 5. Stackelberg - Cournot ,
    6. Collusion under Cournot . """

    def __init__(self, game_type, x=100, ad_varA=150, ad_varB=150, ad_fixA =30, ad_fixB = 30 , rho11 = 1, rho12 = 0,
                 rho21=0, rho22=0):
        """ Here we initlialize the game by storing the game type
        and the matrix player entries ,
        8 the assigned values are default values in case of non -
        assignment by user . """
        self.game_type = game_type # type of game
        self.x = x # positive intercept for demand

        self.ad_varA = ad_varA # the variable advertisement investment cost for player A
        self.ad_varB = ad_varB # the variable advertisement investment cost for player B
        self.ad_fixA = ad_fixA # the fixed advertisement investment cost for player A
        self.ad_fixB = ad_fixB # the fixed advertisement investment cost for player B
        self.rho11 = rho11 # top - left entry of matrix
        self.rho12 = rho12 # bottom - left entry of matrix
        self.rho21 = rho21 # top - right entry of matrix
        self.rho22 = rho22 # bottom - right entry of matrix

        # Here we add a check to ensure the user inputs a valid game type .
        if self.game_type != 0 and self.game_type != 1 and self.game_type != 2 and self.game_type != 3 and \
                self.game_type != 4 and self.game_type != 5:
            print("Please enter a valid game type . The valid game types are 0 , 1, 2, 3, 4 and 5.")
        # We must add a check for non - positive intercepts for demand
        if self.x < 0:
            self.x = 0
            print(" The positive intercept for demand can not be negative , the value has therefore been changed to 0.")

        # Additionally we must add a check to ensure the values input for all advertisement investment costs ( both var and fix )
        # does not take values below 0 since they are realistically unfeasible , and the game changes them to 0, which is the
        # feasible minimum . Below this is done for all 4 variables .
        if self.ad_varA < 0:
            self . ad_varA = 0
            print(" The variable investment costs for advertisement of player A can not be negative , the value has therefore been changed to 0.")

        if self . ad_varB < 0:
            self . ad_varB = 0
            print (" The variable investment costs for advertisement of player B can not be negative , the value has therefore been changed to 0.")

        if self . ad_fixA < 0:
            self . ad_fixA = 0
            print (" The fixed investment costs for advertisement of player A can not be negative , the value has therefore been changed to 0.")

        if self . ad_fixB < 0:
            self . ad_fixB = 0
            print (" The fixed investment costs for advertisement of player B can not be negative , the value has therefore been changed to 0.")

        # Lastly Rho 's needs to sum to 1.
        if ( self . rho11 + self . rho12 + self . rho21 + self . rho22 ) != 1:
            print (" The player rho inputs must sum to 1.")

    def rho_generator ( self ):
        " Generate rho vectors which can be used as mixed strategy input for the players ."

        # Define the way the random rho 's for the mixed strategies are generated .
        draw_rho11 = np . random . uniform () # draw a number from the uniform distribution
        draw_rho12 = np . random . uniform ()
        draw_rho21 = np . random . uniform ()
        draw_rho22 = np . random . uniform ()
        # print ( draw_rho11 , draw_rho12 , draw_rho21 , draw_rho22 )
        sum_draw = draw_rho11 + draw_rho12 + draw_rho21 + draw_rho22 # Sum the rho 's

        gen_rho11 = draw_rho11 / sum_draw # divide single entries by sum to get values between 0 and 1
        gen_rho12 = draw_rho12 / sum_draw
        gen_rho21 = draw_rho21 / sum_draw
        gen_rho22 = draw_rho22 / sum_draw
        sum_check = gen_rho11 + gen_rho12 + gen_rho21 + gen_rho22 # check to see if sum does not exceed 1
        self . rho11 = gen_rho11 # assign generated rho 's to corresponding variable
        self . rho12 = gen_rho12
        self . rho21 = gen_rho21
        self . rho22 = gen_rho22
        # print ( gen_rho11 , gen_rho12 , gen_rho21 , gen_rho22 )

    def extra_rho_generator ( self ) :
        " Generate rho vectors which can be used as mixed strategy input for the players ."

    # Define the way the random rho 's for the mixed strategies are generated .
        draw_rho11 = np . random . beta (20 ,1) # draw a number from the binomial distribution
        draw_rho12 = np . random . beta (2 ,8)
        draw_rho21 = np . random . beta (2 ,8)
        draw_rho22 = 0
        print ( draw_rho11 , draw_rho12 , draw_rho21 , draw_rho22 )
        sum_draw = draw_rho11 + draw_rho12 + draw_rho21 + draw_rho22 # Sum the rho 's
        gen_rho11 = draw_rho11 / sum_draw # divide single entries by sum to get values between 0 and 1
        gen_rho12 = draw_rho12 / sum_draw
        gen_rho21 = draw_rho21 / sum_draw
        gen_rho22 = draw_rho22 / sum_draw
        sum_check = gen_rho11 + gen_rho12 + gen_rho21 + gen_rho22 # check to see if sum does not exceed 1
        self . rho11 = gen_rho11 # assign generated rho 's to corresponding variable
        self . rho12 = gen_rho12
        self . rho21 = gen_rho21
        self . rho22 = gen_rho22
        # print ( gen_rho11 , gen_rho12 , gen_rho21 , gen_rho22 )

    def pure_strategies ( self , counter_ps ) :
        " Function which includes the four pure strategies in which the players only stick to one option ."

        # Assign the rho 's to the variables and count how many pure strategies there are , so that only 4 options are used .
        if counter_ps == 0: # if there are 0 pure strategies
            self . rho11 = 1
            self . rho12 = 0
            self . rho21 = 0
            self . rho22 = 0

        if counter_ps == 1: # if there is 1 pure strategies
            self . rho11 = 0
            self . rho12 = 1

        if counter_ps == 2: # if there are 2 pure strategies

            self . rho12 = 0
            self . rho21 = 1

        if counter_ps == 3: # if there are 3 pure strategies
            self . rho21 = 0
            self . rho22 = 1

    def equilibrium_computation ( self , print_text = False ) :
        " This function contains the calculations of all variables to determine the equilibrium points ."

        # Here we calculate the first attributes of used in both the Bertrand & Cournot models which we can later use to
        # calculate prices .

        # In this case we use fixed formula 's ( possibly add them to input )
        z1 = 24 - 6 * ( self . rho11 + self . rho12 ) # demand modifiers dependent on advertisement behavior
        z2 = 8 - 4 * ( self . rho11 + self . rho21 )
        z3 = 24 - 6 * ( self . rho11 + self . rho21 )
        z4 = 8 - 4 * ( self . rho11 + self . rho12 )
        D = self .x * (3 * self . rho11 + self . rho12 + self . rho21 ) # the variable demand function containing the positive intercept
        DA0 = 100 # fixed demand
        DB0 = 100
        ca = 3 # cost modifiers
        cb = 3

        # For q we add a special statement to ensure no devision by 0 occurs .
        if self . rho11 == 0 and self . rho12 == 0 and self . rho21 == 0:
            q = 0
        else :
            q = ( self . rho11 + self . rho12 ) /(2 * self . rho11 + self . rho12 + self . rho21 )

        # The following four variables are fixed at 150 , 150 , 30 and 30 by default and can be changed as user input .
        ac0a = self.ad_varA # advertisement variable investment cost
        ac0b = self.ad_varB
        c0a = self.ad_fixA # advertisement fixed investment cost
        c0b = self.ad_fixB

        # Now additionally for Cournot we need a transformation on D, the D0 's and the z's which fits the Cournot model .

        # The calculations are as follows .
        g1 = z3 /( z1 * z3 - z2 * z4 )
        g2 = -z2 /( z1 * z3 - z2 * z4 )
        g3 = z1 /( z1 * z3 - z2 * z4 )
        g4 = -z4 /( z1 * z3 - z2 * z4 )
        Ya = ( z3 * ( DA0 + q * D) + z2 * ( DB0 + (1 - q) * D)) /( z1 * z3 - z2 * z4 )
        Yb = ( z4 * ( DA0 + q * D) + z1 * ( DB0 + (1 - q) * D)) /( z1 * z3 - z2 * z4 )

        # Now using the type of ETP game and the above parameters generated we can calculate the optimal price for Bertrand type games
        # and quantities for Cournot type games

        # For regular Bertrand ( game type 0)
        if self . game_type == 0:
            pa = (2 * z3 * ( DA0 + q * D) + z2 * ( DB0 + (1 - q) * D) + 2 * z1 * z3 * ca + z2 * z3 * cb ) /(4 * z1 * z3 - z2 * z4 )
            pb = (2 * z1 * ( DB0 + (1 - q) * D ) + z4 * ( DA0 + q * D) + 2 * z1 * z3 * cb + z1 * z4 * ca ) /(4 * z1 * z3 - z2 * z4 )

        # For Stackelberg - Bertrand ( game type 1)
        if self . game_type == 1:
            paL = (2 * z3 * ( DA0 + q * D) + z2 * ( DB0 + (1 - q) * D) + z2 * z3 * cb ) /(4 * z1 * z3 - 2 * z2 * z4 ) + ca /2
            pbF = ( DB0 + (1 - q) * D + z4 * paL + z3 * cb ) /(2 * z3 )
            pbL = (2 * z1 * ( DB0 + (1 - q) * D) + z4 * ( DA0 + q * D) + z1 * z4 * ca ) /(4 * z1 * z3 - 2 * z2 * z4 ) + cb /2
            paF = ( DA0 + q * D + z2 * pbL + z1 * ca ) /(2 * z1 )

        # For collusion under Bertrand ( game type 2)
        if self . game_type == 2:
            pa = (2 * z3 * ( DA0 + q * D + z1 * ca - z4 * cb ) + ( z2 + z4 ) * ( DB0 + (1 - q) * D + z3 * cb - z2 * ca )) /(4 * z1 * z3 - (z2 + z4 ) **2)
            pb = (2 * z1 * ( DB0 + (1 - q) * D + z3 * cb - z2 * ca ) + ( z2 + z4 ) * ( DA0 + q * D + z1 * ca - z4 * cb )) /(4 * z1 * z3 - ( z2 + z4 ) **2)

        # For regular Cournot ( game type 3)
        if self . game_type == 3:
            xa = ( g2 * Yb + 2 * g3 * Ya - 2 * g3 * ca - g2 * cb ) /(4 * g1 * g3 - g2 * g4 )
            xb = ( g4 * Ya + 2 * g1 * Yb - g4 * ca - 2 * g1 * cb ) /(4 * g1 * g3 - g2 * g4 )

        # For Stackelberg - Cournot ( game type 4)
        if self . game_type == 4:
            xaL = (2 * g3 * ( Ya - ca ) + g2 * ( Yb - cb )) /(4 * g1 * g3 - 2 * g2 * g4 )
            xbF = ( Yb + g4 * xaL - cb ) /(2 * g3 )
            xbL = (2 * g1 * ( Yb - cb ) + g4 * ( Ya - ca )) /(4 * g1 * g3 - 2 * g2 * g4 )
            xaF = ( Ya + g2 * xbL - ca ) /(2 * g1 )

        # For collusion under Cournot ( game type 5)
        if self . game_type == 5:
            xa = (2 * g3 * ( Ya - ca ) + ( g2 + g4 ) * ( Yb - cb )) /(4 * g1 * g3 - ( g2 + g4 ) **2)
            xb = (2 * g1 * ( Yb - cb ) + ( g2 + g4 ) * ( Ya - ca )) /(4 * g1 * g3 - ( g2 + g4 ) **2)

        # Now we calculate the sales potentials for quantity and price here in order to determine the optimal quantity and price.

        # For Bertrand and Cournot respectively.
        # For regular Bertrand and for collusion under Bertrand
        if self . game_type == 0 or self . game_type == 2:
            SPxa = DA0 + q * D - z1 * pa + z2 * pb
            SPxb = DB0 + (1 - q) * D - z3 * pb + z4 * pa

        # For Stackelberg - Bertrand ( since we have leader and follower here )
        if self . game_type == 1:
            SPxaL = DA0 + q * D - z1 * paL + z2 * pbF
            SPxbF = DB0 + (1 - q ) * D - z3 * pbF + z4 * paL
            SPxaF = DA0 + q * D - z1 * paF + z2 * pbL
            SPxbL = DB0 + (1 - q ) * D - z3 * pbL + z4 * paF

        # For regular Cournot and for collusion under Cournot
        if self . game_type == 3 or self . game_type == 5:
            SPpa = Ya - g1 * xa + g2 * xb
            SPpb = Yb - g3 * xb + g4 * xa

        # For Stackelberg - Cournot ( since we have leader and follower here )
        if self . game_type == 4:
            SPpaL = Ya - g1 * xaL + g2 * xbF
            SPpbF = Yb - g3 * xbF + g4 * xaL
            SPpaF = Ya - g1 * xaF + g2 * xbL
            SPpbL = Yb - g3 * xbL + g4 * xaF

        # Calculate the optimal quantities and prices through the sales potential , which we can use to find the profit .
        # Where the margins imposed by advertisement behavior ,
        # by which the actual prices and quanitities are adjusted due to the rho 's, are FIXED

        # For regular Bertrand and collusion under Bertrand
        if self . game_type == 0 or self . game_type == 2:
            xa = SPxa * ( self . rho11 * 1 + self . rho12 * (7/8) + self. rho21 * (5/8) + self . rho22 * (1/2) )
            xb = SPxb * ( self . rho11 * 1 + self . rho12 * (5/8) + self. rho21 * (7/8) + self . rho22 * (1/2) )

        # For Stackelberg - Bertrand
        if self . game_type == 1:
            xaL = SPxaL * ( self . rho11 + self . rho12 * (7/8) + self . rho21 * (5/8) + self . rho22 * (1/2) )
            xbF = SPxbF * ( self . rho11 + self . rho12 * (5/8) + self . rho21 * (7/8) + self . rho22 * (1/2) )
            xaF = SPxaF * ( self . rho11 + self . rho12 * (7/8) + self . rho21 * (5/8) + self . rho22 * (1/2) )
            xbL = SPxbL * ( self . rho11 + self . rho12 * (5/8) + self . rho21 * (7/8) + self . rho22 * (1/2) )

        # For regular Cournot and collusion under Cournot
        if self . game_type == 3 or self . game_type == 5:
            pa = SPpa * ( self . rho11 * 1 + self . rho12 * (7/8) + self . rho21 * (5/8) + self . rho22 * (1/2) )
            pb = SPpb * ( self . rho11 * 1 + self . rho12 * (5/8) + self . rho21 * (7/8) + self . rho22 * (1/2) )

        # For Stackelberg - Cournot
        if self . game_type == 4:
            paL = SPpaL * ( self . rho11 + self . rho12 * (7/8) + self . rho21 * (5/8) + self . rho22 * (1/2) )
            pbF = SPpbF * ( self . rho11 + self . rho12 * (5/8) + self . rho21 * (7/8) + self . rho22 * (1/2) )
            paF = SPpaF * ( self . rho11 + self . rho12 * (7/8) + self . rho21 * (5/8) + self . rho22 * (1/2) )
            pbL = SPpbL * ( self . rho11 + self . rho12 * (5/8) + self . rho21 * (7/8) + self . rho22 * (1/2) )

        # Finally we find the optimal profit for the player entries given the above parameters and a cost - parameter
        # We again seperate regular and collusion Bertrand and Cournot from Stackelberg - Bertrand and Stackelberg - Bertrand ,
        # due to parameter differences .

        # Now we can use the quanitity , price and cost plus the variable and fixed costs of advertisement to calculate the profit
        if self . game_type == 0 or self . game_type == 2:
            PIa = xa * ( pa - ca ) - ( self . rho11 + self . rho12 ) * ac0a - c0a
            PIb = xb * ( pb - cb ) - ( self . rho11 + self . rho21 ) * ac0b - c0b

        # Include statement that if I want printing the seperate equilibrium points are printed for all game - types .
            if print_text :
                if self . game_type == 0:
                    print (" The regular Bertrand profit equilibrium is at:", PIa , PIb )
                else :
                    print (" The collusion under Bertrand profit equilibrium is at:", PIa , PIb )

        if self . game_type == 1:
            PIaL = xaL * ( paL - ca ) - ( self . rho11 + self . rho12 ) * ac0a - c0a
            PIbF = xbF * ( pbF - cb ) - ( self . rho11 + self . rho21 ) * ac0b - c0b
            PIaF = xaF * ( paF - ca ) - ( self . rho11 + self . rho12 ) * ac0a - c0a
            PIbL = xbL * ( pbL - cb ) - ( self . rho11 + self . rho21 ) * ac0b - c0b

            if print_text :
                print (" The Stackelberg - Bertrand profit equilibrium , with Leader A is at:", PIaL , PIbF )
                print (" The Stackelberg - Bertrand profit equilibrium , with Leader B is at:", PIaF , PIbL )

        if self . game_type == 3 or self . game_type == 5:
            PIa = xa * ( pa - ca ) - ( self . rho11 + self . rho12 ) * ac0a - c0a
            PIb = xb * ( pb - cb ) - ( self . rho11 + self . rho21 ) * ac0b - c0b

            if print_text is True :
                if self . game_type == 3:
                    print (" The regular Cournot profit equilibrium is at:", PIa , PIb )
                else :
                    print (" The collusion under Cournot profit equilibrium is at:", PIa , PIb )

        if self . game_type == 4:
            PIaL = xaL * ( paL - ca ) - ( self . rho11 + self . rho12 ) * ac0a - c0a
            PIbF = xbF * ( pbF - cb ) - ( self . rho11 + self . rho21 ) * ac0b - c0b
            PIaF = xaF * ( paF - ca ) - ( self . rho11 + self . rho12 ) * ac0a - c0a
            PIbL = xbL * ( pbL - cb ) - ( self . rho11 + self . rho21 ) * ac0b - c0b

            if print_text is True :
                print (" The Stackelberg - Cournot profit equilibrium , with Leader A is at:", PIaL , PIbF )
                print (" The Stackelberg - Cournot profit equilibrium , with Leader B is at:", PIaF , PIbL )

        if self . game_type == 0 or self . game_type == 2 or self . game_type == 3 or self . game_type == 5:
            return [ PIa , PIb ] # return equilibrium profit for regular - and collusion Bertrand and Cournot
        if self . game_type == 1 or self . game_type == 4:
            return [ PIaL , PIbF , PIaF , PIbL ] # return equilibrium profit for Stackelberg - Bertrand and Stackelberg - Cournot

    def optimal_profit ( self , rho_generated , extra_rho_generated , total_points = 4, extra_points = 0) :
        "We calculate the optimum profit for all game types using a lot of options ."

        #We start off by timing the optimal profit iteration process
        start_time = time . time () # uses the time package to start timer

        # First define the boundries of the equilibrium matrices for both pure strategies and mixed strategies .
        if self . game_type == 0 or self . game_type == 2 or self . game_type == 3 or self . game_type == 5:
            equilibrium_matrix_total = np . zeros (( total_points , 2) )
            pure_equilibrium_total = np . zeros ((4 , 2) )
            extra_equilibrium_total = np . zeros (( extra_points , 2) )

        if self . game_type == 1 or self . game_type == 4:
            equilibrium_matrix_total = np . zeros (( total_points , 4) )
            pure_equilibrium_total = np . zeros ((4 , 4) )
            extra_equilibrium_total = np . zeros (( extra_points , 4) )

        # Now iterate first over the four pure strategies and assign them to the previously bounded matrix .
        for i in range (0 , 4) :
            self . pure_strategies (i )

            pure_equilibrium = self . equilibrium_computation ()
            pure_equilibrium_total [i , :] = pure_equilibrium

        self . pure_equilibrium_total = pure_equilibrium_total

        # For a total_points amount of iterations loop through the calculations (by default 1)
        for i in range (0 , total_points ):
            if rho_generated :
                self . rho_generator ()

            random_equilibrium = self . equilibrium_computation ()
            equilibrium_matrix_total [i , :] = random_equilibrium

        self . equilibrium_matrix_total = equilibrium_matrix_total

        # For a extra_points amount of iterations loop through the rho 's which increase tail - weight .
        for i in range (0 , extra_points ):
            if extra_rho_generated :
                self . extra_rho_generator ()

            extra_equilibrium = self . equilibrium_computation ()
            extra_equilibrium_total [i , :] = extra_equilibrium

        self . extra_equilibrium_total = extra_equilibrium_total

        # Lastly we end the timer and print how long the process took
        end_time = time . time () # stops the timer
        print (" Running time for equilibrium points generation :", end_time - start_time )

    def plot_equilibrium_outcomes ( self , leader = 0) :
        " Plot the equilibrium outcomes of the different type games , and additionally print the investment costs for advertisements used ."

        if self . game_type == 0:
            # Plots first two entries of equilibrium point , does not require axis assigning .
            plt . scatter ( self . pure_equilibrium_total [0 ,0] , self . pure_equilibrium_total [0 ,1] , color ='r', zorder =2 , s =10 , label ="Pure strategy [1 , 0, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [1 ,0] , self .pure_equilibrium_total [1 ,1] , color ='gold', zorder =2 , s =10 , label =" Pure strategy [0 , 1, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [2 ,0] , self . pure_equilibrium_total [2 ,1] , color ='b', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 1, 0]")
            plt . scatter ( self . pure_equilibrium_total [3 ,0] , self . pure_equilibrium_total [3 ,1] , color ='g', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 0, 1]")
            plt . scatter ( self . equilibrium_matrix_total [: ,0] , self . equilibrium_matrix_total [: ,1] , color ='cyan', zorder =1 , s=1 , label ="LAR regular Bertrand ")
            plt . scatter ( self . extra_equilibrium_total [: ,0] , self . extra_equilibrium_total [: ,1] , color ='cyan', zorder =1 , s =1)
            plt . title (" Limiting average rewards for regular Bertrand competition ")
            plt . xlabel (" Optimal profit player A")
            plt . ylabel (" Optimal profit player B")
            plt . legend ( loc ='center left', bbox_to_anchor =(1.1 , 0.5) , labelspacing =3)
            plt . figtext (0.15 , 0.85 , " ac0a = 900 , ac0b = 150 ", horizontalalignment ="left", verticalalignment ="top", wrap = True , fontsize = 10 , bbox ={ 'facecolor': 'grey', 'alpha':0.3 , 'pad':5})
            # plt . show ()
            print (" With the positive intercept for demand at:", self .x)
            print (" With variable advertisement investment cost for player A at:", self . ad_varA )
            print (" With variable advertisement investment cost for player B at:", self . ad_varB )
            print (" With fixed advertisement investment cost for player A at:", self . ad_fixA )
            print (" With fixed advertisement investment cost for player B at:", self . ad_fixB )
            print ("")

        if self . game_type == 1 and leader == 0:
            # Plots first two entries of equilibrium point , does not require axis assigning .
            plt . scatter ( self . pure_equilibrium_total [0 ,0] , self . pure_equilibrium_total [0 ,1] , color ='r', zorder =2 , s =10 , label ="Pure strategy [1 , 0, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [1 ,0] , self . pure_equilibrium_total [1 ,1] , color ='gold', zorder =2 , s =10 ,label =" Pure strategy [0 , 1, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [2 ,0] , self . pure_equilibrium_total [2 ,1] , color ='b', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 1, 0]")
            plt . scatter ( self . pure_equilibrium_total [3 ,0] , self . pure_equilibrium_total [3 ,1] , color ='g', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 0, 1]")
            plt . scatter ( self . equilibrium_matrix_total [: ,0] , self . equilibrium_matrix_total [: ,1] , color ='aqua', zorder =1 , s=1 , label ="LAR Stackelberg - Bertrand ")
            plt . scatter ( self . extra_equilibrium_total [: ,0] , self . extra_equilibrium_total [: ,1] , color ='aqua', zorder =1 , s =1)
            plt . title (" Limiting average rewards for Stack .- Bertrand competition (L-A, F-B).")
            plt . xlabel (" Optimal profit player A")
            plt . ylabel (" Optimal profit player B")

            plt . legend ( loc ='center left', bbox_to_anchor =(1.1 , 0.5) , labelspacing =3)
            # plt . show ()
            print (" With the positive intercept for demand at:", self .x)
            print (" With variable advertisement investment cost for player A at:", self . ad_varA )
            print (" With variable advertisement investment cost for player B at:", self . ad_varB )
            print (" With fixed advertisement investment cost for player A at:", self . ad_fixA )
            print (" With fixed advertisement investment cost for player B at:", self . ad_fixB )
            print ("")

        if self . game_type == 1 and leader == 1:
            # Plots first two entries of equilibrium point , does not require axis assigning .
            plt . scatter ( self . pure_equilibrium_total [0 ,0] , self . pure_equilibrium_total [0 ,1] , color ='r', zorder =2 , s =10 , label ="Pure strategy [1 , 0, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [1 ,0] , self . pure_equilibrium_total [1 ,1] , color ='gold', zorder =2 , s =10 , label =" Pure strategy [0 , 1, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [2 ,0] , self . pure_equilibrium_total [2 ,1] , color ='b', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 1, 0]")
            plt . scatter ( self . pure_equilibrium_total [3 ,0] , self . pure_equilibrium_total [3 ,1] , color ='g', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 0, 1]")
            plt . scatter ( self . equilibrium_matrix_total [: ,2] , self . equilibrium_matrix_total [: ,3] , color ='aqua', zorder =1 , s=1 , label ="LAR Stackelberg - Bertrand ")
            plt . scatter ( self . extra_equilibrium_total [: ,0] , self . extra_equilibrium_total [: ,1] , color ='aqua', zorder =1 , s =1)
            plt . title (" Limiting average rewards for Stackelberg - Bertrand competition with Leader B and Follower A.")
            plt . xlabel (" Optimal profit player A")
            plt . ylabel (" Optimal profit player B")
            plt . legend ( loc ='center left', bbox_to_anchor =(1.1 , 0.5) , labelspacing =3)
            plt . figtext (0.15 , 0.85 , " ac0a = ac0b = 150 ", horizontalalignment ="left", verticalalignment ="top", wrap =True , fontsize = 10 , bbox ={ 'facecolor':'grey', 'alpha ':0.3 , 'pad ':5})
            # plt . show ()
            print (" With the positive intercept for demand at:", self .x)
            print (" With variable advertisement investment cost for player A at:", self . ad_varA )
            print (" With variable advertisement investment cost for player B at:", self . ad_varB )
            print (" With fixed advertisement investment cost for player A at:", self . ad_fixA )
            print (" With fixed advertisement investment cost for player B at:", self . ad_fixB )
            print ("")


        if self . game_type == 2:
            # Plots first two entries of equilibrium point , does not require axis assigning .
            plt . scatter ( self . pure_equilibrium_total [0 ,0] , self . pure_equilibrium_total [0 ,1] , color ='r', zorder =2 , s =10 , label ="Pure strategy [1 , 0, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [1 ,0] , self . pure_equilibrium_total [1 ,1] , color ='gold', zorder =2 , s =10 ,label =" Pure strategy [0 , 1, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [2 ,0] , self . pure_equilibrium_total [2 ,1] , color ='b', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 1, 0]")
            plt . scatter ( self . pure_equilibrium_total [3 ,0] , self . pure_equilibrium_total [3 ,1] , color ='g', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 0, 1]")
            plt . scatter ( self . equilibrium_matrix_total [: ,0] , self . equilibrium_matrix_total [: ,1] , color ='teal', zorder =1 , s=1 , label ="LAR collusion Bertrand ")
            plt . scatter ( self . extra_equilibrium_total [: ,0] , self . extra_equilibrium_total [: ,1] , color ='teal', zorder =1 , s =1)
            plt . title (" Limiting average rewards for collusion under Bertrand competition .")
            plt . xlabel (" Optimal profit player A")
            plt . ylabel (" Optimal profit player B")
            plt . legend ( loc ='center left', bbox_to_anchor =(1.1 , 0.5) , labelspacing =3)
            # plt . show ()
            print (" With the positive intercept for demand at:", self .x)
            print (" With variable advertisement investment cost for player A at:", self . ad_varA )
            print (" With variable advertisement investment cost for player B at:", self . ad_varB )
            print (" With fixed advertisement investment cost for player A at:", self . ad_fixA )
            print (" With fixed advertisement investment cost for player B at:", self . ad_fixB )
            print ("")

        if self . game_type == 3:
            # Plots first two entries of equilibrium point , does not require axis assigning .
            plt . scatter ( self . pure_equilibrium_total [0 ,0] , self .pure_equilibrium_total [0 ,1] , color ='r', zorder =2 , s =10 , label ="Pure strategy [1 , 0, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [1 ,0] , self .pure_equilibrium_total [1 ,1] , color ='gold', zorder =2 , s =10 ,label =" Pure strategy [0 , 1, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [2 ,0] , self .pure_equilibrium_total [2 ,1] , color ='b', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 1, 0]")
            plt . scatter ( self . pure_equilibrium_total [3 ,0] , self .pure_equilibrium_total [3 ,1] , color ='g', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 0, 1]")
            plt . scatter ( self . equilibrium_matrix_total [: ,0] , self .equilibrium_matrix_total [: ,1] , color ='pink', zorder =1 , s=1 , label ="LAR regular Cournot ")
            plt . scatter ( self . extra_equilibrium_total [: ,0] , self .extra_equilibrium_total [: ,1] , color ='pink', zorder =1 , s =1)
            plt . title (" Limiting average rewards for regular Cournot competition .")
            plt . xlabel (" Optimal profit player A")
            plt . ylabel (" Optimal profit player B")
            plt . legend ( loc ='center left', bbox_to_anchor =(1.1 , 0.5) , labelspacing =3)
            plt . figtext (0.80 , 0.85 , "u = 0", horizontalalignment ="left", verticalalignment ="top", wrap = True , fontsize = 10 ,bbox ={ 'facecolor':'grey', 'alpha':0.3 , 'pad':5})
            # plt . figtext (0.80 , 0.85 , "u = 0" , horizontalalignment =" left ", verticalalignment =" top ", wrap = True , fontsize = 10 , box ={ ' facecolor ':' grey ', 'alpha ':0.3 , 'pad ':5})
            # plt . show ()
            print (" With the positive intercept for demand at:", self .x)
            print (" With variable advertisement investment cost for player A at:", self . ad_varA )
            print (" With variable advertisement investment cost for player B at:", self . ad_varB )
            print (" With fixed advertisement investment cost for player A at:", self . ad_fixA )
            print (" With fixed advertisement investment cost for player B at:", self . ad_fixB )
            print ("")

        if self . game_type == 4 and leader == 0:
            # Plots first two entries of equilibrium point , does not require axis assigning .
            plt . scatter ( self . pure_equilibrium_total [0 ,0] , self . pure_equilibrium_total [0 ,1] , color ='r', zorder =2 , s =10 , label ="Pure strategy [1 , 0, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [1 ,0] , self . pure_equilibrium_total [1 ,1] , color ='gold', zorder =2 , s =10 ,label =" Pure strategy [0 , 1, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [2 ,0] , self . pure_equilibrium_total [2 ,1] , color ='b', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 1, 0]")
            plt . scatter ( self . pure_equilibrium_total [3 ,0] , self . pure_equilibrium_total [3 ,1] , color ='g', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 0, 1]")
            plt . scatter ( self . equilibrium_matrix_total [: ,0] , self .equilibrium_matrix_total [: ,1] , color ='orchid', zorder =1 , s=1 , label ="LAR Stackelberg - Cournot ")
            plt . scatter ( self . extra_equilibrium_total [: ,0] , self . extra_equilibrium_total [: ,1] , color ='orchid', zorder =1 , s=1)
            plt . title (" Limiting average rewards for Stackelberg - Cournot competition with Leader A and Follower B.")
            plt . xlabel (" Optimal profit player A")
            plt . ylabel (" Optimal profit player B")
            plt . legend ( loc ='center left', bbox_to_anchor =(1.1 , 0.5) , labelspacing =3)
            # plt . show ()

            print (" With the positive intercept for demand at:", self .x)
            print (" With variable advertisement investment cost for player A at:", self . ad_varA )
            print (" With variable advertisement investment cost for player B at:", self . ad_varB )
            print (" With fixed advertisement investment cost for player A at:", self . ad_fixA )
            print (" With fixed advertisement investment cost for player B at:", self . ad_fixB )
            print ("")

        if self . game_type == 4 and leader == 1:
            # Plots first two entries of equilibrium point , does not require axis assigning .
            plt . scatter ( self . pure_equilibrium_total [0 ,0] , self . pure_equilibrium_total [0 ,1] , color ='r', zorder =2 , s =10 , label ="Pure strategy [1 , 0, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [1 ,0] , self . pure_equilibrium_total [1 ,1] , color ='gold', zorder =2 , s =10 ,label =" Pure strategy [0 , 1, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [2 ,0] , self . pure_equilibrium_total [2 ,1] , color ='b', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 1, 0]")
            plt . scatter ( self . pure_equilibrium_total [3 ,0] , self . pure_equilibrium_total [3 ,1] , color ='g', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 0, 1]")
            plt . scatter ( self . equilibrium_matrix_total [: ,2] , self . equilibrium_matrix_total [: ,3] , color ='orchid', zorder =1 , s=1 , label ="LAR Stackelberg - Cournot ")
            plt . scatter ( self . extra_equilibrium_total [: ,0] , self . extra_equilibrium_total [: ,1] , color ='orchid', zorder =1 , s=1)
            plt . title (" Limiting average rewards for Stack .- Cournot competition (L-B,F-A).")
            plt . xlabel (" Optimal profit player A")
            plt . ylabel (" Optimal profit player B")
            plt . legend ( loc ='center left', bbox_to_anchor =(1.1 , 0.5) , labelspacing =3)
            # plt . show ()
            print (" With the positive intercept for demand at:", self .x)
            print (" With variable advertisement investment cost for player A at:", self . ad_varA )
            print (" With variable advertisement investment cost for player B at:", self . ad_varB )
            print (" With fixed advertisement investment cost for player A at:", self . ad_fixA )
            print (" With fixed advertisement investment cost for player B at:", self . ad_fixB )
            print ("")

        if self . game_type == 5:
            # Plots first two entries of equilibrium point , does not require axis assigning .
            plt . scatter ( self . pure_equilibrium_total [0 ,0] , self . pure_equilibrium_total [0 ,1] , color ='r', zorder =2 , s =10 , label ="Pure strategy [1 , 0, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [1 ,0] , self . pure_equilibrium_total [1 ,1] , color ='gold', zorder =2 , s =10 , label =" Pure strategy [0 , 1, 0, 0]")
            plt . scatter ( self . pure_equilibrium_total [2 ,0] , self . pure_equilibrium_total [2 ,1] , color ='b', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 1, 0]")
            plt . scatter ( self . pure_equilibrium_total [3 ,0] , self . pure_equilibrium_total [3 ,1] , color ='g', zorder =2 , s =10 , label ="Pure strategy [0 , 0, 0, 1]")
            plt . scatter ( self . equilibrium_matrix_total [: ,0] , self . equilibrium_matrix_total [: ,1] , color ='violet', zorder =1 , s=1 , label ="LAR collusion Cournot ")
            plt . scatter ( self . extra_equilibrium_total [: ,0] , self . extra_equilibrium_total [: ,1] , color ='violet', zorder =1 , s=1)
            plt . title (" Limiting average rewards for collusion under Cournot competition .")
            plt . xlabel (" Optimal profit player A")
            plt . ylabel (" Optimal profit player B")
            plt . legend ( loc ='center left', bbox_to_anchor =(1.1 , 0.5) , labelspacing =3)
            # plt . show ()
            print (" With the positive intercept for demand at:", self .x)
            print (" With variable advertisement investment cost for player A at:", self . ad_varA )
            print (" With variable advertisement investment cost for player B at:", self . ad_varB )
            print (" With fixed advertisement investment cost for player A at:", self . ad_fixA )
            print (" With fixed advertisement investment cost for player B at:", self . ad_fixB )
            print ("")

    def check_extremes ( self ):
        " Check extreme values in Pure and mixed strategies ."

        # Check minimum values for pure - and mixed strategies
        print (" The minimum value for the pure strategy equilibrium values :")
        print ( np . min ( self . pure_equilibrium_total , axis =0) )
        print (" The minimum value for the mixed strategy equilibrium values :")
        print ( np . min ( self . equilibrium_matrix_total , axis =0) )
        print ("")

        # Check maximum values for pure - and mixed strategies
        print (" The maximum value for the pure strategy equilibrium values :")
        print ( np . max ( self . pure_equilibrium_total , axis =0) )
        print (" The maximum value for the mixed strategy equilibrium values :")
        print ( np . max ( self . equilibrium_matrix_total , axis =0) )
