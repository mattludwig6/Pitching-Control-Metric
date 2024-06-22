import numpy as np
import numpy.linalg as LA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#manually entered matrices for average, ground ball %, contact %, and swing %. vector for hit distribution (single, double, triple, home run)
JD_fb_avgp = np.array([[0.000, 0.000, 0.013, 0.010, 0.000],
[0.000, 0.053, 0.043, 0.047, 0.037],
[0.029, 0.087, 0.066, 0.051, 0.003],
[0.053, 0.119, 0.091, 0.034, 0.008],
[0.000, 0.000, 0.033, 0.000, 0.010]])
JD_fb_gbp = np.array([[0.00, 0.00, 0.00, 0.00, 0.02],
[0.00, 0.01, 0.04, 0.05, 0.03],
[0.03, 0.10, 0.05, 0.04, 0.02],
[0.05, 0.12, 0.09, 0.06, 0.01],
[0.00, 0.00, 0.10, 0.03, 0.02]])
JD_fb_contact = np.array([[0.22, 0.58, 0.45, 0.57, 0.33],
[0.70, 0.75, 0.72, 0.64, 0.61],
[0.91, 0.90, 0.78, 0.73, 0.55],
[1.00, 0.97, 0.84, 0.84, 0.74],
[0.00, 1.00, 0.86, 0.75, 0.50]])
JD_fb_swing = np.array([[0.26, 0.33, 0.47, 0.36, 0.17],
[0.24, 0.76, 0.84, 0.72, 0.22],
[0.31, 0.69, 0.75, 0.62, 0.24],
[0.11, 0.49, 0.57, 0.32, 0.08],
[0.00, 0.22, 0.23, 0.05, 0.04]])
JD_hit_dist = np.array([0.693, 0.184, 0.13, 0.110])

MT_fb_avgp = np.array([[0.000, 0.002, 0.000, 0.003, 0.000],
[0.009, 0.056, 0.061, 0.036, 0.011],
[0.033, 0.119, 0.125, 0.082, 0.014],
[0.059, 0.113, 0.168, 0.090, 0.013],
[0.024, 0.044, 0.050, 0.024, 0.003]])
MT_fb_gbp = np.array([[0.00, 0.00, 0.00, 0.00, 0.00],
[0.03, 0.03, 0.04, 0.02, 0.00],
[0.10, 0.11, 0.07, 0.06, 0.01],
[0.04, 0.14, 0.10, 0.10, 0.03],
[0.01, 0.05, 0.06, 0.04, 0.01]])
MT_fb_contact = np.array([[0.67, 0.50, 0.55, 0.50, 0.00],
[0.91, 0.83, 0.84, 0.72, 0.93],
[0.93, 0.94, 0.88, 0.86, 0.69],
[0.76, 0.93, 0.92, 0.86, 0.55],
[0.50, 0.69, 0.61, 0.49, 0.16]])
MT_fb_swing = np.array([[0.02, 0.08, 0.14, 0.11, 0.01],
[0.12, 0.47, 0.57, 0.44, 0.05],
[0.28, 0.65, 0.64, 0.48, 0.14],
[0.31, 0.65, 0.62, 0.45, 0.17],
[0.24, 0.32, 0.43, 0.30, 0.11]])
MT_hit_dist = np.array([0.558, 0.189, 0.035, 0.218])

#value of JD against lefties for comparison
JD_fb_avgp_L = np.array([[0.000, 0.000, 0.020, 0.019, 0.000],
[0.000, 0.030, 0.060, 0.081, 0.000],
[0.000, 0.029, 0.104, 0.121, 0.010],
[0.000, 0.046, 0.094, 0.091, 0.022],
[0.000, 0.000, 0.043, 0.000, 0.000]])
JD_fb_gbp_L = np.array([[0.00, 0.00, 0.03, 0.01, 0.00],
[0.01, 0.04, 0.06, 0.07, 0.01],
[0.01, 0.04, 0.11, 0.12, 0.01],
[0.00, 0.09, 0.08, 0.10, 0.02],
[0.00, 0.00, 0.02, 0.00, 0.00]])
JD_fb_contact_L = np.array([[0.40, 0.40, 0.59, 0.58, 0.67],
[0.67, 0.63, 0.72, 0.82, 0.83],
[0.73, 0.75, 0.88, 0.89, 0.89],
[1.00, 0.88, 0.96, 1.00, 0.83],
[0.00, 0.00, 1.00, 0.80, 0.00]])
JD_fb_swing_L = np.array([[0.03, 0.024, 0.49, 0.45, 0.09],
[0.06, 0.56, 0.82, 0.80, 0.27],
[0.09, 0.57, 0.79, 0.81, 0.37],
[0.10, 0.37, 0.50, 0.53, 0.27],
[0.00, 0.00, 0.20, 0.17, 0.00]])

strikezone = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
coordinates_dic = {0:(1,1), 1:(1,2), 2:(1,3), 3:(2,1), 4:(2,2), 5:(2,3), 6:(3,1), 7:(3,2), 8:(3,3)}
gamestate = {0:(0,0,0),1:(1,0,0),2:(0,1,0),3:(0,0,1),4:(1,1,0),5:(0,1,1),6:(1,0,1),7:(1,1,1)}
state_val = {(0, 0): [0.674 , 0.2476, 0.2459, 0.2489, 0.6572, 0.9986, 1.2532, 1.432 ], (1, 0): [1.2232, 0.4405, 0.4274, 0.0882, 1.231 , 1.5767, 2.2483, 2.4149], (2, 0): [1.2236, 0.6984, 0.7008, 0.678 , 1.6778, 2.0027, 2.2648, 2.4189], (3, 0): [1.5722, 0.9039, 1.2437, 0.8659, 1.685 , 2.0014, 2.2347, 2.4136], (4, 0): [2.1377, 0.8338, 0.8156, 0.3493, 2.2139, 2.8153, 3.2381, 3.4281], (5, 0): [2.1291, 1.3287, 1.6874, 1.2926, 2.6873, 2.9669, 3.2511, 3.4104], (6, 0): [2.1119, 1.0232, 1.4388, 1.1029, 2.2464, 2.8364, 3.2557, 3.4151], (7, 0): [3.1068, 1.4648, 1.8526, 1.3547, 3.216 , 3.8439, 4.2324, 4.4275], (0, 1): [0.4217, 0.098 , 0.1053, 0.105 , 0.4271, 0.6853, 0.8703, 1.2616], (1, 1): [0.8503, 0.2067, 0.208 , 0.    , 0.8496, 1.0243, 1.8688, 2.2598], (2, 1): [0.8442, 0.3559, 0.3523, 0.3585, 1.4256, 1.6924, 1.8598, 2.2479], (3, 1): [1.0377, 0.349 , 1.105 , 0.3555, 1.4236, 1.6721, 1.8869, 2.2256], (4, 1): [1.4324, 0.4793, 0.4679, 0.    , 1.8603, 2.3318, 2.8856, 3.249 ], (5, 1): [1.4827, 0.5983, 1.3548, 0.5842, 2.4401, 2.6712, 2.8912, 3.2509], (6, 1): [1.455 , 0.4728, 1.2126, 0.    , 1.8611, 2.3344, 2.88  , 3.2403], (7, 1): [2.445 , 0.7482, 1.4676, 0.    , 2.8468, 3.3234, 3.8678, 4.2449], (0, 2): [0.2065, 0.    , 0.    , 0.    , 0.2191, 0.3512, 0.3628, 1.1025], (1, 2): [0.4796, 0.    , 0.    , 0.    , 0.4753, 0.4528, 1.3537, 2.1097], (2, 2): [0.4823, 0.    , 0.    , 0.    , 1.2181, 1.3497, 1.3696, 2.1013], (3, 2): [0.4704, 0.    , 0.    , 0.    , 1.2195, 1.3502, 1.3635, 2.1101], (4, 2): [0.7591, 0.    , 0.    , 0.    , 1.4629, 1.621 , 2.3488, 3.1034], (5, 2): [0.7666, 0.    , 0.    , 0.    , 2.2118, 2.3541, 2.3422, 3.1061], (6, 2): [0.7617, 0.    , 0.    , 0.    , 1.4655, 1.5931, 2.3362, 3.0982], (7, 2): [1.7861, 0.    , 0.    , 0.    , 2.459 , 2.5797, 3.3539, 4.104 ]}
end_props = np.zeros((9,8))

#main function for running the simulation component
def simulate():
    #create the hitter/pitcher heatmap for all pitches
    mixed_fb_avgp = np.average(np.array([JD_fb_avgp,MT_fb_avgp]),axis=0)
    mixed_fb_gbp = np.average(np.array([JD_fb_gbp,MT_fb_gbp]),axis=0)
    mixed_fb_contact = np.average(np.array([JD_fb_contact,MT_fb_contact]),axis=0)
    mixed_fb_swing = np.average(np.array([JD_fb_swing,MT_fb_swing]),axis=0)
    mixed_hit_dist = np.average(np.array([JD_hit_dist,MT_hit_dist]),axis=0)
    
    #dictionary of outcome probability vectors for each pitch type
    outcomes_fb = outcomes_mat(mixed_fb_avgp,mixed_fb_gbp,mixed_fb_contact,mixed_fb_swing,mixed_hit_dist)
    #game_values is a dictionary set up as defined in the function description
    #game_values = value_maker(outcomes_fb)
    
    scores = np.zeros(10000)
    end_props = fill_end_props(outcomes_fb)
    for i in range(10000):
        scores[i] = inning(outcomes_fb)
    print("mean",np.mean(scores))
    print("var",np.var(scores))
    #print("props",props)

#probability outcome vector in each cell of 5x5 matrix (which represents disretized strike zone)
def outcomes_mat(avg,gb,c,s,hd):
    #(whiff, take, foul, ground ball, popup, 1B, 2B, 3B, HR)
    outcomes = {}
    for i in np.arange(5):
        for j in np.arange(5):
            swing = s[i][j]
            take = 1-swing
            whiff = swing*(1-c[i][j])
            avgp = avg[i][j]
            si = avgp*hd[0]
            do = avgp*hd[1]
            tr = avgp*hd[2]
            hr = avgp*hd[3]
            alpha = 1 - sum([take,whiff,gb[i][j],si,do,tr,hr])
            pu = 0.25*alpha
            foul = 0.75*alpha
            v = [whiff,take,foul,gb[i][j],pu,si,do,tr,hr]
            outcomes[(i,j)] = v
    #special condition for ball completely outside the 5x5 which gives automatic take --> ball
    outcomes[(5,5)] = [0,1,0,0,0,0,0,0,0]
    return outcomes

#generates the outcome for a particular pitch randomly given the probability distribution
def pitch_made(p_dist):
    #(whiff, take, foul, ground ball, popup, 1B, 2B, 3B, HR)
    u = np.random.uniform(0,1)
    if(u <= np.sum(p_dist[:1])):
        return "whiff"
    elif(u <= np.sum(p_dist[:2])):
        return "take"
    elif(u <= np.sum(p_dist[:3])):
        return "foul"
    elif(u <= np.sum(p_dist[:4])):
        return "ground ball"
    elif(u <= np.sum(p_dist[:5])):
        return "popup"
    elif(u <= np.sum(p_dist[:6])):
        return "single"
    elif(u <= np.sum(p_dist[:7])):
        return "double"
    elif(u <= np.sum(p_dist[:8])):
        return "triple"
    else:
        return "home run"

#function for simulating an at bat
def at_bat(outcomes, bases, outs):
    balls = 0
    strikes = 0
    #keep iterating until we reach an at-bat ending action
    while(True):
        #select pitch and location. this can be fixed, random, or chosen according to a policy
        em_location = em(outcomes, bases, outs)
        i = em_location[0]
        j = em_location[1]
        #4 is the horizontal and vertical standard deviation of error for where the ball may end up
        zone = imperfect_placement(i,j, 4, 4)
        p_dist = outcomes.get(zone)
        event = pitch_made(p_dist)
        if(event == "take"):
            if zone in strikezone:
                strikes += 1
                if(strikes == 3):
                    return "strikeout"                    
            else: 
                balls += 1
                if(balls == 4):
                    return "walk"
        elif(event == "whiff"):
            strikes += 1
        elif(event == "foul"):
            if(strikes < 2):
                strikes += 1
        else:
            return event

#simulate an inning of play given the outcomes dictionary
def inning(outcomes):
    score = 0
    outs = 0
    bases = 0
    #[walk,strikeout,popup,gb,1B,2B,3B,HR]
    props = np.zeros(8)
    #first = True
    while(outs <= 2):
        batter = at_bat(outcomes, bases, outs)       
        #depending on their outcome change outs and batting
        if(batter == "strikeout"):
            outs += 1
            props[1] += 1
        elif(batter == "walk"):
            props[0] += 1
            if bases == 0:
                bases = 1
            elif bases == 1:
                bases = 4
            elif bases == 2:
                bases = 4
            elif bases == 3:
                bases = 6
            elif bases in [4,5,6]:
                bases = 7
            else:
                score += 1               
        elif(batter == "ground ball"):
            props[3] += 1
            if bases in [0,2,3,5]:
                outs += 1
            elif bases == 1:
                outs += 2
                bases = 0
            elif bases == 4:
                outs += 2
                bases = 3
            elif bases == 6:
                outs += 2
                bases = 0
                if(outs <= 2):
                    score += 1
            else:
                outs += 2
                bases = 3
                if(outs <= 2):
                    score += 1
        elif(batter == "popup"):
            props[2] += 1
            if outs < 2:
                if bases in [0,1,2,4]:
                    outs += 1
                elif bases == 3:
                    outs += 1
                    score += 1
                    bases = 0
                elif bases == 5:
                    outs += 1
                    score += 1
                    bases = 2
                elif bases == 6:
                    outs += 1
                    score += 1
                    bases = 1
                else:
                    outs += 1
                    score += 1
                    bases = 4
            else:
                outs += 1
        elif(batter == "single"):
            props[4] += 1
            if bases == 0:
                bases = 1
            elif bases == 1:
                bases = 4
            elif bases == 2:
                bases = 1
                score += 1
            elif bases == 3:
                bases = 1
                score += 1
            elif bases == 4:
                score += 1
            elif bases == 5:
                bases = 1
                score += 2
            elif bases == 6:
                bases = 4
                score += 1
            elif bases == 7:
                bases = 4
                score += 2
        elif(batter == "double"):
            props[5] += 1
            if bases == 0:
                bases = 2
            elif bases == 1:
                bases = 6
            elif bases == 2:
                score += 1
            elif bases == 3:
                bases = 2
                score += 1
            elif bases == 4:
                bases = 5
                score += 1
            elif bases == 5:
                bases = 2
                score += 2
            elif bases == 6:
                bases = 5
                score += 1
            else:
                bases = 5
                score += 2
        elif(batter == "triple"):
            props[6] += 1
            if bases == 0:
                bases = 3
            elif bases in [1,2,3]:
                bases = 3
                score += 1
            elif bases in [4,5,6]:
                bases = 3
                score += 2
            else:
                bases = 3
                score += 3
        elif(batter == "home run"):
            props[7] += 1
            if bases == 0:
                score += 1
            elif bases in [1,2,3]:
                score += 2
                bases = 0
            elif bases in [4,5,6]:
                score += 3
                bases = 0
            else:
                score += 4
                bases = 0
    return score

#determines where the ball ends up given intended target (middle of zone (i,j) and std in inches in x&y)
def imperfect_placement(row,col, v_var, h_var):
    intended_x = 5.667*(col-2)
    intended_y = 5.667*(2-row)
    real_x = intended_x + np.random.normal(0,h_var)
    real_y = intended_y + np.random.normal(0,v_var)
    if((real_x < -14.1667) or (real_x > 14.1667) or (real_y < -14.1667) or (real_y > 14.1667)):
        return (5,5)
    if((real_x >= -14.1667) and (real_x < -8.4997)):
        c = 0
    elif((real_x >= -8.4997) and (real_x < -2.8327)):
        c = 1
    elif((real_x >= -2.8327) and (real_x < 2.8343)):
        c = 2
    elif((real_x >= 2.8343) and (real_x < 8.5013)):
        c = 3
    else:
        c = 4
    
    if((real_y >= -14.1667) and (real_y < -8.4997)):
        r = 4
    elif((real_y >= -8.4997) and (real_y < -2.8327)):
        r = 3
    elif((real_y >= -2.8327) and (real_y < 2.8343)):
        r = 2
    elif((real_y >= 2.8343) and (real_y < 8.5013)):
        r = 1
    else:
        r = 0
    return (r,c)

#sets a run value for each game state (outs, runners on base, at-bat outcome)
def value_maker(outcomes):
    vals = {}
    j=0
    actions = ["walk","strikeout","popup","ground ball","single","double","triple","home run"]
    for outs in range(3):
        for state in range(8):
            a_store = np.zeros(8)
            for a in range(8):
                print(j, " Value")
                j+=1
                val_store = np.zeros(10000)
                for i in range(10000):
                    val_store[i] = inning(outcomes, outs, state, actions[a])[0]
                mean_val = np.mean(val_store)
                a_store[a] = mean_val
            vals[(state,outs)] = a_store
    return vals

def batting_result(outcomes, place, v_var, h_var):
    balls = 0
    strikes = 0
    while(True):
        coordinates = coordinates_dic.get(place)
        zone = imperfect_placement(coordinates[0], coordinates[1], v_var, h_var)
        p_dist = outcomes.get(zone)
        event = pitch_made(p_dist)
        if(event == "take"):
            if zone in strikezone:
                strikes += 1
                if(strikes == 3):
                    return "strikeout"                    
            else: 
                balls += 1
                if(balls == 4):
                    return "walk"
        elif(event == "whiff"):
            strikes += 1
            if(strikes == 3):
                return "strikeout"
        elif(event == "foul"):
            if(strikes < 2):
                strikes += 1
        else:
            return event
    
#computing the proportions of each batted ball outcome at specified level of control variance
def outcome_proportions(outcomes, zone, v_var, h_var):
    outcome_props = np.zeros(8)
    for n in range(10000):
        batted_outcome = batting_result(outcomes, zone, v_var, h_var)
        if (batted_outcome == "walk"):
            outcome_props[0] += 1
        elif (batted_outcome == "strikeout"):
            outcome_props[1] += 1
        elif (batted_outcome == "popup"):
            outcome_props[2] += 1
        elif (batted_outcome == "ground ball"):
            outcome_props[3] += 1
        elif (batted_outcome == "single"):
            outcome_props[4] += 1
        elif (batted_outcome == "double"):
            outcome_props[5] += 1
        elif (batted_outcome == "triple"):
            outcome_props[6] += 1
        elif (batted_outcome == "home run"):
            outcome_props[7] += 1
    return outcome_props/10000
    
#for 3x3 discretized strike zone, find the proportion of at-bat outcomes for each of the 9 boxes
def fill_end_props(outcomes):
    global end_props
    end_props = np.zeros((9,8))
    belief_vert_control = 3.25
    belief_hor_control = 3.25
    for i in range(9):
        end_props[i] = outcome_proportions(outcomes, i, belief_vert_control, belief_hor_control)
    return end_props

#determine what is the best spot to throw to given the outs and runners on base alignment (from proportion of different at-bat outcomes and their associated values in state_val)
def em(outcomes, bases, outs):
    v = np.array(state_val.get((bases, outs)))
    expected = np.zeros(9)
    for count, p in enumerate(end_props):
        expected[count] = p@v
    spot = np.argmin(expected)
    return (int(spot/3)+1,spot%3 + 1)