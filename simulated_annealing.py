import pandas as pd
import numpy as np
import random
import time
import math

# funkcja zwraca dwa losowe indeksy wierszy bez powtorzen
def rand_rows(rows_n):
    return random.sample(range(0, rows_n), 2)

# funkcja zamienia wiersze w tablicy miejscami
def swap_rows(array, x, y):
    a = array
    a[[x, y]] = a[[y, x]]
    return a

# funkcja obniżająca temperature
def slow_temp_decrese(temp):
    new_temp=temp/(1+2*temp)
    return new_temp

def linear_temp_decrease(temp):
    new_temp=temp-0.3
    return new_temp

#funkcja oblicza wartosc funkcji celu (ostatnia komorka)
def objective_function(array, rows_n, columns_n):
    rows = rows_n
    cols = columns_n

    for x in range(1, rows):
        array[x, 0] = array[x, 0] + array[x - 1, 0]

    for y in range(1, cols):
        array[0, y] = array[0, y] + array[0, y - 1]

    for x in range(1, rows):
        for y in range(1, cols):
            array[x, y] = array[x, y] + max(array[x - 1, y], array[x, y - 1])

    return array[rows-1,cols-1]


def simulated_annealing(arr, max_i=100000, max_no_progress=1000, temp=50.0, temp_dec_fun="linear"):
    curr_arr = arr
    rows = len(curr_arr)
    cols = len(curr_arr[0])
    no_progress_n = 0 # zmienna liczy przez ile iteracji wartosc funkcji celu nie poprawila sie


    # delete usuwa pierwsza kolumne - Z1,Z2...
    # obliczamy wartosc funkcji celu w tabeli wyjsciowej
    obj_f = objective_function(np.delete(np.array(curr_arr), obj=0, axis=1), rows, cols-1)
    ap=[]
    f=[]
    # szukamy rozwiazania przez wskazana ilosc iteracji
    for i in range(max_i):

        random_i = rand_rows(rows) # losujemy 2 wiersze
        new_arr = swap_rows(curr_arr, random_i[0], random_i[1]) # tworzymy tabele, w ktorej podmieniamy kolejnoscia wczesniej wylosowane 2 wiersze
        new_obj_f = objective_function(np.delete(np.array(new_arr), obj=0, axis=1), rows, cols-1) # obliczamy wartosc funkcj icelu dla nowej tabeli
        # ap.append(curr_arr[:,0])
        # f.append(new_obj_f)
        # ap.append([new_obj_f, curr_arr[:,0]])
        # print(ap)

        # sprawdzamy czy wartosc funckji celu jest mniejsza
        if new_obj_f < obj_f:
            no_progress_n = 0
            curr_arr = new_arr
            obj_f = new_obj_f

        elif random.uniform(0,1) < math.exp(-(new_obj_f-obj_f)/temp):
            no_progress_n = no_progress_n + 1
            curr_arr = new_arr
            obj_f = new_obj_f

        else:
            no_progress_n = no_progress_n + 1

        # jesli zbyt dlugo wartosc funkcji celu nie ulegla poprawie, konczymy dzialanie algorytmu
        if no_progress_n >= max_no_progress:
            print("To many steps without decline of the objective function value. It seems that algorithm can not improve the current solution.")
            return curr_arr[:, 0], obj_f
        # print(temp)

        if temp_dec_fun == "slow":
            temp = slow_temp_decrese(temp)
        elif temp_dec_fun == "linear":
            temp = linear_temp_decrease(temp)
        else:
            print("Unrecognized temperature decreasing function.")
    return curr_arr[:, 0], obj_f #, ap,f # funckja zwraca 3 arg - kolejnosc zadan i wartosc f celu, f_celu i kolejnosc dla kazdej iteracji

# data = pd.read_csv("SzerPerm.csv", sep=';')
# hc = hill_climbing(np.array(data))
# print(hc[1]) # wartosc funkcji celu
# print(hc[0]) # kolejnosc zadan dla obliczonej f celu

data = pd.read_csv("Dane_S2_50_10.csv", sep=';')
data = np.array(data)

iterations_n=[1000, 5000, 10000, 100000]
no_progress_n=[1000, 5000, 10000, 50000]
temperature_n=[25,50,75,80]

start_time = time.time()
dat =[]
dec_fun = 'linear' #'slow'

for h in range(0,1):
    print(h)
    for i in range(0, len(iterations_n)):
        for j in range(0, len(no_progress_n)):
            for k in range(0,len(temperature_n)):
                temp_res = simulated_annealing(data, iterations_n[i], no_progress_n[j], temperature_n[k], dec_fun)
                dat.append([iterations_n[i], no_progress_n[j],temperature_n[k], dec_fun,
                [', '.join(str(v) for v in temp_res[0])], temp_res[1]])

                # dat.append([iterations_n[i], no_progress_n[j], temperature_n[k], dec_fun,
                #             temp_res[2],temp_res[3]])
                h=+1

pf = pd.DataFrame(dat, columns=['iteracje','bez poprawy', 'temp', 'f wygaszania', 'kolejnosc zadan', 'f celu'])
print(time.time() - start_time)

pf.to_csv('simulated_annealing_data.csv', sep=';', mode='a')

