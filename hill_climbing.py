import pandas as pd
import numpy as np
import random
import time


# funkcja zwraca dwa losowe indeksy wierszy
def rand_rows(rows_n):
    return random.sample(range(0, rows_n), 2)

# funkcja zamienia wiersze w tablicy miejscami
def swap_rows(array, x, y):
    a = array
    a[[x, y]] = a[[y, x]]
    return a

#funkcja oblicza wartosc funkcji celu (ostatnia komorka w macierzy czasow)
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


def hill_climbing(arr, iterations=100000, max_no_progress=1000):
    curr_arr = arr  # macierz zadania x maszyny
    rows = len(curr_arr)  # ilosc wierszy (liczba zadan)
    cols = len(curr_arr[0])  # ilosc kolumn (liczba maszyn)
    no_progress_n = 0 # zmienna liczy przez ile iteracji wartosc funkcji celu nie poprawila sie
    max_i = iterations # przez ile iteracji ma byc powtarzany algorytm

    # delete usuwa pierwsza kolumne z indeksami
    # obliczamy wartosc funkcji celu w tabeli wyjsciowej
    obj_f = objective_function(np.delete(np.array(curr_arr), obj=0, axis=1), rows, cols-1)

    # szukamy rozwiazania przez wskazana ilosc iteracji
    for i in range(max_i):

        random_i = rand_rows(rows) # losujemy 2 zadania do zamiany
        new_arr = swap_rows(curr_arr, random_i[0], random_i[1]) # tworzymy sasiedztwo, w ktorym podmieniamy kolejnoscia wczesniej wylosowane 2 wiersze
        new_obj_f = objective_function(np.delete(np.array(new_arr), obj=0, axis=1), rows, cols-1) # obliczamy wartosc funkcji celu dla nowego sasiedztwa

        # sprawdzamy czy wartosc funkcji celu poprawila sie
        # jesli tak, to bierzemy nowe sasiedztwo jako bazowe
        if new_obj_f == obj_f:
            no_progress_n = no_progress_n + 1
            curr_arr = new_arr

        elif new_obj_f > obj_f:
            no_progress_n = no_progress_n + 1 # zwiekszamy, jesli funkcja celu nie poprawila sie

        elif new_obj_f < obj_f:
            no_progress_n = 0
            curr_arr = new_arr
            obj_f = new_obj_f

        # jesli zbyt dlugo wartosc funkcji celu nie ulegla poprawie, konczymy dzialanie algorytmu
        if no_progress_n >= max_no_progress:
            # print("To many steps without decline of the objective function value. It seems that algorithm can not improve the current solution.")
            return curr_arr[:, 0], obj_f

    #zwracamy kolejnosc zadan i obliczony czas
    return curr_arr[:, 0], obj_f


iterations_n=[5000, 10000, 25000, 100000]
no_progress_n=[1000, 1500, 2000, 2500]

data = pd.read_excel("Dane_S2_200_20.xlsx")
data = np.array(data)

start_time = time.time()
res =[]

for h in range (0, 30):
    print(h)
    np.random.shuffle(data) # tasujemy zadania
    for i in range(0, len(iterations_n)):
        for j in range(0, len(no_progress_n)):
            temp_res = hill_climbing(data, iterations_n[i], no_progress_n[j])
            res.append([iterations_n[i], no_progress_n[j], [', '.join(str(v) for v in temp_res[0])], temp_res[1]])
            h=+1

pf = pd.DataFrame(res, columns=['iteracje', 'bez_poprawy', 'kolejnosc_zadan', 'f_celu'])
print(time.time() - start_time)

pf.to_csv('hill_climbing_200_20.csv', sep=';', mode='a')