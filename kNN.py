import pandas as pd
import numpy as np
import random
import time


# Funkcja znajduje miasto najblizsze aktualnemu
def find_nearest_city(dist_matrix, left_cities_names, act_city_name):
    left_cities = left_cities_names # miasta, w ktorych jeszcze nie bylismy
    actual_city = act_city_name # miasto, w ktorym jestesmy aktualnie
    distances = dist_matrix[:, [0, actual_city]] # bierzemy tylko kolumne z indeksami i odległościami od aktualnego miasta do pozostałych
    distances = distances[left_cities, :] # macierz odleglosci miedzy miastem w ktorym jestesmy aktualnie, a miastami jeszcze nieodwiedzonymi
    next_city = distances[distances[:, 1] == np.min(distances[:, 1]), :] # szukamy miasta w najblizszej odleglosci
    next_city_name = next_city[0,0] # pierwszy element to nr miasta
    dist_to_next_city = next_city[0,1] # drugi element do odleglosc miedzy znalezionym a aktualnym miastem
    return next_city_name, dist_to_next_city


def kNN(dist_matrix, start_city):
    distances_matrix = dist_matrix.astype(int) # macierz odleglosci miedzy miastami
    num_cities = distances_matrix.shape[0] - 1 # liczba miast
    first_city = start_city
    #first_city = random.randint(1, num_cities) # losujemy miasto startowe
    actual_city = first_city # miasto startowe jako aktualne miasto
    cities_order = [[actual_city, 0]] # zapisujemy kolejnosc odwiedzanych miast (1sza kolumna) oraz odleglosc miasta od poprzedniego miasta (2ga kolumna)
    visited_cities = [actual_city] # miasta, w ktorych juz bylismy
    left_cities = distances_matrix[0, 1:]
    left_cities = np.delete(distances_matrix[0, 1:], np.where(left_cities == visited_cities)) # miasta, w ktorych jeszcze nie bylismy

    while len(left_cities) > 1: # do poki wszystkie miasta nie zostana odwiedzone
        next_city = find_nearest_city(distances_matrix, left_cities, actual_city) # szukanie miasta bedacego najblizej aktualnego miasta
        # print("Next city:", next_city)
        visited_cities = np.append(visited_cities, next_city[0]) # dolaczenie znalezionego miasta do juz odwiedzonych
        left_cities = left_cities[left_cities != next_city[0]] # usuniecie znalezionego miasta z listy miast, w ktorych nie bylismy
        cities_order = np.vstack([cities_order, next_city])  # dolaczenie nowego miasta do tablicy kolejnosci odwiedzanych miast
        actual_city = next_city[0] # ustawienie nowego miasta jako aktualnego, w ktorym jestesmy

        # print("Cities order:", cities_order)
        # print("Visited", visited_cities)
        # print("Left", left_cities)

    last_city = [left_cities[0], distances_matrix[int(left_cities[0]), int(visited_cities[0])]] # wyszukanie ostatniego miasta i odleglosc miedzy nim a pierwszym miastem
    cities_order = np.vstack([cities_order, last_city]) # dolaczenie ostatniego miasta do tablicy kolejnosci odwiedzania miast

    return cities_order[:,0], sum(cities_order[:,1]) # zwraca kolejnosc odwiedzanych miast oraz laczny dystans do pokonania


input_file_name = 'Dane_TSP_127.xlsx'
output_file_name = 'kNN_127.csv'

data = pd.read_excel(input_file_name, index_col=None, header=None)

distances_matrix = np.array(data)
distances_matrix[0,0] = 0
cities_n = len(distances_matrix)
print(cities_n)

results = []

start_time = time.time()

for i in range (1, cities_n):
    temp_res = kNN(distances_matrix, i)
    results.append([[', '.join(str(v) for v in temp_res[0])], temp_res[1]])

print(time.time() - start_time)

results_df = pd.DataFrame(results, columns=['cities order', 'total distance'])
results_df.to_csv(output_file_name, sep=';', mode='a')
