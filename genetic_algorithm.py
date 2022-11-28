import numpy as np
import pandas as pd
import time as t
import random

czas_rozpoczecia = t.time()


# df - ramka danych, do_spr - lista z indeksami zadań w jaki sposób mają być ułożone, l_masz - liczba maszyn
def czas_zakonczenia(df, do_spr, l_masz):
    ile_zadan = len(do_spr)
    wynikowa = np.zeros((ile_zadan, l_masz))
    czas_zak = 0

    # obliczenie czasu zakonczenia
    for i in range(0, ile_zadan):
        for j in range(0, l_masz):
            wiersz = df[int(do_spr[i]), 1:np.shape(df)[1]]

            if (i == 0 and j == 0):
                wynikowa[i, j] = wiersz[j]
            elif i == 0:
                wynikowa[i, j] = wynikowa[i, j - 1] + wiersz[j]
            elif j == 0:
                wynikowa[i, j] = wynikowa[i - 1, j] + wiersz[j]
            else:
                wynikowa[i, j] = max(wynikowa[i - 1, j], wynikowa[i, j - 1]) + wiersz[j]

    czas_zak = wynikowa[ile_zadan - 1, l_masz - 1]
    return czas_zak


def turniej2(array):
    parent1 = comp(array)
    # print(parent1)
    parent2 = comp(array)
    # print(parent2)

    if (parent1 == parent2).all():  #
        # new_arr = np.delete(array,np.where(parent2==array),axis=0)
        parent2 = comp(array)
    return parent1, parent2


def comp(array):
    select = []
    parents = []
    w = 3
    select = np.array(random.choices(array, k=w))
    # print(select)
    num_rows, num_cols = select.shape
    best = select[0, num_cols - 1]
    for i in range(0, num_rows):
        if (best >= select[i, num_cols - 1]):
            best = select[i, num_cols - 1]
            par1 = select[i, :-1]
    # print(par1)
    return par1


def ranking(array):
    rows, cols = array.shape
    n = rows
    dystribution = []
    based_value = 0.0
    # sortowanie malejace wg ostatniej kolumny, ktora musi byc czasem zakonczenia
    sorting = array[np.argsort(array[:, -1])]
    sorted_array = sorting[::-1][:n]
    # tworzenie dystrybuanty
    for i in range(1, rows + 1):
        if i == 1:
            pr = (i) / sum(range(1, rows + 1))
        else:
            pr = (i) / sum(range(1, rows + 1)) + dystribution[i - 2]
        dystribution.append(pr)

    return dystribution, sorted_array


# funkcja zamienia dwa zadania miejscami
def mutacja(lista_zad):
    indeksy = random.sample(list(lista_zad), 2)
    indeks_i = int(indeksy[0])
    indeks_2 = int(indeksy[1])
    zadania = np.copy(lista_zad)
    zadania[indeks_i], zadania[indeks_2] = zadania[indeks_2], zadania[indeks_i]
    return zadania


def OX(parent_1, parent_2):
    p_1 = np.copy(parent_1)
    p_2 = np.copy(parent_2)

    section_start = random.randint(1, len(p_1) - 2)  # indeks pierwszej cyfry w sekcji kojarzenia
    section_end = random.randint(section_start, len(p_1) - 2)  # indeks ostatniej cyfry w sekcji kojarzenia

    # skopiowanie wartosci macierzy rodzicow do macierzy dzieci
    child_1 = np.copy(p_1)
    child_2 = np.copy(p_2)

    # segmenty kopiowane bez zmian do potomkow
    section_1 = child_1[section_start:section_end + 1]
    section_2 = child_2[section_start:section_end + 1]

    # print("sec_start:", section_start)
    # print("sec_end:", section_end)

    # zmienne pomocnicze do iteracji po macierzach
    i, j = section_end + 1, section_end + 1
    m, n = section_end + 1, section_end + 1

    # tyle powtorzen petli ile zadan trzeba jeszcze skopiowac z drugiego rodzica (miejsca poza skopiowanym juz segmentem)
    for k in range(0, len(p_1) - len(section_1)):

        # wybieramy kolejne zadanie od drugiego rodzica
        curr_obj_1 = p_2[j]
        j += 1

        curr_obj_2 = p_1[n]
        n += 1

        # jesli zadanie istnieje juz w skopiowanym segmencie u dziecka to idziemy do nastepnego,
        # do poki nie znajdziemy takiego, ktorego nie ma w segmencie
        while curr_obj_1 in section_1:
            if j >= len(p_1):
                j = 0
            curr_obj_1 = p_2[j]
            j += 1

        while curr_obj_2 in section_2:
            if n >= len(p_1):
                n = 0
            curr_obj_2 = p_1[n]
            n += 1

        # przypisujemy nowe zadanie na rozwazanej pozycji
        child_1[i] = curr_obj_1
        child_2[m] = curr_obj_2

        # przechodzimy do uzupelniania kolejnej pozycji
        i += 1
        m += 1

        # jesli osiagnelsismy koniec ktoregos z ciagow, przechodzimy do poczatku ciagu
        if i >= len(p_1):
            i = 0

        if m >= len(p_1):
            m = 0

        if j >= len(p_1):
            j = 0

        if n >= len(p_1):
            n = 0

        k += 1

    return child_1, child_2


def ewolucyjny_trening_OX(df, niepotrzebne_zmienne, poprzedni_ID, wielkosc_populacji, epoch_max, prawdopodob_mutacji,
                          nazwa_pliku):
    epoch = 1
    zachowanie_ID = pd.DataFrame(data=np.array([df[poprzedni_ID], range(0, np.shape(df)[0])]).T,
                                 columns=["Stare ID", "Nowe_ID"])

    df = df.drop(niepotrzebne_zmienne + [poprzedni_ID], axis=1)
    l_zadan = np.shape(df)[0]  # Liczba zadan
    l_maszyn = np.shape(df)[1]  # Liczba maszyn
    czas_zak_pop = np.zeros(wielkosc_populacji)

    df["ID"] = range(0, l_zadan)  # tworzy sztucznie ID dla każdego zadania od 0
    # l_zadan=10

    macierz_populacji = np.zeros((wielkosc_populacji, l_zadan))

    # przesunięcie kolumny ID na początek
    df = df.reindex(columns=["ID"] + list(df.columns)[0:-1])

    # Zmienne do funkcji
    zadania = list(df["ID"])[0:l_zadan]  # Lista ID wsyztskich zadań

    df = np.array(df)
    powt_algorytmu = 0

    while (powt_algorytmu < 10):
        # trowzymy pierwsza losowa populacje
        for i in range(0, wielkosc_populacji):
            macierz_populacji[i, :] = random.sample(zadania, len(zadania))  # losujemy ułożenie zadań!!
            czas_zak_pop[i] = czas_zakonczenia(df, macierz_populacji[i, :], l_maszyn)
        min_czas = czas_zak_pop[0]
        minimalne_ulozenie = macierz_populacji[0, :]

        nazwy = [el for el in range(1, l_zadan + 1)]
        for el in nazwy:
            nazwy[el - 1] = "Ulozenie" + str(el)

        while epoch < epoch_max:

            # tworzymy kolejna populacje
            nowa_pop = np.zeros((wielkosc_populacji, l_zadan))
            # print(nowa_pop)
            wielkosc_populacji_rodzicow = wielkosc_populacji  # tu mozna dac inna liczbe
            nowa_pop_rodzicow = np.zeros((wielkosc_populacji_rodzicow, l_zadan))
            rodzice_i_czas = np.c_[macierz_populacji, czas_zak_pop]  # do turnieju
            # print(sorted)
            # rodzice_i_czas[:,wielkosc_populacji]=czas_zak_pop
            # print((ranking_interwaly))
            for i in range(1, wielkosc_populacji_rodzicow, 2):  # tu mozna dac wiecej
                nowa_pop_rodzicow[i - 1, :], nowa_pop_rodzicow[i, :] = turniej2(rodzice_i_czas)

            # print(macierz_populacji,"\n",nowa_pop_rodzicow)
            for i in range(1, wielkosc_populacji_rodzicow, 2):

                nowa_pop[i - 1, :], nowa_pop[i, :] = OX(nowa_pop_rodzicow[i - 1, :], nowa_pop_rodzicow[i, :])
                los = random.uniform(0, 1)
                if (los < prawdopodob_mutacji):
                    nowa_pop[i - 1,] = mutacja(nowa_pop[i - 1, :])
                    # print("Mutacja")
                los = random.uniform(0, 1)
                if (los < prawdopodob_mutacji):
                    nowa_pop[i,] = mutacja(nowa_pop[i, :])
                    # print("Mutacja")

            czas_zak_nowa_pop = np.zeros(wielkosc_populacji)

            # liczymy czasy zakonczenia nowej populacji
            for i in range(0, wielkosc_populacji):
                czas_zak_nowa_pop[i] = czas_zakonczenia(df, nowa_pop[i, :], l_maszyn)

            # nowa populacja
            rodzice_i_czas = np.c_[nowa_pop, czas_zak_nowa_pop]  # do rankigu
            rodzice_i_czas = rodzice_i_czas[np.argsort(rodzice_i_czas[:, -1])]
            macierz_populacji = np.copy(rodzice_i_czas[0:wielkosc_populacji, :-1])

            czas_zak_pop = rodzice_i_czas[:, -1]
            # liczymy czasy zakonczenia nowej populacji
            # for i in range(0,wielkosc_populacji):
            #     czas_zak_pop[i]=czas_zakonczenia(data,macierz_populacji[i,], l_maszyn)
            if min_czas > czas_zak_pop[0]:  # od 0 bo posortowane sa rosnaco
                min_czas = czas_zak_pop[0]
                minimalne_ulozenie = macierz_populacji[0, :]
            epoch += 1

            print("Przejscie algorytmu: ", str(powt_algorytmu + 1), "\tWykonano: ",
                  np.round(epoch / epoch_max * 100, 2), "%", "\tCzas obliczeń: ",
                  np.round(t.time() - czas_rozpoczecia, 4), sep="")
        # print(['Powtórzenie algorytmu','Iteracje', 'Czas zakonczenia', 'Prawdopodobieństwo mutacji']+nazwy)
        # print(np.matrix([[powt_algorytmu,epoch,min_czas,prawdopodob_mutacji]+list(minimalne_ulozenie)]))
        powt_algorytmu += 1

        if 'df_csv' in locals():
            d = pd.DataFrame(data=np.matrix([[powt_algorytmu, epoch, wielkosc_populacji, prawdopodob_mutacji,
                                              min_czas] + list(minimalne_ulozenie)]))
            df_csv = df_csv.append(d)
        else:
            df_csv = pd.DataFrame(data=np.matrix([[powt_algorytmu, epoch, wielkosc_populacji, prawdopodob_mutacji,
                                                   min_czas] + list(
                minimalne_ulozenie)]))  # ,columns=['Powtórzenie algorytmu','Epochi','Wielkosc populacji', 'Prawdopodobieństwo mutacji','Czas zakonczenia']+nazwy)

    df_csv.to_csv('ewolucyjny_OX_turniej_' + nazwa_pliku + '_.csv', mode='a', header=False, index=False)


if __name__ == '__main__':
    # wczytanie danych
    data = pd.read_excel("Dane_S2_200_20.xlsx")
    print(data)
    zmienne_do_wyrzucenia = []  # argument, inne niż ID
    stary_ID = "Zadanie"

    for populacja in [50, 100, 150]:
        for epochi in [500, 1000, 2500]:
            for p_mut in [0.1, 0.2, 0.3]:
                ewolucyjny_trening_OX(data, zmienne_do_wyrzucenia, stary_ID, wielkosc_populacji=populacja,
                                      epoch_max=epochi, prawdopodob_mutacji=p_mut, nazwa_pliku="200_20")



