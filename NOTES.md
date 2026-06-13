# NOTES:
- early stopping?
  - dlaczego drugi vae ma VAE Training:  56%|█████▌    | 28/50 [28:11<19:35, 53.44s/it, kl=94.465, loss=819.807, val=831.978]    ?
  - jak pierwszy ma VAE Training: 100%|██████████| 30/30 [28:24<00:00, 56.83s/it, kl=40.924, loss=229.454, val=230.565]
  - pewnie bo image size
- dodać więcej wykresów do raportów jeśli możliwe
- a co z test setem?
- zmniejszyć liczbę kolorów


# NOWE TODO:
- dla na najlepszych hiperparametrów dodać jeszcze jakiś szake czy lekkie przesunięcia, aby mieć jeszcze więcej danych


# Do raportu:
- najpierw dać claudowi, aby przeszedł przez raporty wszystkich eksperymentów, które już przeprowadziłem i wyciągnął wnioski nt wpływu hiperparametrów
- potem rozpisać notebook, który puści testy grida hiperparametrów; zastanowić się których; ile wgl trwały testy na nowym datasecie? chyba ktrótko
- opisać dataset 1 vs dataset 2, jak poszło z pierwszym, przykłady prób, do jakich wniosków doszliśmy, próby redukcji kolorów, zmiana datasetu, rezultaty na nowym datasecie (w sekcji dataset tylko wspomniany, że został użyty i napisać, że więcej o nim w sekcji x...)
- czyli opisać to jako eksperyment redukcji kolorów? czy eksperyment zmiany datasetu?
- w jakiej sekcji (background?) opisać problem mode colapse i gdzieś tam napisać, że nie wystąpił u nas
- jak opisuję rezultaty to posłużyć się zarówno FID jak i oceną subiektywną
- eksperyment z intormpolacją

# Inne notatki do raportu
- dodać opis jak uruchomić testy
- może opis parametrów interfejsu mojego?
- wypisać wersje bibliotek, albo napisać, że są one w pliku toml
