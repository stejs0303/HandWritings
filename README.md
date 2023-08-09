# Program
- Python verze 3.10.7.
- Všechny používané knihovny pro systém Linux se nachází v *requirements.txt*.

# Cesta
- Pro funkčnost některých skriptů je třeba změnit ```WORKING_DIRECTORY``` v **Handwriting_scripts/config.py** na cestu ke složce **hand_writings**.

# Předtrénované modely
- Předtrénované modely nebylo možné nahrát při odevzdávání složky diplomové práce. *(limit .zip souboru je 15 MB)*
- Lze je nalézt na školním tesla serveru ve složce **/home/xstejs31/hand_writings/Keepers**. Ta obsahuje podsložky pro jednotlivé modely s výsledky trénování, nenatrénovaným modelem a nejlepší epochou.
- Podsložky jsou pojemovány jako *{název modelu}*\_*{rozlišení fotografií}*\_*{použitý optimizer}*. *(Adam - euclidian distance, SGD - fully connected)*


# Předzpracování obrázků - augment_photos.py
1. Vložit složku s originálními obrázky do **Photos/paragraphs/original/**.
2. Přidat název složky do ```folders=[]``` ve funkcích ```generate_downscaled_photos()``` a ```generate_augmented_photos()```.
3. Ujistit se, že je ```MADE_CHANGES = True```.
4. V **Handwriting_scripts/config.py** nastavit požadované rozlišení fotografií ```WIDTH, HEIGHT```, lze také nastavit hodnotu ```INVERT```, která invertuje barevný prostor obrázků. *(černý na bílem > bílý na černém)* 
5. Spustit skript.

- Ve složce **Photos/paragraphs/all/** přibudou složky *nazev* a *nazev*_augmented s vygenerovanými obrázky.


# Tvorba datasetů - create_datasets.py
1. Přidat názvy vygenerovaných složek do ```folders=[]``` ve funkci ```create_pairs()```.
2. Nastavit počet osob pro validační a testovací set.
3. Spustit skript.

- Do složky **Photos/** přibude 5 *.csv* souborů pro trénování, testování, validaci, cross validaci a všechny ukázky v jednom souboru.
- Je předpokládáno, že každá z osob bude mít 4 ukázky. Pokud jich je víc, je třeba změnit konstantu *(momentálně nastaven na 4\*\*2)* ve funkci ```split()``` pro parametr ```test_size=```.


# Tvorba modelů - *siamese_\*.py*
- Pro vytvoření stačí zavolat funkci modelu v ```if __name__=="__main__":``` a spustit skript.
- Model bude uložen do složky **Models/**.


# Trénování - train_model.py
- Pro změnu datasetu lze změnit první předávaný parametr v ```training_generator = ParallelDataLoader()``` na část názvu některého *.csv* souboru v **Photos/**. Funkce z **Handwriting_scripts/dataset_loader.py** načtou všechny *.csv* soubory, vyfiltrují podle části názvu ty správné a všechny *.csv*, které obsahují zmíněnou část, jsou použity.

1. Pro jeden model:
    * v konzoli spustit:
    * ```python3 Handwriting_scripts/train_model.py --gpu *default None* --epochs *default 1* --verbose *default 1* --batch_size *default 16* --model *default None*```
    * lze specifikovat ```--init_epoch *default 0*``` pro pokračování v číslování od dané epochy
    * do ```--model``` je třeba zadat celou cestu: *Models/model.h5*
2. Pro více modelů:
    * lze spustit ./run_training.sh
    * natrénuje všechny modely ve složce *Models/* na gpu 1, na prvních 30 epoch
    * výsledky jsou ukládány do *Results/model.out*

- Modely jsou po každé epoše uloženy do *Models/model/epoch_\*.h5*.


# Testování - test_model.py
- V konzoli spustit:
- ```python3 Handwriting_scripts/test_model.py --gpu *default None* --verbose *default 1*  --model *default None*```.
- Do ```--model``` je třeba zadat celou cestu: *Models/model.h5*.
- Po testování skript vypíše chybovost pro stejné ukázky, různé ukázky a celkovou chybovost. *(100 % - chybovost = přesnost)*


# Attention mapy - attention_maps.py
- Momentálně jsou implementovány tvorby map pro modely VGG16 a ResNet18.
- Výsledné obrázky jsou ukládány do **Attention_maps/** do složkové struktury.
- Je zpracován vždy pouze obrázek jehož umístění je uloženo v proměnné ```img_a_path```.
- V **Handwritings_scripts/config.py** lze nastavit ```FULL_RESOLUTION``` - všechny vygenerované mapy budou v rozlišení vstupní fotografie a ```GENERATE_FILTER_MAPS``` - hodnoty všech filtrů budou jak průměrovány *(výchozí stav)*, tak ukládány zvlášť.