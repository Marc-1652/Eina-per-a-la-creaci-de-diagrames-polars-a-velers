Generador de Polars per a Vaixells a Vela

Aquest projecte és una eina educativa per processar dades de navegació i generar diagrames polars. El codi llegeix fitxers CSV amb dades de velocitat i angles del vent, crea taules dinàmiques i genera un gràfic polar amb suavitzat parabòlic. Aquest projecte s'ha desenvolupat amb fins educatius.

Característiques

    Processament de dades:
        Llegeix només les columnes necessàries dels fitxers CSV.
        Filtra les dades en funció d'un mínim de velocitat real del vent.
        Converteix la velocitat de m/s a nusos.
        Crea dues versions dels angles del vent:
            TWA_raw (deg): arrodonit a l'unitat (per a la taula VMG).
            TWA (deg): arrodonit al múltiple de 5 (per a la taula SOG i el diagrama polar).

    Generació de taules:
        Taula SOG: calcula la percentil 95 de SOG per cada combinació d'angle (TWA) i TWS.
        Taula VMG: troba l'angle de millor VMG per a rumbs obert i tancats a les dues amures.
        Taula polar: per a la representació gràfica, utilitzant TWA (deg).

    Diagrama Polar:
        Representa el SOG màxim en funció de l'angle del vent.
        Aplica un ajust parabòlic per suavitzar la corba.
        Inclou funcionalitat interactiva per mostrar valors al passar el cursor.

    Interfície Gràfica (GUI):
        Permet seleccionar la carpeta amb fitxers CSV.
        Opcions per exportar taules i desar el diagrama com a imatge PNG.
        Botó per reiniciar el processament de dades.

Requisits

    Python 3.x
    Llibreries:
        tkinter (inclòs amb Python)
        pandas
        numpy
        matplotlib
        mplcursors

Ús

    Executa el codi:

    python catala_v2.py

    Selecciona la carpeta amb els fitxers CSV que contenen les dades de navegació.

    Un cop processades les dades, utilitza la GUI per:
        Exportar les taules SOG i VMG en format CSV.
        Visualitzar el diagrama polar.
        Desa el diagrama polar com a imatge PNG.

    Utilitza el botó "Reinicia" per esborrar les dades processades i començar un nou processament.

Nota

Aquest projecte està destinat a ús educatiu dins del Treball de Fi de Grau (TFG). És una eina bàsica que pot servir com a base per a futurs estudis.
