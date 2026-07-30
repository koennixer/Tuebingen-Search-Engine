### Search Result Presentation
Basic presentation with webserver works

TODOS:
- "invention" (we should add some feature that is not just the blue-line links)
    - link/embedd osm (explore Tübingen on osm, maybe mark out results on map, but would require good results and their address)
        - leaflet
    - button leading to page with flyers from the city about Tübingen (maybe like: new in Tübingen or just visiting? view these Flyers by the city)
        - https://www.tuebingen.de/Dateien/broschuere_willkommen_englisch.pdf, https://www.tuebingen-info.de/_Resources/Persistent/5b64fd3ecc0c244207d301b60db28ba0e33a1a4a/Tour_of_the_city_2022.pdf
    - checkbox to exclude uni-tuebingen.de websites from the results
        - just have less than 100 results then or somehow filter out before and have up to 100 results without uni websites


optional:
- make it look nicer
- cool name

DONE:
- do not start the web server right away, wait on command in command line interface
+ run web command inside CLI or directly (python3 tuebingen_search_engine.py web) with(out) arguments (port, host)

-move web UI code to a separate file

- Additional page (or another solution) for the batch query examination
+ Use the attach at the search bar to upload a .txt/.tsv file of queries. Then run and see/download the results.

- add clickable options in the web UI to select if/what scores should be shown

