# The big plan
	* Uporabimo dis3 za generacijo mapa
	* Združimo dis3 in dis4, za premikanje po mapi
	* Ustvarimo svoj predefined path
	* Uporabimo dis3 za face-detection
	* Izracunamo globalno pozicijo obraza in poslejmo v rviz marker
	* Hranimo lokacije videnih obrazov, in dodajamo nove, ce so dost stran
	* Ob gotovi zaznavi obraza (treshold) dodamo nesldnji waypoint do obraza, da ga gremo pozdravit
	* Ko najdes obraz, izracunas globalno pozicijo obraza, potem z laserja najdes dve relativno bliznji tocki na steni, in iz njih izracunas normalo stene
		Normala je pomembna, za tresholding, racunanje odmika od stene, ...
	* Pozdravimo obraz z neko python knjiznico
	* Ustavimo se, ko pozdravimo 3 obraze

# TODO:
	* V detect people mogoce malo se povecat sigurnost zaznave, ker se vcasih zgodi, da en valj yolo zazna kot obraz.
	* Na novo keypointe naredit.
	* Neki naredit, da se robot ne zatakne...
	* Ko zaznamo obraz, bi blo dobro takoj prepisat trenuten goal in it k obrazi, se posebej, ce je obraz na poti do nasega goala.
	  lahhko se sicer zgodi, da smo zaznani obraz, ki je kr dalec stran, v tem primeru bi blo boljse it prvo do goala ...
	* Popravit je treba nas keypoint seznam, da se ne na koncu vedno obraca k theta=0
	* Dodat je treba timout, na odziv talkerja, ce ni odziva po 4s gremo vseeno naprej
	* Dodat je treba nekaksen failsave v primeru, da nav2 ne more uresnicit goala (npr, ce smo v kotu iz katerega si ne upa...)
	* Pogledat si je treba in napisat launch fajle.
	* Detect people: mogoce bi blo pametno kaj naredit tudi v primeru, ko zaznamo vec kot 1 obraz v enem framu
