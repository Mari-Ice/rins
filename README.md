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
	* Detect people: mogoce bi blo pametno kaj naredit tudi v primeru, ko zaznamo vec kot 1 obraz v enem framu
