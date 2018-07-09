Road detection and classification in satellite images
=====================================================

Portfolio Project by Lisa Hesse, Kiran Prakash and Harald Hentschke, Data Science Retreat Berlin (Batch 15)

Challenges/things to consider/questions
---------------------------------------
* how to deal with edges and differences in resolution resulting from the overlay of images taken by different satellites or at different times?
* clouds may be an issue
* shall we do rural areas only or include settlements? Use different networks for different areas?
* averaging images taken over time to enhance signal to noise ratio - does Google Earth (GE) do that already? In general, what kind of satellite imagery do we get out of GE, and how (API)?
* hi-resolution images: tire tracks may be a defining feature of unpaved roads
* we should decide on fixed image sizes for both hi and lo resolution images early on
* maybe label and classify creeks and rivers as well so as to prevent the network from misclassifing them?
* maybe classify not only paved and unpaved roads, but also 'roads in settlements'?

Ressources
----------

Data set from Mnih & Hinton, 2010
http://www.cs.toronto.edu/~vmnih/data/

Copernicus satellite websites

best starting point for technical info:
https://earth.esa.int/web/sentinel/user-guides

https://scihub.copernicus.eu/
https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1
https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1/production-scenario
https://sentinels.copernicus.eu/web/sentinel/sentinel-data-access

Website of researcher who also expressed interest in working on the challenge (contacted June 27)
http://www.hnee.de/de/Fachbereiche/Wald-und-Umwelt/Professorinnen-und-Professoren/Pierre-Ibisch/Pierre-L.-Ibisch-E2208.htm

Copernicus satellite info (from Wikipedia)

* Sentinel-1 will provide all-weather, day and night radar imaging for land and ocean services. The first Sentinel-1A satellite was successfully launched on 3 April 2014, by an Arianespace Soyuz, from the Guyana Space Center;.[9] The second Sentinel-1B satellite was launched on 25 April 2016 from same spaceport. 
* Sentinel-2 will provide high-resolution optical imaging for land services (e.g. imagery of vegetation, soil and water cover, inland waterways and coastal areas). Sentinel-2 will also provide information for emergency services. The first Sentinel-2 satellite has successfully launched on 23 June 2015.[10] 
* Sentinel-3 will provide ocean and global land monitoring services. The first Sentinel-3A satellite was launched on 16 January 2016 by a Eurockot Rokot vehicle from the Plesetsk Cosmodrome in Russia;[11][12] 

Ideas
-----
* divide each satellite image into small segments, possibly in several runs of overlapping segments, classify these into two or three categories (paved road/unpaved road/no road), then try to stick the road-detecting ones together so that we have an outline of the roads
* use nested crossvalidation on small sample-dats sets
