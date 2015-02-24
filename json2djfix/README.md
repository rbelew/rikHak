# json2djfix

Construct a data fixture suitable for import by django from a "standard" JSON data file

For example, suppose you are building an app `sffilm` that helps film buffs visiting San Francisco visit filming locations in the city.
A simple django model for this might look something like this:

	class SFFilm4LL(models.Model):
	    idx = models.IntegerField(primary_key=True)
	    title = models.CharField(max_length=100)
	    location = models.CharField(max_length=100)
	    year = models.IntegerField()
	    writer = models.CharField(max_length=100)
	    director = models.CharField(max_length=100)
 

It so happens there is a nice little data set [provided via the (Socrata) DataSF API](https://data.sfgov.org/Culture-and-Recreation/Film-Locations-in-San-Francisco/yitu-d5am).  This data is in the "standard" format of a list-of-dictionaries, ala:

	[ {
	  "title" : "180",
	  "actor_1" : "Siddarth",
	  "locations" : "Epic Roasthouse (399 Embarcadero)",
	  "release_year" : "2011",
	  "production_company" : "SPI Cinemas",
	  "actor_2" : "Nithya Menon",
	  "writer" : "Umarji Anuradha, Jayendra, Aarthi Sriram, & Suba ",
	  "director" : "Jayendra",
	  "actor_3" : "Priya Anand"
	}
	, {
	  "title" : "24 Hours on Craigslist",
	  "actor_1" : "Craig Newmark",
	  "release_year" : "2005",
	  ...

Call each separate film's dictionary a "data element" and each key of that dictionary a "field."  Note you want to import *some* but not all of the input JSON's attributes.

django (>= v1.7) can initialize databases via ["fixture migration"](https://docs.djangoproject.com/en/1.7/topics/migrations/#data-migrations).  JSON is one common format for fixtures,  and it also assumes a list-of-dictionaries format for all new data elements.  However, django's fixtures must have, for each data element:  

* a `model: <appName>.<modelName>` key:value pair 
 
* a `pk: <primaryKeyValue>` key:value pair   

* all other fields of the data element as part of a `fields` dictionary

This script does nothing but reformat the input JSON to produce a new JSON fixture file conforming to these conventions.  

But it needs a few specifications to do that.  These are provided by a parameter file that looks like this:

	sffilm.SFFilm4LL
	title,title
	actor_1
	locations,location
	release_year,year,PK
	production_company
	actor_2
	writer,writer
	director,director
	actor_3
	distributor
	fun_facts

The first line of the file specifies the `<appName>.<modelName>`.  All other lines are comma separated and correspond to features in the input JSON data.  Typical lines have two entries: the field's name in the input JSON, and its name in the output model.  *If the line has only one element, it means that field is dropped in the output fixture.*  

If there is a third element and it is the token `'PK'`, that attribute will be used to produce the data elements' primary key ordering.  If no fields have this flag, the original ordering in the input JSON file is used.

## Running script

	python json2djfix.py sffilm2djf.txt SFfilm_api.json sffilm_fix_year.json 
	json2djfix: done. 1000/1000 data converted

## Importing fixture into django

`loaddata` is the command to import this fixture: 
 
	$ python manage.py loaddata sffilm/fixtures/sffilm_fix_year.json  
	Installed 1000 object(s) from 1 fixture(s)
	
Looks like it worked; let's check:

	$ python manage.py shell
	In [1]: from sffilm.models import SFFilm4LL
	
	In [2]: allFilms = SFFilm4LL.objects.all()
	
	In [3]: len(allFilms)
	Out[3]: 1000
	
	In [4]: film0 = allFilms[0]
	
	In [5]: film0
	Out[5]: <SFFilm4LL: 0:A Jitney Elopement>

