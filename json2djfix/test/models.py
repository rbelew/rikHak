"""djOakData.showCrime.models.py: """

__author__ = "rik@electronicArtifacts.com"
__version__ = "0.1"

class SFFilm4LL(models.Model):
    idx = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    year = models.IntegerField()
    writer = models.CharField(max_length=100)
    director = models.CharField(max_length=100)
    
    def __unicode__(self):
        return '%d:%s' % (self.idx,self.title)
