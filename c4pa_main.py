''' CP4A 
routines in support of MeasureLL
Created on Sep 25, 2016

@author: rik
'''

from collections import defaultdict
import csv
import cPickle
import os
import sys

def loadVoters(inf):

	voterTbl = {}
	reader = csv.DictReader(open(inf))

	print 'loadVoters: ...'
	for i,entry in enumerate(reader):
		# signup_id,voter_guid,first_name,last_name,election_country,election_state,election_at,ballot_vote_method,ballot_party,ballot_cast_at
		nbUID= entry['signup_id']
		if nbUID not in voterTbl:
			newDict = {'fname': entry['first_name'],
						'lname': entry['last_name'],
						'voterID': entry['voter_guid'],
						'history': []
						}
			voterTbl[nbUID] = newDict
		vdate = entry['election_at']
		vmethod = entry['ballot_vote_method']
		vparty = entry['ballot_party']
		voterTbl[nbUID]['history'].append( (vdate,vmethod,vparty))

	print 'loadVoters: NVoters=%d' % len(voterTbl)
	return voterTbl

def loadNB(inf):

	nbTbl = {}
	reader = csv.DictReader(open(inf))
	print 'loadNB: ...'
	nbadDist=0
	nbadPrec=0

	for i,entry in enumerate(reader):
		# nationbuilder_id,first_name,middle_name,last_name,born_at,sex,email,phone_number,tag_list,created_at,donations_amount,state_file_id,city_district,precinct_name,primary_zip,primary_address1,primary_address2,primary_address3,mailing_street_number,mailing_street_prefix,mailing_street_name,mailing_street_type,mailing_street_suffix,mailing_unit_number

		nbUID= entry['nationbuilder_id']
		newDict = {'fname': entry['first_name'],
					'lname': entry['last_name'],
					'mname': entry['middle_name'],
					'fullName': entry['full_name'],
					'sex': entry['sex'],
					'dob': entry['born_at'],
					'email': entry['email'],
					'twitter': entry['twitter_login'],
					'supporter': bool(entry['is_supporter']=='true'),
					'voter': bool(entry['tag_list'].find('cavoters') != -1),
					'address1': entry['mailing_address1'],		
					'address2': entry['mailing_address2'],
					'address3': entry['mailing_address3'],
					'state': entry['mailing_state'],
					'city': entry['mailing_city'],
					'zip': entry['mailing_zip'],
					}
		district = entry['city_district']
		if district.startswith('Oakland-'):
			newDict['district'] = int(district[8:])
		else:
			nbadDist+=1
			newDict['district'] = 0
			
		precinct = entry['precinct_name']
		if precinct.startswith('06001-Alameda-'):
			newDict['precinct'] = int(precinct[14:])
		else:
			nbadPrec+=1
			newDict['precinct'] = 0

		nbTbl[nbUID] = newDict

	print 'loadNB: NNB=%d NBadDistrict=%d NBadPrecinct=%d' % (len(nbTbl),nbadDist,nbadPrec)
	
	return nbTbl

def addNBInfo(voterTbl,nbTbl):

	nmiss=0
	for nbid in nbTbl.keys():
		if nbid not in voterTbl:
			nmiss += 1
			voterTbl[nbid] = {}

		# Add all fields grabbed by loadNB()
		for k in nbTbl[nbid].keys():
			voterTbl[nbid][k] = nbTbl[nbid][k]
		
	print 'addNBInfo: NMiss=%d' % (nmiss)
	return voterTbl
				
if __name__ == '__main__':
	c4paDir = '/Data/sharedData/c4a_oakland/copOver/C4PA-NB/'
	votersFile = c4paDir+'NB-voter-160928.csv'
	
	voterpklf = c4paDir+'voterInfo_160929.pkl'
	if os.path.isfile(voterpklf):
		print 'CP4A: voterInfoPkl exists, loading', voterpklf
		voterTbl = cPickle.load(open(voterpklf,'rb'))
	else:
		print 'CP4A: voterInfoPkl does not exist, building...' 
		
		nbFile = c4paDir+'NB-people-160928.csv'
		nbTbl = loadNB(nbFile)

		voterTbl = loadVoters(votersFile)
		
		# NB: NB info added to voters, 
		voterTbl = addNBInfo(voterTbl,nbTbl)
		
		cPickle.dump(voterTbl, open(voterpklf,'wb'))
		
		print 'CP4A: done building voterInfoPkl; dumped to', voterpklf

	
	