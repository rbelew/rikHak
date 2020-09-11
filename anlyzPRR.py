'''harvestPRR: analyze Public Record Requests from CSV data provided by NextRequest

Created 27 Aug 20

@author: rik@electronicArtifacts.com
'''

from collections import defaultdict
import csv
import datetime
import json
import random
import re
import requests
import sys
import time
import urllib

import re


PRRDateFmt = '%Y-%m-%dT%H:%M:%S'
PRRDateMicroSecFmt = '%Y-%m-%dT%H:%M:%S.%f'

DateTypes = {'date_received': 'recdDate',
			'date_created': 'createDate',
			'status_updated': 'statusUpDate'}

def freqHist3(tbl):
	'''python3 version
	ASSUME: values are frequencies, returns sorted list of (val,freq) items in descending freq order
	'''
	
	from functools import cmp_to_key
	def cmpd1(a,b):
		"decreasing order of frequencies"
		return b[1] - a[1]

	
	flist = list(tbl.items()) #python3
	flist.sort(key=cmp_to_key(cmpd1))
	return flist

AllCSVHeader = ['Id', 'Created At', 'Request Text', 'Due Date', 'Point of Contact', 'Request Date',
			'Status', 'URL', 'Visibility', 'Closed Date', 'Closure Reasons',
			'Departments', 'Format Received', 'Staff Time (hrs:minutes)',
			'Staff Time (minutes)', 'Tags', 'Embargo Ends On Date',
			'Staff Cost', 'Date First Contact', 'First Contact Event',
			'Compliance', 'Anticipated Fulfillment Date', 'Expiration Date',
			'Requester City', 'Requester State', 'Requester Zipcode', 'Requester Company']

DeptNorm = {"Admin: Planning, Building & Neighborhood Preserv": "Admin: Building Inspection",
			"Budget and Fiscal": "Budget and Revenue - Revenue Division",
			"City Attorney Administration Unit": "City Attorney",
			"City Auditor Unit": "City Auditor",
			"City Clerk Unit": "City Clerk",
			"Oakland Police Department": "Police Department",
			"Contracts and Compliance": "Contracts Compliance",
			"Transportation Services - Administration": "Department of Transportation",
			"Fire": "Fire Department",
			"Human Resources Management": "Human Resources",
			"Information Technology (IT)": "Information Technology",
			"Public Works Agency": "Public Works"}

CSVDTFormat = '%m/%d/%Y %H:%M:%S %p'
# 07/01/2020 09:54:53 AM

def bldIndexTblCSV(inf,startDate=None):
	'''return prrIDTbl, deptTbl
	'''

	prrTbl = {}
	deptTbl = defaultdict(list) # keep list of all prrIDs
	statusTbl = defaultdict(int)
	ncloseDate = 0
	nolder = 0
	nmultDept = 0
	deptSepChar = b'\xef\xbf\xbd' # only used in Finance
	
	reader = csv.DictReader(open(inf,encoding = "utf8",errors='replace'))
	for i,entry in enumerate(reader):
		prr = {}
		prrID = entry['Id']
		
		createDateStr = entry['Created At'].strip()
		prr['createDate'] = datetime.datetime.strptime(createDateStr,CSVDTFormat) if createDateStr != '' else None

		if prr['createDate'] == None or \
			(startDate != None and prr['createDate'] < startDate):
			nolder += 1
			continue
		
		deptStr = entry['Departments'].strip()
		# NB: multiple department separated by semi-colon
		if deptStr.find(';') == -1:
			deptList = [deptStr]
		else:
			nmultDept += 1
			deptList = [dept.strip() for dept in deptStr.split(';')]
			
		deptList2 = []
		for dept in deptList:
			ndept = DeptNorm[dept] if dept in DeptNorm else dept
			if ndept != '':
				deptList2.append(ndept)
				deptTbl[ndept].append(prrID)
		prr['dept'] = deptList2
			
		closeDateStr = entry['Closed Date'].strip()
		prr['closeDate'] = datetime.datetime.strptime(closeDateStr,CSVDTFormat)  if closeDateStr != '' else None
		prr['status'] = entry['Status'].strip()
		prr['text'] = entry['Request Text'].strip()
		prr['closeReason'] = entry['Closure Reasons'].strip()
		prr['URL'] = entry['URL'].strip()
		
		
		statusTbl[ prr['status'] ] += 1
		if prr['closeDate'] != None:
			ncloseDate += 1
			
		prrTbl[prrID] = prr
		
	print('bldIndexTblCSV: NPRR=%d NDept=%d NMultDept=%d NCloseDate=%d' % \
		(len(prrTbl),len(deptTbl),nmultDept,ncloseDate))
	if startDate != None:
		print('bldIndexTblCSV: NOld dropped=%d' % (nolder))

# 	freqList = freqHist3(deptTbl)
# 	print('Dept,Freq')
# 	for dept,freq in freqList:
# 		print('"%s",%d' % (dept,freq))

	freqList = freqHist3(statusTbl)
	print('Status,Freq')
	for status,freq in freqList:
		print('"%s",%d' % (status,freq))
	
	
	return (prrTbl, deptTbl)
		
def compHistAvg(hist):
	'''compute first moment
	ASSUME hist: value -> freq 
	'''
	sum = n = 0
	for v in hist.keys():
		n += hist[v]
		sum += v * hist[v]
		
	return n,float(sum) / n

def compMedian(hist):
	'''compute MEDIAN value
	ASSUME hist: value -> freq 
	'''

	# only singletons thwart the search for half-way point
	if len(hist) == 1:
		return hist[0]
	
	sum = n = 0
	vn = {}
	for v in sorted(hist.keys()):
		n += hist[v]
		sum += v * hist[v]
		vn[v] = n
		
	half = float(n/2.)
	for v in sorted(hist.keys()):
		if vn[v] > half:
			return v	

def anlyzCreateDates(prrIDTbl,outf):
	'''distribution of create dates
	'''
	
	dateDist = defaultdict(int)
	nmissdate = 0
	for prrID,prr in prrIDTbl.items():
		# 180204
# 		for dtype in DateTypes.values():
# 			if dtype in prr:
# 				if cdateFnd == None:
# 					cdateFnd = prr[dtype]
# 				else:
# 					if prr[dtype] != cdateFnd:
# 						cdateFnd = min([cdateFnd,prr[dtype]])

		cdateFnd = prr['createDate']
						
		if cdateFnd== None:
			nmissdate += 1
			continue
		mkey = '%d-%02d' % (cdateFnd.year, cdateFnd.month)
		dateDist[mkey] += 1
		
	print('anlyzCreateDates: NPRR=%d NBadDate=%d' % (len(prrIDTbl),nmissdate))
	allMon = list(dateDist.keys())
	allMon.sort()
	outs = open(outf,'w')
	outs.write('Month,Freq\n')
	for mkey in allMon:
		outs.write('%s,%d\n' % (mkey,dateDist[mkey]))
	outs.close()		

def normDeptName(dept):
	return re.sub('\W','_',dept.upper())
	
def anlyzClearDates(prrIDTbl,deptTbl,startDate,outdir,minDeptFreq=10):
	'''Compute average (over previous 90 days) number of days to respond to request
				Number requests open at month start
	'''
	
	allDept = [dept for dept in deptTbl.keys() if len(deptTbl[dept]) > minDeptFreq ]
	allDept.sort()

	nonOPDresp =  defaultdict(lambda: defaultdict(int)) # month -> ndays -> freq
	nonOPDopen = defaultdict(int) # month -> freq
	
	print('\n# Dept,NOld,NMissRecd,NMissClose')
	missCloseDetails = defaultdict(lambda: defaultdict(list)) # dept -> recd -> [prrID]
	
	for dept in allDept:
		responseMon = defaultdict(lambda: defaultdict(int)) # month -> ndays -> freq
		openReqMon = defaultdict(int) # month -> freq
		
		nmissRecd = 0
		nmissClose = 0
		nolder = 0
		for prrID in deptTbl[dept]:
			prr = prrIDTbl[prrID]
			# 180228
			# recdDateTime = prr['recdDate']
			recdDateTime = prr['createDate']

			if recdDateTime==None:
				nmissRecd += 1
				continue
			
			if recdDateTime < startDate:
				nolder += 1
				continue
			try:
				recdMonKey = '%d-%02d' % (recdDateTime.year, recdDateTime.month)
			except Exception as e:
				print('huh')
		
			if prr['status'] == 'Closed':
				# 180228
				# closeDate = prr['statusUpDate']
				closeDate = prr['closeDate']
				if closeDate==None:
					nmissClose += 1
					missCloseDetails[dept][recdMonKey].append(prrID)
					continue

				respDelay = closeDate - recdDateTime
				delayDays = respDelay.days
				responseMon[recdMonKey][delayDays] += 1
				
				# NB: was 'Oakland Police Deparment' in 180204
				if dept != 'Police Department':
					nonOPDresp[recdMonKey][delayDays] += 1
			
			else:
				openReqMon[recdMonKey] += 1
		
				# NB: was 'Oakland Police Deparment' in 180204
				if dept != 'Police Department':
					nonOPDopen[recdMonKey] += 1
		
		print('"%s",%d,%d,%d' % (dept,nolder,nmissRecd,nmissClose))
				
		allMonth = list(responseMon.keys())
		allMonth.sort()
		
		normDept = normDeptName(dept)
		
		outf = outdir + normDept + '-RT.csv'
		outs = open(outf,'w')		
		outs.write('Month,NClose,NOpen,Avg,Median\n')
		for recdMonKey in allMonth:
			nreq,avgDelay = compHistAvg(responseMon[recdMonKey])
			medianDelay = compMedian(responseMon[recdMonKey])
			outs.write('%s,%d,%d,%f,%d\n' % (recdMonKey,nreq,openReqMon[recdMonKey],avgDelay,medianDelay))
		outs.close()
		
# 		outf = outdir + normDept + '-nopen.csv'
# 		outs = open(outf,'w')		
# 		outs.write('Month,NOpen\n')
# 		for recdMonKey in allMonth:
# 			outs.write('%s,%d\n' % (recdMonKey,openReqMon[recdMonKey]))
# 		outs.close()
		
	allMonth = list(nonOPDresp.keys())
	allMonth.sort()

	outf = outdir + 'NonOPD-RT.csv'
	outs = open(outf,'w')		
	
	outs.write('Month,N,NOPen,Avg,Median\n')
	for recdMonKey in allMonth:
		nreq,avgDelay = compHistAvg(nonOPDresp[recdMonKey])
		medianDelay = compMedian(nonOPDresp[recdMonKey])
		outs.write('%s,%d,%d,%f,%d\n' % (recdMonKey,nreq,nonOPDopen[recdMonKey],avgDelay,medianDelay))
	outs.close()
	
# 	outf = outdir + 'NonOPD-NOpen.csv'
# 	outs = open(outf,'w')		
# 	outs.write('Month,NOpen\n')
# 	for recdMonKey in allMonth:
# 		outs.write('%s,%d\n' % (recdMonKey,nonOPDopen[recdMonKey]))
# 	outs.close()
	
	outf = outdir + 'missClose.csv'
	outs = open(outf,'w')
	# missCloseDetails: dept -> recd -> freq
	
	allDateSet = set()
	for dept in missCloseDetails.keys():
		allDateSet.update(missCloseDetails[dept].keys())
	allDates = sorted(list(allDateSet))
	
	hdr = 'Dept'
	for date in allDates:
		hdr += ',%s' % (date,)
	outs.write(hdr+'\n')
	
	for dept in sorted(missCloseDetails.keys()):
		line = dept
		for date in allDates:
			if date in missCloseDetails[dept]:
				line += ',%d' % (len(missCloseDetails[dept][date]),)
			else:
				line += ', '
		outs.write(line+'\n')
	outs.close()
	
		
def rptDeptFreq(prrTbl, deptTbl,startDate,outf):
	
	# freq = defaultdict(int)
	outs = open(outf,'w')
	outs.write('Dept,Freq\n')
	
	for dept in sorted(deptTbl.keys()):
		nrecent = 0
		for prrIdx in deptTbl[dept]:
			prr = prrTbl[prrIdx]
			if prr['createDate'] >= startDate:
				nrecent += 1
		outs.write('%s,%d\n' % (dept,nrecent))
	
	outs.close()

def rptOpenPRR(prrTbl,outf):
	
	daysOpen = defaultdict(lambda: defaultdict(list)) # ndays -> OPD/non -> [prrID]
	runDate = datetime.datetime.today()
	
	for prrID in prrTbl.keys():
		prr = prrTbl[prrID]
		opdP = 'Police Department' in prr['dept']

		if prr['status'] == 'Open' or prr['status'] == 'Overdue' or prr['status'] == 'Due soon':
			recdDateTime = prr['createDate']
			openPeriod = runDate - recdDateTime
			openDays = openPeriod.days
			# NB: capture integer dividend
			openYears = openDays // 365
			if openYears == 0:
				dkey = openDays
			else:
				dkey = 1000 + openYears
			daysOpen[opdP][dkey].append(prrID)			
		
	outs = open(outf,'w')
	outs.write('DaysOpen,NOPD,NOther,PRR-OPD,PRR-non\n')
	allNDaySet = set(daysOpen[0].keys()).union(set(daysOpen[0].keys()))
	allNDay = sorted(list(allNDaySet))
	for nday in allNDay:
		if nday > 365:
			lbl = '> %d year' % (nday-1000)
		else:
			lbl = '%d' % nday
		opdList = daysOpen[1][nday] if nday in daysOpen[1] else []
		nonList = daysOpen[0][nday] if nday in daysOpen[0] else []
			
		outs.write('%s,%d,%d,"%s","%s"\n' % (lbl,len(opdList),len(nonList), opdList,nonList))
		
	outs.close()

def getWebPages(prrTbl,outf):
	
	outs = open(outf,'w')
	outs.write('PRRID,OPD,Text\n')
	nempty = 0
	npdf = 0
	for i,prrID in enumerate(sorted(prrTbl.keys())):

		prr = prrTbl[prrID]
		if prr['URL'] == '':
			nempty += 1
			continue
			
		opdP = 'Police Department' in prr['dept']
		
		url = prr['URL']
		response = urllib.request.urlopen(url)
		webContentBytes = response.read()
		webContent = webContentBytes.decode("utf-8")
		if webContent.find('pdf') != -1:
			print('here')
			npdf += 1
		else:
			continue
	
		if i % 100 == 0:
			print(i,npdf,nempty)
			
		# outs.write('%s,%d,"%s"\n' % (prrID,opdP,prr['text']))
	outs.close()
	print('prr20-text: NPRR=%d NEmpty=%d' % (len(prrTbl),nempty))

def loadPRRQuery(inf):
	
	reader = csv.DictReader(open(inf))
	prrIDList = []
	for i,entry in enumerate(reader):
		# Exhibit,PRRId
		prrIDList.append(entry['PRRId'].strip())
	return prrIDList
		
def rptQry(qryList,outf):
	outs = open(outf,'w')
	outs.write('PRID,CreateDate,DaysOpen,Status\n')
	
	runDate = datetime.datetime.today()
	for prrID in qryList:
		prr = prr20Recent[prrID]
		recdDateTime = prr['createDate']
		openPeriod = runDate - recdDateTime
		openDays = openPeriod.days
		outs.write('%s,%s,%d,%s\n' % (prrID,prr['createDate'].date(),openDays,prr['status']))
		
	outs.close()
	
	
if __name__ == '__main__':

	dataDir = '/Users/rik/Data/c4a-Data/OAK_data/recordTrac/'
	

	startDate = datetime.datetime(2017,1,1)
	
	csvFile = dataDir + 'requests-2020-07-01-sdoran.csv'
	# prr20, deptTbl = bldIndexTblCSV(csvFile)
	prr20Recent, deptTbl = bldIndexTblCSV(csvFile,startDate)
	
	openPRRFile = dataDir + 'openPRR_200831.csv'
	rptOpenPRR(prr20Recent,openPRRFile)

	deptFreqFile = dataDir + 'deptFreq2.csv'
	rptDeptFreq(prr20Recent, deptTbl,startDate,deptFreqFile)
	
	createDateFile = dataDir + 'createDate_200831.csv'
	anlyzCreateDates(prr20Recent,createDateFile)
	
	clearDateDir = dataDir + 'deptClear_200831/'
	anlyzClearDates(prr20Recent,deptTbl,startDate,clearDateDir)
	
	openOPDFile = dataDir + 'openOPD_200831.csv'
	rptOpenPRR(prr20Recent,openOPDFile)

	

