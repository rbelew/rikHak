''' updateDB: merge DB with newly imported songs into primary DB
Created on Jul 3, 2025. 
	
@author: rik
'''


from collections import defaultdict

import os
import sqlite3 as sqlite
		
def ppSchema(cur):	
	result = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
	table_names = [r[0] for r in result]
	
	print('# tbl,cidx,cname,ctype,cNotNull,cDefVal,cpkey')
	for tbl in table_names:
		result = cur.execute("PRAGMA table_info('%s')" % tbl).fetchall()
		for colInfo in result:
			(cidx,cname,ctype,cNotNull,cDefVal,cpkey) = colInfo
			print (f'{tbl},{cidx},{cname},{ctype},{cNotNull},{cDefVal},{cpkey}')

def collectSchema(tblList,cur):
	'''schema: tbl -> colName -> [colType,colDefault,colPKey]
	'''
	
	schema = defaultdict(lambda: defaultdict(list))
	
	for table_name in tblList:
		result = cur.execute("PRAGMA table_info('%s')" % table_name)
		cols = result.fetchall()
		
		for colInfo in cols:
			# (colIdx,colName,colType,colNNull,colDefault,colPKey) = colInfo
			# prevSchema[table_name][colName] = [colType,colNNull,colDefault,colPKey]
			
			schema[table_name][colInfo[1]] = colInfo
			
	return schema

def compareBeetSchemata(db1path,db2path,rptFile):
	db1 = sqlite.connect(db1path)
	cur1 = db1.cursor()
	
	db2 = sqlite.connect(db2path)
	cur2 = db2.cursor() 
	
	tblList = ['albums','items']
	
	schema1 = collectSchema(tblList,cur1)
	schema2 = collectSchema(tblList,cur2)
	
	outs = open(rptFile,'w')
	for tbl in tblList:
			
		prevSet = set(schema1[tbl].keys())
		newSet = set(schema2[tbl].keys())
		
		both = prevSet.intersection(newSet)
		add = newSet.difference(prevSet)
		drop= prevSet.difference(newSet)
		
		outs.write(f'{tbl} both: {len(both)} {sorted(both)}\n')
		outs.write(f'{tbl} add: {len(add)} {sorted(add)}\n')
		outs.write(f'{tbl} drop: {len(drop)} {sorted(drop)}\n')
		
		outs.write('tbl,col,PrevIdx,NewIdx,TypeChg\n')
		for k in sorted(both):
			# info: (colIdx,colName,colType,colNNull,colDefault,colPKey)
			colInfo1 = schema1[tbl][k]
			colInfo2 = schema2[tbl][k]
			
			prevIdx =  colInfo1[0]
			newIdx =   colInfo2[0]
			prevType = colInfo1[2]
			newType =  colInfo2[2]
			
			tchg = f'{prevType}->{newType}' if prevType != newType else ''
			outs.write(f'{tbl},{k},{prevIdx},{newIdx},{tchg}\n')

	outs.close()

def migrateBeets(db1path,db2path):
	'''Add all tables' contents from db1 to db2
		albums, items: use db2 schema to shape inserted db1 rows with NULL for missing values
		album_attributes, item_attributes: add with updated entity_id's
	'''
	db1 = sqlite.connect(db1path)
	cur1 = db1.cursor()
	
	db2 = sqlite.connect(db2path)
	cur2 = db2.cursor() 
	
	tblList = ['albums','items']
	
	schema1 = collectSchema(tblList,cur1)
	schema2 = collectSchema(tblList,cur2)
	
	for tbl in tblList:
		
		result = cur1.execute(f'select count(1) from {tbl}')
		nrow1 = result.fetchone()[0]
		result = cur2.execute(f'select count(1) from {tbl}')
		nrow2 = result.fetchone()[0]
		
		print(f'* {tbl} {nrow1=} {nrow2=}')
		
		prevSet = set(schema1[tbl].keys())
		newSet = set(schema2[tbl].keys())
		ncol2 = len(newSet)
		
		colSeq = [ (schema2[tbl][colName][0],colName) for colName in schema2[tbl].keys()]
		orderedCol = sorted(colSeq)
		
		# create transform: ordered list of matched (prevIdx), missing (None) or transformed type
		transform = []
		for colIdx,colName in orderedCol:
			# info: (colIdx,colName,colType,colNNull,colDefault,colPKey)
			
			# NB: drop primary key, use auto-increment
			if colName=='id':
				continue
			
			# need to retain album_id colIdx for items
			if tbl=='items' and colName=='album_id':
				itemAlbumIdColIdx = colIdx
			
			if colName in schema1[tbl]:
				# 250715: albums.r128_album_gain,items.r128_album_gain,items.r128_track_gain
				# are the only ones with type change (int -> real)
				# but these are always NULL, so drop type change code
				# if schema1[tbl][colName][2] == schema2[tbl][colName][2]:
				# 	transform.append( (schema1[tbl][colName][0], ) )
				# else:
				# 	transform.append( (schema1[tbl][colName][0],schema2[tbl][colName][2]) )
				transform.append( (schema1[tbl][colName][0], ) )
				
			else:
				transform.append(None)
		
		# 2do: perhaps easier to include the COLUMN_NAMES param to insert?

		sql1 = f'select * from {tbl}'
		qMarks = ','.join(['?' for i in range(ncol2)])
		sql2 = f'insert into {tbl} values({qMarks})'
		
		# old->new row index for both albums, items  attributes
		# ASSUME albums processed first, so prev2newAlbumIdx available for items.album_id
		if tbl=='albums':
			prev2newIdx = {}
		else:
			prev2newAlbumIdx = prev2newIdx.copy()
			prev2newIdx = {}
			
		result = cur1.execute(sql1)
		for rowIdx,row in enumerate(result.fetchall()):
			# ASSUME AUTO-INCREMENT with None/NULL corresponding FIRST column in newDB
			values = [None]
			prevId = row[ schema1[tbl]['id'][0] ]
			for tidx,t in enumerate(transform):
				if t == None:
					values.append(None)
				elif len(t) == 1:
					# ASSUME albums processed first, so prev2newAlbum available
					if (tbl=='items' and tidx == itemAlbumIdColIdx):
						values.append(prev2newAlbumIdx[row[itemAlbumIdColIdx]])
					else:
						values.append(row[t[0]])
						
				# 250715:  drop type change code
				# else:
				# 	pv = row[t[0]]
				# 	# ASSUME ONLY type changes are INT -> REAL
				# 	if pv == None:
				# 		v = None
				# 	elif  t[1]=='REAL':
				# 		v = float(pv)
				# 	else:
				# 		assert False, f'{rowIdx} bad type? {t[1]}'
				# 		v = None
				# 	values.append(v)
			
			result = cur2.execute(sql2,values)
			newId = cur2.lastrowid
			prev2newIdx[prevId] = newId
		
		result = cur2.execute(f'select count(1) from {tbl}')
		newNrow2 = result.fetchone()[0]
		print(f'* {tbl} {newNrow2=} ({nrow1+nrow2})\n')

		# commit each table
		db2.commit()

		## bring forward table attributes
		atblName = tbl[:-1]+'_attributes' # drop S from tbl name
		result = cur1.execute(f'select count(1) from {atblName}')
		nattr1 = result.fetchone()[0]
		result = cur2.execute(f'select count(1) from {atblName}')
		nattr2 = result.fetchone()[0]
		print(f'* {atblName} {nattr1=} {nattr2=}')
		
		sql3 = f'select * from {atblName}'
		qMarks = '?,?,?,?'
		sql4 = f'insert into {atblName} values({qMarks})'
		result = cur1.execute(sql3)
		for rowIdx,row in enumerate(result.fetchall()):
			(idx,prevId,k,v) = row
			values = [None,prev2newIdx[prevId],k,v]
			result = cur2.execute(sql4,values)

		result = cur2.execute(f'select count(1) from {atblName}')
		newNrow2 = result.fetchone()[0]
		print(f'* {atblName} {newNrow2=} ({nattr1+nattr2})\n')
			
		# commit each table
		db2.commit()
			
if __name__ == '__main__':
	# hancock
	dataDir = '/path_to_directory_with both databses/'
	
	db1path = dataDir+'library_prev.db'
	db2path = dataDir+'library_new.db'
	
	# for dbpath in [db1path,db2path]:
	#	 ppSchema(dbpath)

	# rptFile = dataDir + 'schemaComp.csv'
	# compareBeetSchemata(db1path,db2path,rptFile)

	db2path = dataDir+'library2_tst.db'
	migrateBeets(db1path,db2path)
