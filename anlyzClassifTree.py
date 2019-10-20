''' anlyzClassifTree: annotate dot trees produced by weka's graphviz plugin
Created on Apr 28, 2019

@version 0.1 191020

@author: rik
'''

import math
import re 
import pygraphviz as pgv
import pygraphml

class DTNode():
	def __init__(self,id):
		self.id = id
		self.name = ''
		self.leaf = None
		
		# if self.leaf
		self.predicate = None
		self.nsample = 0
		self.nneg = 0
		self.npos = 0
		self.nsuccess = 0
		self.misclass = 0
		# else
		self.attrib = None

		self.cummPos = 0
		self.cummNeg = 0
		self.totSample = 0
		self.cummSucc = 0
		self.cummFail = 0
		
		self.outNbr = []
		self.inNbr = []
		self.depth = None
		
		
		# print('Node',str(self))
		
	def __str__(self):
		return '%s' % (self.id)
		
	def addToNbr(self,nbr,lbl):
		l = DTLink(self,nbr,lbl)
		self.outNbr.append(l)
		nbr.inNbr.append(l)
		# print('addNbr',str(l))
		global allEdges
		allEdges[(self.id,nbr.id)] = l
		return l

	def useName(self,name):
		self.name = name
		if name.find('(') == -1:
			self.leaf = False
			self.attrib = name
		else:
			self.leaf = True
			m = re.match(LeafNamePat,name)
			if m == None:
				print('addName: unmatched name?!', self, name)
				return					
			mdict = m.groupdict()
			self.predicate = (True if mdict['pred'] == BoolTrueString else False)
			# https://waikato.github.io/weka-wiki/not_so_faq/j48_numbers/
			# If your data has missing attribute values then you will end up with fractional instances at the leafs. 
			self.nsample = float(mdict['nsample'])
			if mdict['misclass'] == None:
				self.misclass = 0
			else:
				self.misclass = float(mdict['misclass'])
			self.nsuccess = self.nsample - self.misclass
			if self.predicate:
				self.npos = self.nsample - self.misclass
				self.nneg = self.nsample - self.npos
			else:
				self.nneg = self.nsample - self.misclass
				self.npos = self.nsample - self.nneg		
	
	def info(self):
		istr = ''
		if self.leaf:
			istr = '. %s %d/%d' % (self.predicate,self.npos,self.nneg)
		else:
			istr = '? %s' % (self.attrib)
		return istr
	
	def accummSample(self):
		tot = 0
		if self.leaf:
			assert len(self.outNbr) == 0, 'leaves have no outNbr?!'

			self.cummPos = self.npos
			self.cummNeg = self.nneg
			self.totSample = self.nsample
			return self.npos, self.nneg, self.nsample
		
		for edge in self.outNbr:
			onbr = edge.target
			onpos,onneg,otot = onbr.accummSample()
			self.cummPos += onpos
			self.cummNeg += onneg
			self.totSample += otot
		
		return self.cummPos, self.cummNeg, self.totSample	

	def accummSuccess(self):
		nsucc = 0
		nfail = 0
		
		if self.leaf:
			self.cummSucc = self.nsuccess
			self.cummFail = self.misclass						
			return self.nsuccess, self.misclass
		
		for edge in self.outNbr:
			onbr = edge.target
			onnsucc, onnfail = onbr.accummSuccess()
			nsucc += onnsucc
			nfail += onnfail
		
		self.cummSucc = nsucc
		self.cummFail = nfail
		return nsucc,nfail	

	def cummCounts(self):
		istr = '%d = %d + %d' % (self.totSample,self.cummPos,self.cummNeg)
		return istr
			
			
class DTLink():
	def __init__(self,src,target,lbl):
		self.src = src
		self.target = target
		self.label = lbl
		
		m = re.match(EdgePat,lbl)
		if m == None:
			print('DTLink: unmatched lbl?!', self, lbl)
			return					
		mdict = m.groupdict()
		self.reln = mdict['reln']
		self.val = mdict['val']
		
	def __str__(self):
		return '%s->%s: %s %s' % (self.src.id,self.target.id,self.reln,self.val)

# https://waikato.github.io/weka-wiki/not_so_faq/j48_numbers/
# What do those numbers mean in a J48 tree?
# The first number is the total number of instances (weight of instances) reaching the leaf. 
# The second number is the number (weight) of those instances that are misclassified.
						
LeafNameRE = r'(?P<pred>.+) \((?P<nsample>[0-9.]+)(/(?P<misclass>[0-9.]+))?\)'
LeafNamePat = re.compile(LeafNameRE)

EdgeRE = r'(?P<reln>.+) (?P<val>.+)'
EdgePat = re.compile(EdgeRE)

def loadGraphML(inf):
	parser = pygraphml.GraphMLParser()
	g = parser.parse(inf)
	return g

def bldDT(dotGraph):
	'''build decision tree from dot graph
	'''
		
	global visited
	global allEdges
	
	def _dfs(g,momDT,kidNd,kidID,depth,elbl):
		'NB: _dfs passed mom=DTNode, kid=dotGraph.Node'
		
		global visited

		name = kidNd.attr['label']
		
		kidDT = visited[kidID]
		kidDT.useName(name)
		
		edge = momDT.addToNbr(kidDT,elbl)

		succ = dotGraph.successors(kidID)
		assert len(succ) == 0 or len(succ) == 2, 'ASSUME binary tree?! %s' % kidDT
		
		for succIdx,toNd in enumerate(succ):
			dedge = dotGraph.get_edge(kidID,toNd)
			tondID = str(toNd)
			# NB: create DTNode, but don't yet know it's label attribute
			toDT = DTNode(tondID)
			toDT.depth = depth+1
			visited[tondID] = toDT
			elbl = dedge.attr['label']
			_dfs(dotGraph,kidDT,toNd,tondID,depth+1,elbl)
		
		
	root = dotGraph.nodes()[0]
	rootname = root.attr['label']
	
	visited = {}
	allEdges = {}
	rootID = str(root)
	# NB: dotGraph IDs also used for DTNodes, to facilitate forward-linking by DTNode.addToNbr()
	dt = DTNode(rootID)
	dt.useName(rootname)
	dt.depth = 0
	visited[rootID] = dt

	succ = dotGraph.successors(root)
	assert len(succ) == 0 or len(succ) == 2, 'ASSUME binary tree?! %s' % dt
	
	for succIdx,toNd in enumerate(succ):
		dedge = dotGraph.get_edge(root,toNd)
		elbl = dedge.attr['label']
		toID = str(toNd)
		toDT = DTNode(toID)
		toDT.depth = 1
		visited[toID] = toDT
		# NB: _bfs passed dt=DTNode, toNd=dotGraph.Node()
		_dfs(dotGraph,dt,toNd,toID,1,elbl)
		
	return dt

def rptDT(allDTID):
	for dtID in allDTID:
		dt = visited[dtID]
		dtInfo = dt.info()
		print('%s %s %s %s' % (dt.depth*'   ', dt, dt.info(), dt.cummCounts()))

def dtAttrib2dot(gname,maxLeafSample,newdotf):
	'''produce dot graph with attributes appropriate to trained decision tree
		- LEAF node size proportional to nsample
		- decision reln,val incorporated into node label
		- also cummPos,cummNeg,nsuccess,nfail
		- different shapes for TRUE/FALSE
		- 2do: fill indicating npos/nneg
	'''
	
	minRadius = 0.125
	maxRadius = 2.
	def _scale(nsample):
		s = float(nsample)/maxLeafSample
		r = math.sqrt(s)
		d = 2 * (maxRadius * r + minRadius)
		return d
		
	noteWorthySampleSize = 10
	
	piAttr = {'fval': 1, 'nedge': 1}
	ggenAttr = {'degree','ggen','q'}
	dots = open(newdotf,'w')
	dots.write('digraph %s {\n' % (gname))
	
	global visited
	global allEdges

	allDTID = list(visited.keys())
	allDTID.sort(key=lambda k: int(k[1:])) # drop 'N' prefix for node, treat as int
	
	for dtID in allDTID:
		dt = visited[dtID]
		if dt.leaf:
			shape = 'circle' if dt.predicate else 'square'
			size = _scale(dt.totSample)
			sizeStr = ('%6.3f' % (size)).strip()
			lbl = '%d / %d' % (dt.npos,dt.nneg)
			posRatio = float(dt.npos) / dt.totSample
			lineType = 'solid'
			ndLine = '%s [label="%s" shape="%s" size="%s" style="%s"  ]\n' % \
					(dt.id,lbl,shape,sizeStr,lineType)
		else:
			shape = 'diamond' 
			outlink = dt.outNbr[0]
			if dt.attrib in piAttr:
				color = 'aquamarine'
			elif dt.attrib in ggenAttr:
				color = 'mistyrose'
			else:
				color = 'cornsilk'
			lbl = '%s %s %s\n(%d/%d)\n[%d/%d]' % (dt.attrib,outlink.reln,outlink.val, \
													dt.cummPos,dt.cummNeg,dt.cummSucc,dt.cummFail)

			ndLine = '%s [label="%s" shape="%s" size="1" style="filled" fillcolor="%s" ]\n' % \
					(dt.id,lbl,shape,color)
		dots.write(ndLine)
		
	allEdgePairs = list(allEdges.keys())
	allEdgePairs.sort(key=lambda k: (int(k[0][1:]),int(k[1][1:])))
	for epair in allEdgePairs:
		dots.write('%s->%s\n' % (epair[0],epair[1]))
			
	dots.write('}\n')
	dots.close()
	
	
if __name__ == '__main__':
	
	gname = 'graphName'
	dataDir = '<pathToFile>'
	dotfile =  dataDir + gname + '.dot'
	dotGraph = pgv.AGraph(dotfile)

	global BoolTrueString
	BoolTrueString = 'TRUE' # 't'
	
	assert dotGraph.directed == True, 'ASSUME DIRECTED graph?!'
	dt = bldDT(dotGraph)
	npos,nneg,totsample = dt.accummSample()
	nsucc,nfail = dt.accummSuccess()

	global visited
	allDTID = list(visited.keys())
	allDTID.sort(key=lambda k: int(k[1:])) # drop 'N' prefix for node, treat as int
	
	maxLeaf = None
	maxLeafSample = 0
	for dtID in allDTID:
		dt = visited[dtID]
		if dt.leaf and dt.totSample > maxLeafSample:
			maxLeafSample = dt.totSample
			maxLeaf = dt
	
	print('done TotSample=%d NPos=%d NNeg=%d MaxLeafSample=%d (%s)' % \
		(totsample,npos,nneg,maxLeafSample,maxLeaf.id))
	
	newdotf = dataDir + '%s-dtattrib.dot' % (gname)
	dtAttrib2dot(gname,maxLeafSample,newdotf)
	print('done')