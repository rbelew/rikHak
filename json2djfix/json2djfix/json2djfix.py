__author__ = "rik@electronicArtifacts.com"
__version__ = "0.1"

import json
import argparse
import unittest

def loadParams(paramf):
    ps = open(paramf)
    fldMap = {}
    modelName = ''
    pk = None
    for il,line in enumerate(ps.readlines()):
        line = line[:-1] # strip \n
        if il==0:
            # NB: no need to split appName.modelName!
            modelName = line.strip()
            continue
        bits = line.split(',')
        if len(bits)==1:
            fldMap[bits[0]] = None
        else:
            fldMap[bits[0]] = bits[1]
        if len(bits)>2 and bits[2]=='PK':
            # NB: if there are multiple PK's specified, LAST will be used!
            pk = bits[1]
    ps.close()
    return modelName,fldMap,pk
    
def json2djfix(paramf,inf,outf,verbose=False):

    modelName,fldMap,pk = loadParams(paramf)
    
    if verbose:
        print 'json2djfix: Model=%s\nInField,ModelField' % (modelName)
        for k,v in fldMap.items():
            print ('%s,%s' % (k,v)),
            if pk==v:
                print ' PK'
            else:
                print
        if not pk:
            print '(no pk)'
        
    rawdata = json.load(open(inf))
    dataList1 = []
    for data in rawdata:
        newdata1 = {}
        for k,v in data.items():
            assert k in fldMap, 'json2djfix: field %s not specified in paramfile?!' % (k)
            # NB: fields without model fieldname are dropped
            if fldMap[k]:
                newdata1[fldMap[k]] = v
        dataList1.append(newdata1)

    if pk:
        dataList1.sort(key = lambda d: d[pk])

    dataList2 = []
    for id,data in enumerate(dataList1):
        newdata2  = {'model': modelName, 'pk': id}
        newdata2['fields'] = data
        dataList2.append(newdata2)

    outs = open(outf,'w')
    json.dump(dataList2,outs,indent=4)
    outs.close()

    print 'json2djfix: done. %d/%d data converted' % (len(rawdata),len(dataList2))

parser = argparse.ArgumentParser(description='json2djf arguments')
parser.add_argument('paramf',type=str,help='Path to parameter file')
parser.add_argument('inf',type=str,help='Path to input JSON data file')
parser.add_argument('outf',type=str,help='Path to output fixture file')
parser.add_argument("--verbose",  action="store_true",help="increase output verbosity")

if __name__ == '__main__': 
    
    args, unknown = parser.parse_known_args()
    if args.verbose:
        print 'json2djf: arguments'
        # NB: args is a Namespace object; 
        argsDict = vars(args)
        for k,v in argsDict.items():
            print '\t%s = %s' % (k,v)
    
    if len(unknown)>0:
        print 'json2djf: huh?! Unkown arguments=', unknown
        assert False # can't do break or return here!

    json2djfix(args.paramf,args.inf,args.outf,verbose=args.verbose)

