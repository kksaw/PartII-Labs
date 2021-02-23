import numpy as np
import matplotlib.pyplot as plt
import time
from math import floor, ceil, log2
from itertools import groupby
import os
from sys import stdout as so
from bisect import bisect
import trees as trees
from vl_codes import *
from json import dump, load

'''Intro'''

H = lambda pr: -sum([pr[a]*log2(pr[a]) for a in pr])
    
def common_variables():
    frequencies = dict([(key, len(list(group))) for key, group in groupby(sorted(hamlet))])
    Nin = sum([frequencies[a] for a in frequencies])
    p = dict([(a,frequencies[a]/Nin) for a in frequencies])
    
    f = [0]    
    for a in p:
        f.append(f[-1]+p[a])
    f.pop(-1)
    f = dict([(a,mf) for a,mf in zip(p,f)])


def vl_encode(x, c):
    y = []
    for a in x:
        y.extend(c[a])
    return y
    
def vl_decode(y, xt):
    x = []
    root = [k for k in range(len(xt)) if xt[k][0]==-1]
    if len(root) != 1:
        raise NameError('Tree with no or multiple roots!')
    root = root[0]
    leaves = [k for k in range(len(xt)) if len(xt[k][1]) == 0]

    n = root
    for k in y:
        if len(xt[n][1]) < k:
            raise NameError('Symbol exceeds alphabet size in tree node')
        if xt[n][1][k] == -1:
            raise NameError('Symbol not assigned in tree node')
        n = xt[n][1][k]
        if len(xt[n][1]) == 0:
            x.append(xt[n][2])
            n = root
    return x


'''SF'''
class SF(object):
    def __init__(self, p):
        self.p = p
        self.c = self.shannon_fano(self.p)        
        
    def shannon_fano(self, p):
        p = dict(sorted([(a,p[a]) for a in p if p[a]>0.0], key = lambda el: el[1], reverse = True))
        
        f = [0]    
        for a in p:
            f.append(f[-1]+p[a])
        f.pop(-1)
        f = dict([(a,mf) for a,mf in zip(p,f)])
        
        code = {}
        for a in p:
            length = ceil(-log2(p[a]))
            codeword = []
            myf = f[a]
            for pos in range(length):
                myf *= 2
                if myf >= 1:
                    codeword.append(1)
                    myf -= 1
                else:
                    codeword.append(0)
            code[a] = codeword        
        return code
    
    def encode(self, src):
        y = vl_encode(src, self.c)
        return y
            
    def decode(self, y):
        xt = trees.code2xtree(self.c)
        x = vl_decode(y, xt)
        return x

'''Huffman'''
class HM(object):
    def __init__(self, p):
        self.p = p
        self.xt = self.huffman(self.p)
        
    def huffman(self, p):
        xt = [[-1,[], a] for a in p]
        p = [(k,p[a]) for k,a in zip(range(len(p)),p)]
    
        nodelabel = len(p)
    
        while len(p) > 1:
            p = sorted(p, key = lambda el: el[1])
            xt.append([-1, [], str(nodelabel)])
            nodelabel += 1
            
            xt[p[0][0]][0] = len(xt)-1
            xt[p[1][0]][0] = len(xt)-1
            xt[-1][1] = [p[0][0], p[1][0]]
            
            p.append((len(xt)-1, sum([p[0][1],p[1][1]])))
    
            p.pop(0)
            p.pop(0)        
    
        return xt
        
    def encode(self, src):
        c = trees.xtree2code(self.xt)
        y = vl_encode(src, c)
        return y
    
    def decode(self, y):        
        x = vl_decode(y, self.xt)
        return x
    

'''Arithmetic'''
class AM(object):
    def __init__(self, p, precision=32):
        self.precision = precision
        self.one = int(2**precision - 1)
        self.quarter = int(ceil(self.one/4))
        self.half = 2*self.quarter
        self.threequarters = 3*self.quarter
        
        p = dict([(a,p[a]) for a in p if p[a]>0])

        f = [0]    
        for a in p:
            f.append(f[-1]+p[a])
        f.pop(-1)
        
        self.p = p
        self.f = f

    def encode(self, src):
        p,f = self.p, self.f
        f = dict([(a,mf) for a,mf in zip(p,f)])
        one, quarter, half, threequarters = self.one, self.quarter, self.half, self.threequarters

        y = []
        lo,hi = 0, one
        straddle = 0
    
        for k in range(len(src)): 
            if k % 1000 == 0:
                so.write('Arithmetic encoded %d%%    \r' % int(floor(k/len(src)*100)))
                so.flush()
            
            a = src[k]
            lohi_range = hi-lo + 1
            lo = lo + ceil(lohi_range*f[a])
            hi = lo + floor(lohi_range*p[a])
    
            if (lo == hi):
                raise NameError('Zero interval!')
    
            while True:
                if hi < half: # if lo < hi < 1/2
                    y.append(0)
                    y.extend(straddle*[1])
                    straddle = 0
                   
                elif lo >= half: # if hi > lo >= 1/2
                    y.append(1)
                    y.extend(straddle*[0])
                    straddle = 0
                    lo -= half
                    hi -= half
                    
                elif lo >= quarter and hi < threequarters: # if 1/4 < lo < hi < 3/4
                    straddle += 1
                    lo -= quarter
                    hi -= quarter
                    
                else:
                    break 
                
                lo *= 2
                hi = hi*2 + 1
    
        straddle += 1 
        if lo < quarter: 
            y.append(0)
            y.extend(straddle*[1])        
        else:
            y.append(1)
            y.extend(straddle*[0])
    
        return y
    
    def decode(self, y, n):
        precision, one, quarter, half, threequarters = self.precision, self.one, self.quarter, self.half, self.threequarters
        p, f = self.p, self.f
        alphabet = list(p)
        p = list(p.values())
    
        y.extend(precision*[0])
        x = n*[0]
    
        value = int(''.join(str(a) for a in y[0:precision]), 2) 
        y_position = precision
        lo,hi = 0, one
    
        x_position = 0
        while 1:
            if x_position % 1000 == 0:
                so.write('Arithmetic decoded %d%%    \r' % int(floor(x_position/n*100)))
                so.flush()
    
            lohi_range = hi - lo + 1
            a = bisect(f, (value-lo)/lohi_range) - 1
            x[x_position] = alphabet[a]
            
            lo = lo + int(ceil(f[a]*lohi_range))
            hi = lo + int(floor(p[a]*lohi_range))
            if (lo == hi):
                raise NameError('Zero interval!')
    
            while True:
                if hi < half:
                    pass
                elif lo >= half:
                    lo = lo - half
                    hi = hi - half
                    value = value - half
                elif lo >= quarter and hi < threequarters:
                    lo = lo - quarter
                    hi = hi - quarter
                    value = value - quarter
                else:
                    break
                lo = 2*lo
                hi = 2*hi + 1
                value = 2*value + y[y_position]
                y_position += 1
                if y_position == len(y):
                    break
            
            x_position += 1    
            if x_position == n or y_position == len(y):
                break
            
        return x
    

'''Adaptive arithmetic'''
class AAM(object):
    def __init__(self, delta=1, precision=32):
        self.delta = delta
        self.precision = 32
        self.one = int(2**self.precision - 1)
        self.quarter = int(ceil(self.one/4))
        self.half = 2*self.quarter
        self.threequarters = 3*self.quarter
    
    def estimating_probabilities(self):
        def calculateLav(self, src, delta, N=10000):
            f = [delta]*256
            Ltot = 0
            Lav = []
            for k in range(N):
                p = [x/sum(f) for x in f]
                Ltot += -log2(p[ord(src[k])])
                Lav.append(Ltot/(k+1))
                f[ord(src[k])] += 1
            return Lav
    
        Ldelta = []
        dList = [0.01,0.5,1.0,5.0]    
        for d in dList:
            Ldelta.append(self.calculateLav(d))    

    def encode(self, src, delta=1):
        one, quarter, half, threequarters = self.one, self.quarter, self.half, self.threequarters
        
        f = [delta]*256
        p = [x/sum(f) for x in f]
        
        y = []
        lo,hi = 0, one
        straddle = 0
    
        for k in range(len(src)): # for every symbol
            if k % 1000 == 0:
                so.write('Arithmetic encoded %d%%    \r' % int(floor(k/len(src)*100)))
                so.flush()
    
            a = src[k]
            
            lohi_range = hi-lo + 1
            lo = lo + ceil(lohi_range*sum(p[:a]))
            hi = lo + floor(lohi_range*p[a])
            
            if (lo == hi):
                raise NameError('Zero interval!')
    
            while True:
                if hi < half: # if lo < hi < 1/2
                    y.append(0)
                    y.extend(straddle*[1])
                    straddle = 0
                   
                elif lo >= half: # if hi > lo >= 1/2
                    y.append(1)
                    y.extend(straddle*[0])
                    straddle = 0
                    lo -= half
                    hi -= half
                    
                elif lo >= quarter and hi < threequarters: # if 1/4 < lo < hi < 3/4
                    straddle += 1
                    lo -= quarter
                    hi -= quarter
                    
                else:
                    break 
                
                lo *= 2
                hi = hi*2 + 1
            
            f[a]+=1        
            p = [x/sum(f) for x in f]
    
        straddle += 1 
        if lo < quarter:
            y.append(0)
            y.extend(straddle*[1])        
        else:
            y.append(1)
            y.extend(straddle*[0])
    
        return y
    
    
    def decode(self, y, n, delta=1):
        precision, one, quarter, half, threequarters = self.precision, self.one, self.quarter, self.half, self.threequarters
        
        f = [delta]*256
        p = [x/sum(f) for x in f]
        cf = list(np.cumsum(p[:-1]))
        cf.insert(0,0)
    
        lo,hi = 0, one
        
        y.extend(precision*[0])
        x = n*[0]
    
        value = int(''.join(str(a) for a in y[0:precision]), 2) 
    
        lo,hi = 0,one
        x_position, y_position = 0, precision
        
        while 1:
            if x_position % 1000 == 0:
                so.write('Arithmetic decoded %d%%    \r' % int(floor(x_position/n*100)))
                so.flush()
            
            lohi_range = hi-lo + 1
            a = bisect(cf, (value-lo)/lohi_range)-1
            x[x_position] = a
                    
            lo = lo + ceil(cf[a]*lohi_range)
            hi = lo + floor(p[a]*lohi_range)
            
            if (lo == hi):
                raise NameError('Zero interval!')
    
            while True:
                if hi < half:
                    pass
                elif lo >= half:
                    lo -= half
                    hi -= half
                    value -= half
                elif lo >= quarter and hi < threequarters:
                    lo -= quarter
                    hi -= quarter
                    value -= quarter
                else:
                    break
                
                lo = 2*lo
                hi = 2*hi + 1
                value = 2*value + y[y_position]
                y_position += 1
                if y_position == len(y):
                    break
            
            x_position += 1    
            f[a]+=1
            p = [x/sum(f) for x in f]
            cf = list(np.cumsum(p[:-1]))
            cf.insert(0,0)
            
            if x_position == n or y_position == len(y):
                break
            
        return x

'''Contextual adaptive arithmetic'''
class CAAM(object):
    def __init__(self, delta=1, precision=32):
        self.delta = delta
        self.precision = 32
        self.one = int(2**self.precision - 1)
        self.quarter = int(ceil(self.one/4))
        self.half = 2*self.quarter
        self.threequarters = 3*self.quarter
    
    def encode(self, src, delta=1):
        one, quarter, half, threequarters = self.one, self.quarter, self.half, self.threequarters
        
        p = np.array([[delta/256]*256]*256)
        jk = np.cumsum(p[0])[:-1]
        f = np.zeros((256,256))
        for i in range(255):
            f[1:,i] = jk
        
        counter = np.ones((256,256))
        ind2 = 0
            
        y = []
        lo,hi = 0, one
        straddle = 0
    
        for k in range(len(src)): # for every symbol
            if k % 1000 == 0:
                so.write('Arithmetic encoded %d%%    \r' % int(floor(k/len(src)*100)))
                so.flush()
            
            lohi_range = hi-lo + 1
            
            ind1 = ind2 if ind2!=0 else 1
            ind2 = src[k]
            
            lo = lo + ceil(lohi_range*f[ind2,ind1])
            hi = lo + floor(lohi_range*p[ind2,ind1])
            
            if (lo == hi):
                raise NameError('Zero interval!')
    
            while True:
                if hi < half: # if lo < hi < 1/2
                    y.append(0)
                    y.extend(straddle*[1])
                    straddle = 0
                   
                elif lo >= half: # if hi > lo >= 1/2
                    y.append(1)
                    y.extend(straddle*[0])
                    straddle = 0
                    lo -= half
                    hi -= half
                    
                elif lo >= quarter and hi < threequarters: # if 1/4 < lo < hi < 3/4
                    straddle += 1
                    lo -= quarter
                    hi -= quarter
                    
                else:
                    break 
                
                lo *= 2
                hi = hi*2 + 1
            
            if k > 0:
                p[:,ind1] *= counter[ind1]
                p[ind2,ind1] += 1
                counter[ind1] += 1
                p[:,ind1] /= counter[ind1]
                jk = np.cumsum(p[0:-1,ind1])
                f[1:,ind1] = jk
    
        straddle += 1 
        if lo < quarter:
            y.append(0)
            y.extend(straddle*[1])        
        else:
            y.append(1)
            y.extend(straddle*[0])
    
        return y
    
    
    def decode(self, y, n, delta=1):
        precision, one, quarter, half, threequarters = self.precision, self.one, self.quarter, self.half, self.threequarters
                
        p = np.array([[delta/256]*256]*256)
        jk = np.cumsum(p[0])[:-1]
        f = np.zeros((256,256))
        for i in range(255):
            f[1:,i] = jk
        
        counter = np.ones((256,256))
        ind2 = 0
        
        lo,hi = 0, one
        y.extend(precision*[0])
        x = n*[0]
    
        value = int(''.join(str(a) for a in y[0:precision]), 2) 
    
        lo,hi = 0,one
        x_position, y_position = 0, precision
        
        for k in range(n):
            if x_position % 1000 == 0:
                so.write('Arithmetic decoded %d%%    \r' % int(floor(x_position/n*100)))
                so.flush()
            
            lohi_range = hi-lo + 1
            
            ind1 = ind2 if ind2!=0 else 1
            ind2 = bisect(f[:,ind1], (value-lo)/lohi_range)-1
            
            x[x_position] = ind2
                    
            lo = lo + ceil(f[ind2,ind1]*lohi_range)
            hi = lo + floor(p[ind2,ind1]*lohi_range)
            
            if (lo == hi):
                raise NameError('Zero interval!')
    
            while True:
                if hi < half:
                    pass
                elif lo >= half:
                    lo -= half
                    hi -= half
                    value -= half
                elif lo >= quarter and hi < threequarters:
                    lo -= quarter
                    hi -= quarter
                    value -= quarter
                else:
                    break
                
                lo = 2*lo
                hi = 2*hi + 1
                value = 2*value + y[y_position]
                y_position += 1
                if y_position == len(y):
                    break
            
            x_position += 1    
            
            if k > 0:
                p[:,ind1] *= counter[ind1]
                p[ind2,ind1] += 1
                counter[ind1] += 1
                p[:,ind1] /= counter[ind1]
                jk = np.cumsum(p[0:-1,ind1])
                f[1:,ind1] = jk
            
            if x_position == n or y_position == len(y):
                break        
            
        return x

'''main'''
def camzip(method, filename):
    with open(filename, 'rb') as fin:
        x = fin.read()
        fin.close

    frequencies = dict([(key, len(list(group))) for key, group in groupby(sorted(x))])
    n = sum([frequencies[a] for a in frequencies])
    p = dict([(a,frequencies[a]/n) for a in frequencies])

    methods = {'shannon_fano':  SF(p),
               'huffman':       HM(p),
               'arithmetic':    AM(p),
               'adaptive arithmetic': AAM(),
               'context AAM': CAAM()}
    
    if method not in methods.keys():
        raise NameError('Compression method %s unknown' % method)
        
    m = methods[method]
    
    start = time.perf_counter()
    y = m.encode(x)
    end = time.perf_counter()
    
    H = len(y)/n    
    T = end-start

    y = bytes(bits2bytes(y))
    
    outfile = filename + '.cz' + method[:2]

    with open(outfile, 'wb') as fout:
        fout.write(y)

    pfile = filename + '.czp'
    n = len(x)

    with open(pfile, 'w') as fp:
        dump(frequencies, fp)
        
    print([H,T])
        
def camunzip(filename):
    pfile = filename[:-2] + 'p'
    with open(pfile, 'r') as fp:
        frequencies = load(fp)
    n = sum([frequencies[a] for a in frequencies])
    p = dict([(int(a),frequencies[a]/n) for a in frequencies])

    methods = {'sh':SF(p),
               'hu':HM(p),
               'ar':AM(p),
               'ad':AAM(),
               'co': CAAM()}
    
    if filename[-2:] not in methods.keys():
        raise NameError('Unknown compression method')

    m = methods[filename[-2:]]
    
    with open(filename, 'rb') as fin:
        y = fin.read()
    y = bytes2bits(y)
    
    if filename[-2:] == 'sh' or filename[-2:] == 'hu':
        start = time.perf_counter()
        x = m.decode(y)
        end = time.perf_counter()
        
    elif filename[-2:] == 'ar' or filename[-2:] == 'ad' or filename[-2:] == 'CA':
        start = time.perf_counter()
        x = m.decode(y, n)
        end = time.perf_counter()
        
    else:
        raise NameError('This will never happen (famous last words)')
        
    T = end-start
    print([T])
        
    # '.cuz' for Cam UnZipped (don't want to overwrite the original file...)
    outfile = filename[:-5] + '.cuz' 

    with open(outfile, 'wb') as fout:
        fout.write(bytes(x))

def main():
    methods = ['shannon_fano', 'huffman', 'arithmetic', 'adaptive arithmetic', 'context AAM']
    files = []
    for filename in os.listdir():
        with open(filename, 'r') as f:
            files.append(f.read())
            f.close
    
    for m in methods:
        camzip(m,f)
        camunzip(m)
