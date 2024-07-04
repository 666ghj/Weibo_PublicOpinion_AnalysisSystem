# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from math import log, exp

def getz(r, nr):
    z = [2*nr[0]/r[1]]
    for i in xrange(len(nr)-2):
        z.append(2*nr[i+1]/(r[i+2]-r[i]))
    z.append(nr[-1]/(r[-1]-r[-2]))
    return z

def least_square(x, y): # y=a+bx
    meanx = sum(x)/len(x)
    meany = sum(y)/len(y)
    xy = sum((x[i]-meanx)*(y[i]-meany) for i in range(len(x)))
    square = sum((x[i]-meanx)**2 for i in range(len(x)))
    b = xy/square
    return (meany-b*meanx, b)

def main(dic):
    values = sorted(dic.values())
    r, nr, prob = [], [], []
    for v in values:
        if not r or r[-1] != v:
            r.append(v)
            nr.append(1)
        else:
            nr[-1] += 1
    rr = dict(map(lambda x:list(reversed(x)), enumerate(r)))
    total = reduce(lambda x, y:(x[0]*x[1]+y[0]*y[1], 1), zip(nr, r))[0]
    z = getz(r, nr)
    a, b = least_square(map(lambda x:log(x), r), map(lambda x:log(x), z))
    use_good_turing = False
    nr.append(exp(a+b*log(r[-1]+1)))
    for i in xrange(len(r)):
        good_turing = (r[i]+1)*(exp(b*(log(r[i]+1)-log(r[i]))))
        turing = (r[i]+1)*nr[i+1]/nr[i] if i+1<len(r) else good_turing
        diff = ((((r[i]+1)**2)/nr[i]*nr[i+1]/nr[i]*(1+nr[i+1]/nr[i]))**0.5)*1.65
        if not use_good_turing and abs(good_turing-turing)>diff:
            prob.append(turing)
        else:
            use_good_turing = True
            prob.append(good_turing)
    sump = reduce(lambda x, y:(x[0]*x[1]+y[0]*y[1], 1), zip(nr, prob))[0]
    for cnt, i in enumerate(prob):
        prob[cnt] = (1-nr[0]/total)*i/sump
    return nr[0]/total/total, dict(zip(dic.keys(), map(lambda x:prob[rr[x]], dic.values())))

if __name__ == '__main__':
    print(main({1:1,2:1,3:1,4:2,5:2,6:3,7:1,8:2,9:3}))
