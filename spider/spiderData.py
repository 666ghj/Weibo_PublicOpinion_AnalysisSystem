from spiderDataPack.spiderNav import start as spiderNavStart
from spiderDataPack.spiderContent import start as spiderContentStart
from spiderDataPack.spiderComments import start as spiderCommentsStart
import os

def spiderData():
    if not os.path.exists('./nav.csv'):
        spiderNavStart()
    spiderContentStart(1,1)
    spiderCommentsStart()

if __name__ == '__main__':
    spiderData()