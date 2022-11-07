import pandas as pd
import numpy as np

def getGrid(labels, boxes):
    df = pd.DataFrame(boxes, columns = ['xmin', 'ymin', 'xmax', 'ymax'])
    df['labels']= labels

    df = df.sort_values('ymax')
    df.loc[(df.ymax.shift() < df.ymax - df.ymax * 0.1),'group'] = 1 # everytime the jump betweeen two row is more than 20
    # use cumsum, ffill and fillna to complete the column group and have a different number for each one
    df['group'] = df['group'].cumsum().ffill().fillna(0)

    npData = df.to_records(index=False)
    
    grid = []
    for group in set(npData['group']):
        shelf = npData[npData['group']==group]
        shelf.sort(order='xmin')
        grid.append(shelf['labels'].tolist())
    
    return grid