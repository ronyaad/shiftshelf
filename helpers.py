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

def getDifferenceGrid(labels: list, boxes: list, scores: list, planogram: list) -> list:
    
    # Remove low accuracy items
    n = len([score for score in scores if score>0.6])

    # Construct grid
    grid = getGrid(labels[:n], boxes[:n])

    wrong_items = []
    # Iterating over shelves
    for shelf_index in range(len(planogram)):
        # Iterating over items in each shelf
        for item_index in range(len(planogram[shelf_index])):
            correctItem = planogram[shelf_index][item_index]
            gridItem = grid[shelf_index][item_index]

            if correctItem != gridItem:
                wrong_items.append({correctItem:(shelf_index, item_index)})
    
    return wrong_items