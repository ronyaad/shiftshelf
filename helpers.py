import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import matplotlib
import matplotlib.patches as patches
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detecto.utils import reverse_normalize
from detecto.core import Model
def getGrid(labels, boxes):
    
    df = pd.DataFrame(boxes, columns = ['xmin', 'ymin', 'xmax', 'ymax'])
    df['labels']= labels

    df = df.sort_values('ymax')
    df.loc[(df.ymax.shift() < df.ymax - np.std(df.ymax)),'group'] = 1 # everytime the jump betweeen two row is more than 20
    # use cumsum, ffill and fillna to complete the column group and have a different number for each one
    df['group'] = df['group'].cumsum().ffill().fillna(0)

    npData = df.to_records(index=False)
    
    grid = []
    for group in set(npData['group']):
        shelf = npData[npData['group']==group]
        shelf.sort(order='xmin')
        grid.append(shelf['labels'].tolist())
    
    return grid

def getDifferenceGrid1(labels: list, boxes: list, scores: list, planogram: list) -> list:
    
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

def getDifferenceGrid(model: Model, image, planogram: list = []) -> list:

    labels, boxes, scores = model.predict(image)
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
            try:
                gridItem = grid[shelf_index][item_index]
            except:
                gridItem = ""
            if correctItem != gridItem:
                wrong_items.append({correctItem:(shelf_index, item_index)})
    
    return wrong_items

def plot_prediction_grid_with_ImageSave(model, images, dim=None, figsize=None, score_filter=0.6):
    # If not specified, show all in one column
    if dim is None:
        dim = (len(images), 1)

    if dim[0] * dim[1] != len(images):
        raise ValueError('Grid dimensions do not match size of list of images')

    fig, axes = plt.subplots(dim[0], dim[1], figsize=figsize)

    # Loop through each image and position in the grid
    index = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            image = images[index]
            preds = model.predict(image)

            # If already a tensor, reverse normalize it and turn it back
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(reverse_normalize(image))
            index += 1

            # Get the correct axis
            if dim[0] <= 1 and dim[1] <= 1:
                ax = axes
            elif dim[0] <= 1:
                ax = axes[j]
            elif dim[1] <= 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            ax.imshow(image)

            # Plot boxes and labels
            for label, box, score in zip(*preds):
                if score >= score_filter:
                    width, height = box[2] - box[0], box[3] - box[1]
                    initial_pos = (box[0], box[1])
                    rect = patches.Rectangle(initial_pos, width, height, linewidth=1,
                                             edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                    ax.text(box[0] + 5, box[1] - 10, '{}: {}'
                            .format(label, round(score.item(), 2)), color='red')
                ax.set_title('Image {}'.format(index))
    plt.savefig('result.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    path = 'result.png'