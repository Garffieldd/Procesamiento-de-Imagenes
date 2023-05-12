import numpy as np

def region_growing_segmentation(image, tol):
    origin_x = 100
    origin_y = 100
    x = 1
    y = 1
    valor_medio_cluster = image[origin_x, origin_y, 20]
    #tol = 50
    segmentation = np.zeros_like(image)
    point = [origin_x,origin_y]

    tail = [point]
    evaluated = image == True

    while True:
        punto = tail.pop(0)
        
        for dx in [-x, 0, x] :
            for dy in [-y, 0, y] :
                nuevoPunto = [punto[0]+dx, punto[1]+dy]
                if((nuevoPunto[0] < 230) and ((nuevoPunto[0]) > 0) and (nuevoPunto[1] < 230) and ((nuevoPunto[1]) > 0) ):
                    if (not evaluated[nuevoPunto[0], nuevoPunto[1],20]):
                        if np.abs(valor_medio_cluster - image[nuevoPunto[0], nuevoPunto[1], 20]) < tol :
                            segmentation[nuevoPunto[0], nuevoPunto[1], 20] = 1
                            tail.append([nuevoPunto[0], nuevoPunto[1]])
                            evaluated[nuevoPunto[0], nuevoPunto[1], 20] = True
                            evaluated[punto[0], punto[1], 20] = True
                        else :
                            segmentation[nuevoPunto[0], nuevoPunto[1], 20] = 0
                            tail.append([nuevoPunto[0], nuevoPunto[1]])
                            evaluated[nuevoPunto[0], nuevoPunto[1], 20] = True
                            evaluated[punto[0], punto[1], 20] = True
        valor_medio_cluster = image[segmentation == 1].mean()

        if len(tail) == 0:
            break
    return segmentation