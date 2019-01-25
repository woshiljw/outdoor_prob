import numpy as np

n_points=3448

def readTmatrix(file_name):
    file=open(file_name)
    out_array=np.empty([n_points,360],dtype=float)
    row,col=0,0
    for line in file:
        linestr = line.split(',')
        col=0
        for s in linestr:
            out_array[row,col]=float(s)
            col+=1
        row+=1
        if(row==n_points) :
            break
    file.close()
    return out_array

def readfaceHdr(file_name):
    file = open(file_name)
    out_array = np.empty([n_points, 3], dtype=float)
    row, col = 0, 0
    for line in file:
        linestr = line.split(',')
        col = 0
        for s in linestr:
            out_array[row, col] = float(s)
            col += 1
        row += 1
        if(row==n_points):
            break
    file.close()
    return out_array

def readTheta_sigma(file_name):
    file=open(file_name)
    out_array = np.empty([5, 10], dtype=float)
    row, col = 0, 0
    for line in file:
        linestr = line.split(',')
        col = 0
        for s in linestr:
            out_array[row, col] = float(s)
            col += 1
        row += 1
    file.close()
    return out_array

def readTheta_mu(file_name):
    file=open(file_name)
    out_array = np.empty([5, 10], dtype=float)
    row, col = 0, 0
    for line in file:
        linestr = line.split(',')
        col = 0
        for s in linestr:
            out_array[row, col] = float(s)
            col += 1
        row += 1
    file.close()
    return out_array

def readTheta_componentProportion(file_name):
    file = open(file_name)
    out_array = np.empty([1, 5], dtype=float)
    row, col = 0, 0
    for line in file:
        linestr = line.split(',')
        col = 0
        for s in linestr:
            out_array[row, col] = float(s)
            col += 1
        row += 1
    file.close()
    return out_array

def readRho_sigma(file_name):
    file = open(file_name)
    out_array = np.empty([4, 3], dtype=float)
    row, col = 0, 0
    for line in file:
        linestr = line.split(',')
        col = 0
        for s in linestr:
            out_array[row, col] = float(s)
            col += 1
        row += 1
    file.close()
    return out_array

def readRho_mu(file_name):
    file = open(file_name)
    out_array = np.empty([4, 3], dtype=float)
    row, col = 0, 0
    for line in file:
        linestr = line.split(',')
        col = 0
        for s in linestr:
            out_array[row, col] = float(s)
            col += 1
        row += 1
    file.close()
    return out_array

def readRho_componentProportion(file_name):
    file = open(file_name)
    out_array = np.empty([1, 4], dtype=float)
    row, col = 0, 0
    for line in file:
        linestr = line.split(',')
        col = 0
        for s in linestr:
            out_array[row, col] = float(s)
            col += 1
        row += 1
    file.close()
    return out_array