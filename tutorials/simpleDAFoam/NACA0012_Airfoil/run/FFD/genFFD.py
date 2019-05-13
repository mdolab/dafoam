import numpy as np
import sys

def writeFFDFile(fileName,nBlocks,nx,ny,nz,points):
    '''
    Take in a set of points and write the plot 3dFile
    '''

    f = open(fileName,'w')

    f.write('%d\n'%nBlocks)
    for i in range(nBlocks):
        f.write('%d %d %d '%(nx[i],ny[i],nz[i]))
    # end
    f.write('\n')
    for block in range(nBlocks):
        for k in range(nz[block]):
            for j in range(ny[block]):
                for i in range(nx[block]):
                    f.write('%f '%points[block][i,j,k,0])
                # end
            # end
        # end
        f.write('\n')

        for k in range(nz[block]):
            for j in range(ny[block]):
                for i in range(nx[block]):
                    f.write('%f '%points[block][i,j,k,1])
                # end
            # end
        # end
        f.write('\n')

        for k in range(nz[block]):
            for j in range(ny[block]):
                for i in range(nx[block]):
                    f.write('%f '%points[block][i,j,k,2])
                # end
            # end
        # end
    # end
    f.close()
    return

def returnBlockPoints(corners,nx,ny,nz):
    '''
    corners needs to be 8 x 3
    '''
    points = np.zeros([nx,ny,nz,3])

    # points 1 - 4 are the iMin face
    # points 5 - 8 are the iMax face

    for idim in range(3):
        edge1 = np.linspace(corners[0][idim],corners[4][idim],nx)
        edge2 = np.linspace(corners[1][idim],corners[5][idim],nx)
        edge3 = np.linspace(corners[2][idim],corners[6][idim],nx)
        edge4 = np.linspace(corners[3][idim],corners[7][idim],nx)

        for i in range(nx):
            edge5 = np.linspace(edge1[i],edge3[i],ny)
            edge6 = np.linspace(edge2[i],edge4[i],ny)
            for j in range(ny):
                edge7 = np.linspace(edge5[j],edge6[j],nz)
                points[i,j,:,idim] = edge7
            # end
        # end
    # end
                
    return points

################ FFD ##############
nBlocks = 1

nx = [10]
ny = [2]
nz = [2]

corners = np.zeros([nBlocks,8,3])

corners[0,0,:] = [-0.010,-.0700,0.0]
corners[0,1,:] = [-0.010,-0.0700,0.1]
corners[0,2,:] = [-0.010,0.0700,0.0]
corners[0,3,:] = [-0.010,0.0700,0.1]
corners[0,4,:] = [ 1.01,-.07,0.0]
corners[0,5,:] = [ 1.01,-.07,0.10]
corners[0,6,:] = [ 1.01,0.07,0.0]
corners[0,7,:] = [ 1.01,0.07,0.1]


points = []
for block in range(nBlocks):
    points.append(returnBlockPoints(corners[block],nx[block],ny[block],nz[block]))

#print points
fileName = 'wingFFD.xyz'
writeFFDFile(fileName,nBlocks,nx,ny,nz,points)

