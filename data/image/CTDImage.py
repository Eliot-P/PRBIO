import copy
import sys
sys.path.append('..')

from PRBIO.data.image.utils import *



from opentps.core.processing.imageProcessing.sitkImageProcessing import *
from opentps.core.data.images import Image3D

from opentps.core.data.images import ROIMask

class CTDImage(Image3D):
    """
    Class for the clinical target distribution. Inherit from Image3D (OpenTPS)

    Attributes:
    ----------
    ct: Image3D
        CT image
    gtv: ROIMask
        Gross target volume mask
    tumorR: float (default = 5mm)
        Tumor radius
    mu: float (default = 2mm)
        Mean of the gaussian distribution
    sigma: float (default = 2mm)
        Standard deviation of the gaussian distribution
    DilateMask: bool (default = True)
        If True, the mask will be dilated by 2.5*sigma
    ctv : ROIMask
        Clinical target volume mask. Needed because OpenTPS needs a binary mask to optimize the dose. But because the CTD is probabilistic, the CTV has no clinical meaning in this situation.
    ptv : ROIMask
        Planning target volume mask. The PTV is the CTV dilated by 2.5*sigma
    target : ROIMask
        Target mask. The target is the PTV dilated by 2.5*sigma
    layersMask : list of ROIMask
        List of the different layers of the CTD. Each layer is the GTV dilated by i*sigma
    layersProba : list of float
        List of the probability of each layer of the CTD
    imageArray : np.array
        3D array of the CTD containing the probability of each voxel of being tumorous.
    """

    def __init__(self, ct:Image3D, gtv:ROIMask, tumorR = 5,mu = 2, sigma = 2, DilateMask = True):
        super().__init__(self,name = 'CTD',origin = ct.origin ,spacing = ct.spacing)
        self.sigma = sigma
        self.ct = ct
        self.gtv = gtv
        self.mu = mu
        self.sigma = sigma

        ctv = gtv.copy()
        ctv.dilateMask(2.5*sigma)
        self.ctv = ctv
        ptv = ctv.copy()
        ptv.dilateMask(2.5*sigma)
        self.ptv = ptv
        target =ptv.copy()
        target.dilateMask(2.5*sigma)
        self.target = target


        self.tumorR = tumorR
        self.layersMask = []
        self.layersProba = []

        self.imageArray =  self.computeCTD(DilateMask= DilateMask)

    def fromImage3D(cls, image, **kwargs):
        dic = {'imageArray': copy.deepcopy(image.imageArray), 'origin': image.origin, 'spacing': image.spacing,
               'angles': image.angles, 'seriesInstanceUID': image.seriesInstanceUID, 'patient': image.patient, 'name': image.name}
        dic.update(kwargs)
        return cls(**dic)

    def computeCTD(self, DilateMask= False,com = [-1,-1,-1],sigma = -1):
        """
        Compute the Clinical Target Distribution. Method is inspired by Shusharina et al.2018 (https://doi.org/10.1088/1361-6560/aacf b4).

        Parameters:
        -----------
        DilateMask: bool (default = True)
            If True, the mask will be dilated by 2.5*sigma
        com: list of float (default = [-1,-1,-1])
            Center of mass of the GTV mask
        sigma: float (default = -1)
            Standard deviation of the gaussian distribution. If -1, the sigma of the CTD will be used

        Returns:
        --------
        proba: np.array
            3D array of the CTD containing the probability of each voxel of being tumorous.

        """
        if com[0] == -1:
            com = self.gtv.centerOfMass #in mm
        if sigma == -1:
            sigma = self.sigma
        imgSize = self.ct.imageArray.shape
        maxphi = phi(0,self.mu,sigma)
        if DilateMask == False:

            x = np.arange(0,imgSize[0], 1)
            y = np.arange(0, imgSize[1] , 1)
            z = np.arange(0, imgSize[2], 1)
            X, Y, Z = np.meshgrid(x, y, z,sparse=True)
            distance = np.sqrt((X - com[1])**2 + (Y - com[0])**2 + (Z - com[2])**2)
            proba = phi(distance-self.tumorR,self.mu,sigma)/maxphi
            proba[proba>1] = 1

        elif DilateMask == True:
            proba = np.zeros(self.ct.imageArray.shape)
            proba[self.ptv.imageArray == 1] = 0.0001

            i = 10
            while i > 0:
                maskproba = self.gtv.copy()
                maskproba.dilateMask(i)
                proba[maskproba.imageArray == 1] = phi(i,self.mu,sigma)/maxphi
                #print(phi(i,self.mu,sigma)*100/maxphi)
                self.layersMask.append(maskproba)
                self.layersProba.append(phi(i,self.mu,sigma)/maxphi)
                i -= 1
            proba[self.gtv.imageArray == 1] = 1

        else:
            proba = np.zeros(self.ct.imageArray.shape)
            COM_coord = self.gtv.centerOfMass
            COM_index = self.gtv.getVoxelIndexFromPosition(COM_coord)
        #Compute for each point of ct the distance to GTV edge
            proba = np.zeros(self.ct.imageArray.shape)
            x = np.arange(0,imgSize[0], 1,)
            y = np.arange(0, imgSize[1] , 1)
            z = np.arange(0, imgSize[2], 1)
            X, Y, Z = np.meshgrid(x, y, z)
            gtv_contour = self.gtv.getBinaryContourMask().imageArray

            gtv_edge_points = np.array([X[gtv_contour==True],Y[gtv_contour==True],Z[gtv_contour==True]])
            margin = 10*sigma
            dist = np.full(self.ct.imageArray.shape,1000)

            x_range = np.arange(COM_index[0]-margin,COM_index[0]+margin+1,1)
            y_range = np.arange(COM_index[1]-margin,COM_index[1]+margin+1,1)
            z_range = np.arange(COM_index[2]-margin,COM_index[2]+margin+1,1)
        #reduce the computation by only computing the distance for point near the GTV edge (distance < 5*sigma)
            for i  in x_range:
                print('idx {} / {}'.format(i,x_range[-1]))
                for j in y_range:
                    for k in z_range:
                        idx = np.array([i,j,k])
                        #for every point of the GTV edge, compute the distance to the point of the CT
                        for l in range(len(gtv_edge_points[0])):
                            gtv_idx = gtv_edge_points[:,l]
                            distance_from_edge = np.sqrt(((idx[0]-gtv_idx[0])*self.ct.spacing[0])**2 + ((idx[1]-gtv_idx[1])*self.ct.spacing[1])**2 + ((idx[2]-gtv_idx[2])*self.ct.spacing[2])**2)

                            if distance_from_edge < dist[i,j,k]:
                                dist[i,j,k] = distance_from_edge*1

                        proba[i,j,k] = phi(dist[i,j,k],self.mu,sigma)/maxphi

            proba[self.gtv.imageArray == 1] = 1
            print(COM_index,np.mean(gtv_edge_points,axis = 1),np.std(gtv_edge_points,axis = 1))
            print(np.min(dist),np.max(dist))

        return(np.flip(proba,axis=(0,1)))


    def shiftCTDandCTV(self,dx,dy,dz,sigma = 1.8):
        """
        Shift the CTD and the CTV by dx,dy,dz mm

        Parameters:
        -----------
        dx: float
            Shift in x direction (mm)
        dy: float
            Shift in y direction (mm)
        dz: float
            Shift in z direction (mm)
        sigma: float (default = 1.8mm)
            Standard deviation of the gaussian distribution.
        """

        com = self.gtv.centerOfMass #in mm
        NewGtvMask = createCircularMask(128,128,128,center = [com[1]+dx,com[0]+dy,com[2]+dz ],radius = self.tumorR)
        shifttedCenterGtv = ROIMask(imageArray=NewGtvMask, name='GTV', origin=[0, 0, 0], spacing=self.ct.spacing)
        ctv = shifttedCenterGtv.copy()
        ctv.dilateMask(2.5*sigma)
        self.ctv = ctv
        ptv = ctv.copy()
        ptv.dilateMask(2.5*sigma)
        self.ptv = ptv
        comNewGtv = shifttedCenterGtv.centerOfMass
        self.proba = self.computeCTD(DilateMask= False,com = comNewGtv,sigma = sigma)


    def plotCTD(self):
        """
        Plot the CTD and save the figure in the current directory.
        """

        plt.contour(self.ctv.imageArray[35:95,35:95,64],colors = 'red',label = 'CTV')
        plt.contour(self.gtv.imageArray[35:95,35:95,64],colors = 'blue',label = 'GTV')
        plt.imshow(self.imageArray[35:95,35:95,64],cmap = 'jet',alpha = 0.9)
        plt.title('GTV and CTV contour (with GTV mask)')
        plt.colorbar()
        plt.show()



        plt.contour(self.ptv.imageArray[35:95,35:95,64],colors = 'orange')
        plt.contour(self.ctv.imageArray[35:95,35:95,64],colors = 'red')
        plt.contour(self.gtv.imageArray[35:95,35:95,64],colors = 'blue')
        plt.imshow(self.imageArray[35:95,35:95,64],cmap = 'jet',alpha = 0.9)
        plt.title('GTV, CTV and PTV contour (with GTV mask)')
        plt.colorbar()
        plt.savefig('CTD')
        plt.show()
        plt.close()

