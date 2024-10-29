import matplotlib.pyplot as plt
from opentps.core.processing.planOptimization.objectives.baseFunction import BaseFunc
import numpy as np
import scipy.sparse as sp
import logging
logger = logging.getLogger(__name__)
try:
    import cupy as cp
    import cupyx as cpx
    cupy_available = True
except:
    cupy_available = False


class RadiobiologicalObjective(BaseFunc):
    """
    Class to compute the radiobiological objective function and its gradient. Inherits from BaseFunc (OpenTPS).
    Full derivations can be found at https://dial.uclouvain.be/memoire/ucl/en/object/thesis%3A48785

    Attributes
    ----------
    list: list
        list of objectives
    xSquare: bool
        if True, the beamlet weights are squared
    beamlets: sparse matrix
        beamlet matrix
    CTD: CTDImage
        CTD image
    voxelsize: float
        voxel size
    gamma: float
        learning rate of the UTCP
    Nfractions: int
        number of fractions of the treatment delivery
    targetMask: np.array
        mask of the target ROI
    GPU_acceleration: bool (default=False)
        if True, the computation is done on the GPU
    """
    def __init__(self, plan, CTD,gamma = 1, Nfractions = 1, xSquare=True, expModel=True, GPU_acceleration=False):
        super().__init__()
        self.list = plan.planDesign.objectives
        self.xSquare = xSquare
        self.beamlets = plan.planDesign.beamlets.toSparseMatrix()
        self.CTD = CTD
        self.voxelsize = self.list.fidObjList[0].roi.spacing[0]*self.list.fidObjList[0].roi.spacing[1]*self.list.fidObjList[0].roi.spacing[2]
        self.expModel = expModel
        self.bestUTCP = 0
        self.gamma = gamma
        if Nfractions < 1:
            print('Nfractions must be greater than 1, Using Nfrac = 1 for the simulation')
            Nfractions = 1
        self.Nfractions = Nfractions

        self.plot = 0

        self.GPU_acceleration = GPU_acceleration


        for objective in self.list.fidObjList:
            if objective.roiName == self.list.targetName:
                self.targetMask = objective.maskVec


        if plan.planDesign.robustness.scenarios:
            self.scenariosBL = [plan.planDesign.robustness.scenarios[s].toSparseMatrix() for s in
                                range(len(plan.planDesign.robustness.scenarios))]
        else:
            self.scenariosBL = []

    def SurvivingFractionLQ(self):
        """
        Compute the surviving fraction off cells with the Linear-Quadratic model.

        Parameters
        ----------
        dose: Image3D
            dose distribution

        Returns
        -------
        SF: np.array
            surviving fraction values
        """
        dose = self.doseTotal
        SF = np.ones(self.CTD.ct.imageArray.shape).flatten(order='F')
        for objective in self.list.fidObjList:
            alpha = objective.radiosensitivity
            beta = objective.beta
            maskVec = objective.maskVec
            SF[maskVec] = np.exp((-alpha*dose[maskVec] - ((beta/self.Nfractions)*dose[maskVec]**2)))
        return SF

    def TCPiExpMap(self):
        """
        Compute the Tumor Control Probability (TCP) for the target ROI with and exponential model.

        Parameters
        ----------
        dose: Image3D
            dose distribution

        Returns
        -------
        TCPmap: np.array
            TCP values
        """
        TCPimap = np.zeros(self.CTD.ct.imageArray.shape).flatten(order='F')
        for objective in self.list.fidObjList:
            maskVec = objective.maskVec
            eta = objective.cellDensity
            TCPimap[maskVec] = np.exp(-eta*(self.SF[maskVec]))

        threshold = 1e-10
        TCPimap[TCPimap > 1-threshold] = 1-threshold
        TCPimap[TCPimap < threshold] = threshold

        return TCPimap

    def mlnTCP(self):
        """
        Compute the probabilistic Tumor Control Probability (TCP) for the target ROI with and exponential model and a probabilistic target.

        Parameters
        ----------
        dose: Image3D
            dose distribution
        CTD: CTD
            CTD distribution

        Returns
        -------
        TCPmap: flaot
            -ln(TCP) value
        """
        mlnTCP = np.zeros(self.CTD.ct.imageArray.shape).flatten(order='F')
        CTD = self.CTD.imageArray.flatten(order='F')
        for objective in self.list.fidObjList:
            if objective.roiName == self.list.targetName:
                targetmask = objective.maskVec

        mlnTCP[targetmask] = -(np.log(1-CTD[targetmask] + CTD[targetmask] * self.TCPimap[targetmask]))*objective.weight

        realTCP = np.exp(-mlnTCP)

        return mlnTCP

    def mlnNTCP(self):
        """
        Compute the Normal Tissue Complication Probability (NTCP) for the ROI with and exponential model.

        Parameters
        ----------
        dose: Image3D
            dose distribution

        Returns
        -------
        NTCPmap: np.array
            NTCP values
        """
        mlnNTCP = np.zeros(self.CTD.ct.imageArray.shape).flatten(order='F')
        for objective in self.list.fidObjList:
            if objective.roiName != self.list.targetName:
                maskVec = objective.maskVec
                Vk = objective.partialVolume
                mlnNTCP[maskVec] = -(Vk*np.log(1-self.TCPimap[maskVec]))*objective.weight


        realNTCP = np.exp(-mlnNTCP)
        print('NTCP real ',np.prod(realNTCP))

        return mlnNTCP


    def mlnUTCP(self):
        """
        Computes the uncomplicated tumor control probability (UTCP) map.
        -ln(UTCP) = -ln(TCP) - ln(1-NTCP)

        Returns
        -------
        mlnUTCP: np.array
            -ln(UTCP) values map
        """
        gamma = self.gamma
        mlnTCP = self.mlnTCP()
        mlnNTCP = self.mlnNTCP()
        realNTCP = np.exp(-mlnNTCP)
        realTCP = np.exp(-mlnTCP)

        mlnUTCP = gamma*mlnTCP + (1/gamma)*mlnNTCP
        realUTCP = np.power(np.exp(-mlnTCP),gamma) * np.power(np.exp(-mlnNTCP),1/gamma)

        if self.plot%15 == 0:
            plt.title('real UTCP : '+str(np.prod(realUTCP)) + '\n -ln(UTCP) :'+str(np.sum(mlnUTCP)))
            plt.imshow(realUTCP.reshape(self.CTD.ct.imageArray.shape,order='F')[:,:,77],cmap = 'jet')
            plt.colorbar()
            plt.savefig('UTCP_Nfrac{}.png'.format(self.Nfractions))
            plt.show()
            plt.close()

        return mlnUTCP

    def grad_mlnUTCP(self):
        """
        Compute the gradient of the -ln(UTCP) function.

        Returns
        -------
        gradmap: np.array
            gradient values
        """
        gamma = self.gamma
        CTD = self.CTD.imageArray.flatten(order='F')
        A = np.zeros(self.CTD.ct.imageArray.shape).flatten(order='F')
        B = np.zeros(self.CTD.ct.imageArray.shape).flatten(order='F')
        for objective in self.list.fidObjList:
            beta = objective.beta
            eta = objective.cellDensity
            alpha = objective.radiosensitivity
            maskVec = objective.maskVec
            if objective.roiName != self.list.targetName:
                Vk = objective.partialVolume
                Bnum = (alpha + 2*(beta/self.Nfractions)*self.doseTotal[maskVec]) * eta * Vk * self.SF[maskVec]
                Bdenum = (1 / self.TCPimap[maskVec])-1
                B[maskVec] = (Bnum / Bdenum)*objective.weight

            else:
                A1 = eta*((alpha+2*(beta/self.Nfractions)*self.doseTotal[maskVec]))
                A2 = 1/(((1/CTD[maskVec]-1)/self.TCPimap[maskVec])+1)
                A3 = self.SF[maskVec]
                A[maskVec] = -(A1*A2*A3)*objective.weight
                targetmask = maskVec

        gradmap = gamma*A + (1/gamma)*B

        if self.plot%15 == 0:
            plt.title('grad')
            plt.imshow(targetmask.reshape(self.CTD.ct.imageArray.shape,order='F')[:,:,77],cmap = 'gray',alpha=1)
            plt.imshow(gradmap.reshape(self.CTD.ct.imageArray.shape,order='F')[:,:,77],cmap = 'jet',alpha=0.5)

            plt.colorbar()
            plt.show()
            plt.savefig('grad_Nfrac{}.png'.format(self.Nfractions))
            plt.close()
        self.plot += 1

        grad = np.sum(gradmap)
        return gradmap
    def computeFidelityTCP(self, x, returnWorstCase=False):
        """
        Compute the Uncomplicated Tumor Control Probability (UTCP) function.

        Parameters
        ----------
        x: np.array
            beamlet weights

        Returns
        -------
        UTCP: float
            UTCP value
        """
        if self.xSquare:
            if self.GPU_acceleration:
                weights = cp.asarray(np.square(x).astype(np.float32))
            else:
                weights = np.square(x).astype(np.float32)
        else:
                if self.GPU_acceleration:
                    weights = cp.asarray(x.astype(np.float32))
                else:
                    weights = x.astype(np.float32)

        if self.GPU_acceleration:
            doseTotal = cp.sparse.csc_matrix.dot(self.beamlets_gpu, weights).get()
        else:
            doseTotal = sp.csc_matrix.dot(self.beamlets, weights)

        self.doseTotal = doseTotal
        self.SF = self.SurvivingFractionLQ()
        self.TCPimap = self.TCPiExpMap()

        UTCPmap = self.mlnUTCP()
        UTCP = np.sum(UTCPmap)

        return UTCP
    def computeFidelityGradTCP(self, x):
        """
        Compute the gradient of the Uncomplicated Tumor Control Probability (UTCP) function.

        Parameters
        ----------
        x: np.array
            beamlet weights

        Returns
        -------
        dgrad: np.array
            gradient values
        """
        if self.xSquare:
            if self.GPU_acceleration:
                weights = cp.asarray(np.square(x).astype(np.float32))
            else:
                weights = np.square(x).astype(np.float32)
        else:
            if self.GPU_acceleration:
                weights = cp.asarray(x.astype(np.float32))
            else:
                weights = x.astype(np.float32)

        if self.GPU_acceleration:
            doseTotal = cp.sparse.csc_matrix.dot(self.beamlets_gpu, weights).get()
        else:
            doseTotal = sp.csc_matrix.dot(self.beamlets, weights)
        if self.GPU_acceleration:
            xDiag = cpx.scipy.sparse.diags(x.astype(np.float32), format='csc')
        else:
            xDiag = sp.diags(x.astype(np.float32), format='csc')

        if self.xSquare:
            doseNominalBL = sp.csc_matrix.dot(self.beamlets, xDiag)
        else:
            doseNominalBL = self.beamlets
        doseNominalBL = sp.csc_matrix.transpose(doseNominalBL)
        doseBL = doseNominalBL

        self.doseTotal = doseTotal
        self.SF = self.SurvivingFractionLQ()
        self.TCPimap = self.TCPiExpMap()


        dUTCP = self.grad_mlnUTCP()
        dDose = doseBL
        dW = (2*xDiag)

        interestmask = np.zeros(self.CTD.ct.imageArray.shape,dtype=bool).flatten(order='F')
        interestmask[dUTCP != 0] = True

        dUTCP = np.transpose(dUTCP[interestmask])
        dDose = np.transpose(dDose[:,interestmask])

        dgrad = dUTCP @ dDose @ dW

        lr = 1#e-3
        dgrad = np.multiply(dgrad,lr)

        return dgrad


    def _eval(self, x):
        f = self.computeFidelityTCP(x)
        return f

    def _grad(self, x):
        g = self.computeFidelityGradTCP(x)
        return g