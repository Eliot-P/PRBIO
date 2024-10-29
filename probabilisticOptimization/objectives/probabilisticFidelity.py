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


class ProbabilisticFidelity(BaseFunc):
    """
    Class to compute the probabilistic dose fidelity cost function. Inherit from BaseFunc (OpenTPS).
    Dose fidelity cost function is based on Buti et al. 2022 (DOI:10.1002/mp.16097).

    Attributes
    ----------
    plan: Plan
        Plan object
    CTD: CTDImage
        CTD image
    xSquare: bool (default = True)
        If True, the beamlet weights are squared
    GPU_acceleration: bool (default = False)
        If True, the computation is done in GPU
    list: list
        List of objectives
    beamlets: scipy.sparse.csc_matrix
        Beamlets matrix
    targetMask: np.ndarray
        Target mask
    scenariosBL: list
        List of scenarios (!Robust optimisation is not supported yet)
    """

    def __init__(self, plan,CTD,xSquare=True,GPU_acceleration = False):
        super().__init__()
        self.list = plan.planDesign.objectives
        self.xSquare = xSquare
        self.beamlets = plan.planDesign.beamlets.toSparseMatrix()
        self.CTD = CTD
        self.GPU_acceleration = GPU_acceleration
        if GPU_acceleration:
            if cupy_available :
                logger.info('cupy imported and will be used in Probabilisticfidelity with version : {}'.format(cp.__version__))
                self.beamlets_gpu = cpx.scipy.sparse.csc_matrix(self.beamlets.astype(np.float32))
            else:
                self.GPU_acceleration = False
        for objective in self.list.fidObjList:
            if objective.roiName == self.list.targetName:
                self.targetMask = objective.maskVec

        self.CTD.imageArray = np.ndarray.flatten(self.CTD.imageArray, 'F')
        if plan.planDesign.robustness.scenarios:
            self.scenariosBL = [plan.planDesign.robustness.scenarios[s].toSparseMatrix() for s in
                                range(len(plan.planDesign.robustness.scenarios))]
        else:
            self.scenariosBL = []

    def unload_blGPU(self):
        """
        Unload the beamlets from the GPU
        """
        del self.beamlets_gpu
        cp._default_memory_pool.free_all_blocks()
    def computeFidelityFunction(self,x, returnWorstCase=False):
        """
        Compute the dose fidelity cost function.

        Parameters
        ----------
        x: np.ndarray
            Beamlet weights
        returnWorstCase: bool (default = False)
            If True, the worst case scenario is returned
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

        fTot = 0.0
        fTotScenario = 0.0
        scenarioList = []
        # compute objectives for nominal scenario
        if self.GPU_acceleration:
            doseTotal = cp.sparse.csc_matrix.dot(self.beamlets_gpu, weights)
        else:
            doseTotal = sp.csc_matrix.dot(self.beamlets, weights)

        for objective in self.list.fidObjList:
            if self.GPU_acceleration:
                if objective.roiName == self.list.targetName:
                    if objective.metric == objective.Metrics.DMAX:
                        f = np.sum(self.CTD.imageArray[objective.maskVec] * np.maximum(0, (
                                    doseTotal[objective.maskVec].get() - objective.limitValue)) ** 2) / np.sum(
                            self.CTD.imageArray[objective.maskVec])
                    elif objective.metric == objective.Metrics.DMIN:
                        f = np.sum(self.CTD.imageArray[objective.maskVec] * np.minimum(0, (
                                    doseTotal[objective.maskVec].get() - objective.limitValue)) ** 2) / np.sum(
                            self.CTD.imageArray[objective.maskVec])
                    else:
                        raise Exception(
                            objective.metric + ' is not supported as an objective metric for the target in probabilistic optimization')
                else:
                    if objective.metric == objective.Metrics.DMAX:
                        f = np.mean(np.maximum(0, doseTotal[objective.maskVec].get() - objective.limitValue) ** 2)
                    elif objective.metric == objective.Metrics.DMEAN:
                        f = np.maximum(0, np.mean(doseTotal[objective.maskVec].get(),
                                                  dtype=np.float32) - objective.limitValue) ** 2
                    elif objective.metric == objective.Metrics.DMIN:
                        f = np.mean(np.minimum(0, doseTotal[objective.maskVec].get() - objective.limitValue) ** 2)
                    elif objective.metric == objective.Metrics.DFALLOFF:
                        f = np.mean(np.maximum(0, doseTotal[objective.maskVec].get() - objective.voxelwiseLimitValue) ** 2)
                    else:
                        raise Exception(objective.metric + ' is not supported as an objective metric')

            else:
                if objective.roiName == self.list.targetName :
                    if objective.metric == objective.Metrics.DMAX:
                        f = np.sum(self.CTD.imageArray[objective.maskVec]*np.maximum(0, (doseTotal[objective.maskVec] - objective.limitValue)) ** 2)/np.sum(self.CTD.imageArray[objective.maskVec])
                    elif objective.metric == objective.Metrics.DMIN:
                        f = np.sum(self.CTD.imageArray[objective.maskVec]*np.minimum(0, (doseTotal[objective.maskVec] - objective.limitValue)) ** 2)/np.sum(self.CTD.imageArray[objective.maskVec])
                    else:
                        raise Exception(objective.metric + ' is not supported as an objective metric for the target in probabilistic optimization')
                else:
                    if objective.metric == objective.Metrics.DMAX:
                        f = np.mean(np.maximum(0, doseTotal[objective.maskVec] - objective.limitValue) ** 2)
                    elif objective.metric == objective.Metrics.DMEAN:
                        f = np.maximum(0,np.mean(doseTotal[objective.maskVec],dtype=np.float32) - objective.limitValue) ** 2
                    elif objective.metric == objective.Metrics.DMIN:
                        f = np.mean(np.minimum(0, doseTotal[objective.maskVec] - objective.limitValue) ** 2)
                    elif objective.metric == objective.Metrics.DFALLOFF:
                        f = np.mean(np.maximum(0, doseTotal[objective.maskVec] - objective.voxelwiseLimitValue) ** 2)
                    else:
                        raise Exception(objective.metric + ' is not supported as an objective metric')
            if not objective.robust:
                fTot += objective.weight * f
            else:
                fTotScenario += objective.weight * f



        scenarioList.append(fTotScenario)

        # skip calculation of error scenarios if there is no robust objective
        robust = False
        for objective in self.list.fidObjList:
            if objective.robust:
                robust = True

        if self.scenariosBL == [] or robust is False:
            if not returnWorstCase:
                return fTot
            else:
                return fTot, -1  # returns id of the worst case scenario (-1 for nominal)

        # Compute objectives for error scenarios
        for ScenarioBL in self.scenariosBL:
            fTotScenario = 0.0
            doseTotal = sp.csc_matrix.dot(ScenarioBL, weights)

            for objective in self.list.fidObjList:
                if not objective.robust:
                    continue

                if objective.roiName == self.list.targetName:
                    if objective.metric == objective.Metrics.DMAX:
                        f = np.sum(self.CTD.imageArray[objective.maskVec] * np.maximum(0, (
                                doseTotal[objective.maskVec] - objective.limitValue)) ** 2) / np.sum(
                            self.CTD.imageArray[objective.maskVec])
                    elif objective.metric == objective.Metrics.DMIN:
                        f = np.sum(self.CTD.imageArray[objective.maskVec] * np.minimum(0, (
                                doseTotal[objective.maskVec] - objective.limitValue)) ** 2) / np.sum(
                            self.CTD.imageArray[objective.maskVec])
                    else:
                        raise Exception(
                            objective.metric + ' is not supported as an objective metric for the target in probabilistic optimization')
                else:
                    if objective.metric == objective.Metrics.DMAX:
                        f = np.mean(np.maximum(0, doseTotal[objective.maskVec] - objective.limitValue) ** 2)
                    elif objective.metric == objective.Metrics.DMEAN:
                        f = np.maximum(0, np.mean(doseTotal[objective.maskVec],
                                                  dtype=np.float32) - objective.limitValue) ** 2
                    elif objective.metric == objective.Metrics.DMIN:
                        f = np.mean(np.minimum(0, doseTotal[objective.maskVec] - objective.limitValue) ** 2)
                    elif objective.metric == objective.Metrics.DFALLOFF:
                        f = np.mean(np.maximum(0, doseTotal[objective.maskVec] - objective.voxelwiseLimitValue) ** 2)
                    else:
                        raise Exception(objective.metric + ' is not supported as an objective metric')

                fTotScenario += objective.weight * f

            scenarioList.append(fTotScenario)

        fTot += max(scenarioList)
        if self.GPU_acceleration:  # try to prevent OOM error
            del weights, doseTotal
            cp._default_memory_pool.free_all_blocks()

        if not returnWorstCase:
            return fTot
        else:
            return fTot, scenarioList.index(
                max(scenarioList)) - 1  # returns id of the worst case scenario (-1 for nominal)


    def computeFidelityGradient(self, x):
        """
        Compute the gradient of the dose fidelity cost function.

        Parameters
        ----------
        x: np.ndarray
            Beamlet weights
        """
        # get worst case scenario
        if self.scenariosBL:
            fTot, worstCase = self.computeFidelityFunction(x, returnWorstCase=True)
        else:
            worstCase = -1
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
            xDiag = cpx.scipy.sparse.diags(x.astype(np.float32), format='csc')
        else:
            xDiag = sp.diags(x.astype(np.float32), format='csc')
        if self.GPU_acceleration:
            doseNominal = cp.sparse.csc_matrix.dot(self.beamlets_gpu, weights)
            if self.xSquare:
                doseNominalBL = cp.sparse.csc_matrix.dot(self.beamlets_gpu, xDiag)
            else:
                doseNominalBL = self.beamlets_gpu
            if worstCase != -1:
                doseScenario = cp.sparse.csc_matrix.dot(self.scenariosBL[worstCase], weights)
                doseScenarioBL = cp.sparse.csc_matrix.dot(self.scenariosBL[worstCase], xDiag)
            dfTot = np.zeros((1, len(x)), dtype=np.float32)
        else:
            doseNominal = sp.csc_matrix.dot(self.beamlets, weights)
            if self.xSquare:
                doseNominalBL = sp.csc_matrix.dot(self.beamlets, xDiag)
            else:
                doseNominalBL = self.beamlets
            doseNominalBL = sp.csc_matrix.transpose(doseNominalBL)
            if worstCase != -1:
                doseScenario = sp.csc_matrix.dot(self.scenariosBL[worstCase], weights)
                doseScenarioBL = sp.csc_matrix.dot(self.scenariosBL[worstCase], xDiag)
                doseScenarioBL = sp.csc_matrix.transpose(doseScenarioBL)
            dfTot = np.zeros((len(x), 1), dtype=np.float32)

        for objective in self.list.fidObjList:
            if worstCase != -1 and objective.robust:
                doseTotal = doseScenario
                doseBL = doseScenarioBL
            else:
                doseTotal = doseNominal
                doseBL = doseNominalBL

            if self.GPU_acceleration:
                if objective.roiName == self.list.targetName:
                    if objective.metric == objective.Metrics.DMAX:
                        #make next code compatible with GPU and CTD
                        #f = sp.csr_matrix(np.maximum(0, (doseTotal[objective.maskVec] - objective.limitValue)*self.CTD.imageArray[objective.maskVec])/np.sum(self.CTD.imageArray[objective.maskVec]))
                        #df = f.multiply(doseBL[:, objective.maskVec])
                        #dfTot += objective.weight * df.sum(axis=1)
                        maskVec_gpu = cp.asarray(objective.maskVec)
                        limitValue_gpu = cp.asarray(objective.limitValue)
                        CTD_gpu = cp.asarray(self.CTD.imageArray[objective.maskVec])
                        f = cpx.scipy.sparse.diags(cp.maximum(0, (doseTotal[maskVec_gpu] - limitValue_gpu)*CTD_gpu) , format='csc')
                        df = cp.sparse.csc_matrix.dot(f, doseBL[maskVec_gpu, :])
                        dfTot += objective.weight * (cpx.scipy.sparse.csr_matrix.mean(df, axis=0)).get()
                    elif objective.metric == objective.Metrics.DMIN:
                        maskVec_gpu = cp.asarray(objective.maskVec)
                        limitValue_gpu = cp.asarray(objective.limitValue)
                        f = cpx.scipy.sparse.diags(cp.minimum(0, (doseTotal[maskVec_gpu] - limitValue_gpu)*CTD_gpu), format='csc')
                        df = cp.sparse.csc_matrix.dot(f, doseBL[maskVec_gpu, :])
                        dfTot += objective.weight * (cpx.scipy.sparse.csr_matrix.mean(df, axis=0)).get()

                else:
                    if objective.metric == objective.Metrics.DMAX:
                        maskVec_gpu = cp.asarray(objective.maskVec)
                        limitValue_gpu = cp.asarray(objective.limitValue)
                        f = cp.maximum(0, doseTotal[maskVec_gpu] - limitValue_gpu)
                        f = cpx.scipy.sparse.diags(f, format='csc')
                        df = cp.sparse.csc_matrix.dot(f, doseBL[maskVec_gpu, :])
                        dfTot += objective.weight * (cpx.scipy.sparse.csr_matrix.mean(df, axis=0)).get()

                    elif objective.metric == objective.Metrics.DMIN:
                        maskVec_gpu = cp.asarray(objective.maskVec)
                        limitValue_gpu = cp.asarray(objective.limitValue)
                        f = cp.minimum(0, doseTotal[maskVec_gpu] - limitValue_gpu)
                        f = cpx.scipy.sparse.diags(f, format='csc')
                        df = cp.sparse.csc_matrix.dot(f, doseBL[maskVec_gpu, :])
                        dfTot += objective.weight * (cpx.scipy.sparse.csr_matrix.mean(df, axis=0)).get()
                    elif objective.metric == objective.Metrics.DMEAN:
                        maskVec_gpu = cp.asarray(objective.maskVec)
                        limitValue_gpu = cp.asarray(objective.limitValue)
                        f = cp.maximum(0, cp.mean(doseTotal[maskVec_gpu], dtype=np.float32) - limitValue_gpu)
                        try:
                            df = cpx.scipy.sparse.csr_matrix.multiply(doseBL[maskVec_gpu, :], float(
                                f))  # inconsistent behaviour when multiplied by scalar ?cupy/_compressed
                            dfTot += objective.weight * (cpx.scipy.sparse.csr_matrix.mean(df, axis=0)).get()
                        except ValueError:
                            df = cpx.scipy.sparse.csr_matrix.multiply(doseBL[maskVec_gpu, :].T, float(
                                f))  # inconsistent behaviour when multiplied by scalar ?cupy/_compressed
                            dfTot += objective.weight * (cpx.scipy.sparse.csr_matrix.mean(df, axis=1)).get().T
                    elif objective.metric == objective.Metrics.DFALLOFF:
                        maskVec_gpu = cp.asarray(objective.maskVec)
                        voxelwiseLimitValue_gpu = cp.asarray(objective.voxelwiseLimitValue)
                        f = cp.maximum(0, doseTotal[maskVec_gpu] - voxelwiseLimitValue_gpu)
                        f = cpx.scipy.sparse.diags(f, format='csc')
                        df = cp.sparse.csc_matrix.dot(f, doseBL[maskVec_gpu, :])
                        dfTot += objective.weight * (cpx.scipy.sparse.csr_matrix.mean(df, axis=0)).get()

            else:
                if objective.roiName == self.list.targetName:
                    if objective.metric == objective.Metrics.DMAX:
                        f = sp.csr_matrix(np.maximum(0, (doseTotal[objective.maskVec] - objective.limitValue)*self.CTD.imageArray[objective.maskVec])/np.sum(self.CTD.imageArray[objective.maskVec]))
                        df = f.multiply(doseBL[:, objective.maskVec])
                        dfTot += objective.weight * df.sum(axis=1)


                    elif objective.metric == objective.Metrics.DMIN:
                        f = sp.csr_matrix(np.minimum(0, doseTotal[objective.maskVec] - objective.limitValue)*self.CTD.imageArray[objective.maskVec]/np.sum(self.CTD.imageArray[objective.maskVec]))
                        df = f.multiply(doseBL[:, objective.maskVec])
                        dfTot += objective.weight * df.sum(axis=1)

                else:
                    if objective.metric == objective.Metrics.DMAX:
                        f = np.maximum(0, doseTotal[objective.maskVec] - objective.limitValue)
                        df = sp.csr_matrix.multiply(doseBL[:, objective.maskVec], f)
                        dfTot += objective.weight * sp.csr_matrix.mean(df, axis=1)

                    elif objective.metric == objective.Metrics.DMIN:
                        f = np.minimum(0, doseTotal[objective.maskVec] - objective.limitValue)
                        df = sp.csr_matrix.multiply(doseBL[:, objective.maskVec], f)
                        dfTot += objective.weight * sp.csr_matrix.mean(df, axis=1)

                    elif objective.metric == objective.Metrics.DMEAN:
                        f = np.maximum(0, np.mean(doseTotal[objective.maskVec], dtype=np.float32) - objective.limitValue)
                        df = sp.csr_matrix.multiply(doseBL[:, objective.maskVec], f)
                        dfTot += objective.weight * sp.csr_matrix.mean(df, axis=1)

                    elif objective.metric == objective.Metrics.DFALLOFF:
                        f = np.maximum(0, doseTotal[objective.maskVec] - objective.voxelwiseLimitValue)
                        df = sp.csr_matrix.multiply(doseBL[:, objective.maskVec], f)
                        dfTot += objective.weight * sp.csr_matrix.mean(df, axis=1)

                    else:
                        raise Exception(objective.metric + ' is not supported as an objective metric')

        if self.xSquare:
            dfTot = 4 * dfTot
        else:
            dfTot = 2 * dfTot
        dfTot = np.squeeze(np.asarray(dfTot)).astype(np.float64)
        if self.GPU_acceleration:
            del weights, doseNominal, doseNominalBL
            if worstCase != -1:
                del doseScenario, doseScenarioBL
            cp._default_memory_pool.free_all_blocks()

        return dfTot
    def _eval(self, x):
        f = self.computeFidelityFunction(x)
        return f

    def _grad(self, x):
        g = self.computeFidelityGradient(x)
        return g

