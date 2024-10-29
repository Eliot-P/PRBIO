from opentps.core.processing.planOptimization.planOptimization import PlanOptimizer
from PRBIO.radiobiologicalOptimization.objectives.radiobiologicalEvalutation import RadiobiologicalObjective
import scipy.sparse as sp
import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import cupyx as cpx
    cupy_available = True
except:
    cupy_available = False


class RadiobiologicalPlanOptimizer(PlanOptimizer):
    """
    Class for the radiobiological plan optimization. Inherits from PlanOptimizer (OpenTPS).

    Attributes
    ----------
    plan : Plan
        The plan to optimize.
    CTD : CTDImage
        The CTD image.
    gamma : float
        Learning rate
    Nfractions : int
        Number of fractions of the treatment
    Initial_weights : numpy.ndarray (optional)
        Initial weights for the optimization
    """
    def __init__(self, plan, CTD,gamma = 1,Nfractions=1, initial_weights=None ,**kwargs):
        super().__init__(plan,**kwargs)
        self.CTD = CTD
        self.use_GPU_acceleration()
        self.initial_weigths = initial_weights
        self.gamma = gamma
        self.Nfractions = Nfractions

    def initializeBiologicalObjectiveFunction(self):
        """
        Initialize the dose biological objective function.
        """
        self.plan.planDesign.setScoringParameters()
        # crop on ROI
        roiObjectives = np.zeros(len(self.plan.planDesign.objectives.fidObjList[0].maskVec)).astype(bool)
        roiRobustObjectives = np.zeros(len(self.plan.planDesign.objectives.fidObjList[0].maskVec)).astype(bool)
        robust = False
        for objective in self.plan.planDesign.objectives.fidObjList:
            if objective.robust:
                robust = True
                roiRobustObjectives = np.logical_or(roiRobustObjectives, objective.maskVec)
            else:
                roiObjectives = np.logical_or(roiObjectives, objective.maskVec)
        roiObjectives = np.logical_or(roiObjectives, roiRobustObjectives)

        OARTotalVolume = 0
        weightTOT = 0
        for objective in self.plan.planDesign.objectives.fidObjList:
            if objective.roiName != self.plan.planDesign.objectives.targetName:
                if objective.roiName != 'BODY':
                    objective.partialVolume = np.count_nonzero(objective.maskVec)
                    OARTotalVolume += objective.partialVolume
                    weightTOT += objective.weight


        for objective in self.plan.planDesign.objectives.fidObjList:
            if objective.roiName != self.plan.planDesign.objectives.targetName:
                if objective.roiName != 'BODY':
                    objective.partialVolume /= (OARTotalVolume*weightTOT)


        beamletMatrix = sp.csc_matrix.dot(sp.diags(roiObjectives.astype(np.float32), format='csc'),self.plan.planDesign.beamlets.toSparseMatrix())

        self.plan.planDesign.beamlets.setUnitaryBeamlets(beamletMatrix)

        if robust:
            for s in range(len(self.plan.planDesign.robustness.scenarios)):
                beamletMatrix = sp.csc_matrix.dot(sp.diags(roiRobustObjectives.astype(np.float32), format='csc'),self.plan.planDesign.robustness.scenarios[s].toSparseMatrix())
            self.plan.planDesign.robustness.scenarios[s].setUnitaryBeamlets(beamletMatrix)

        objectiveFunction = RadiobiologicalObjective(self.plan,self.CTD,gamma = self.gamma, Nfractions=self.Nfractions ,xSquare=self.xSquared)
        self.functions.append(objectiveFunction)

    def optimize(self):
        """
        Optimize the plan.

        Returns
        -------
        numpy.ndarray
            The optimized weights.
        numpy.ndarray
            The total dose.
        float
            The cost.
        """
        logger.info('Prepare optimization ...')
        self.initializeBiologicalObjectiveFunction()
        if self.initial_weigths is None:
            x0 = self.initializeWeights()
        else:
            x0 = self.initial_weigths
        # Optimization

        result = self.solver.solve(self.functions, x0)
        if self.GPU_acceleration:
            logger.info('abnormal used memory: {}'.format(cp.get_default_memory_pool().used_bytes()))
            cp._default_memory_pool.free_all_blocks()

        return self.postProcess(result)