from opentps.core.processing.planOptimization.planOptimization import PlanOptimizer
from PRBIO.probabilisticOptimization.objectives.probabilisticFidelity import ProbabilisticFidelity
import scipy.sparse as sp
import logging
logger = logging.getLogger(__name__)
import numpy as np

try:
    import cupy as cp
    import cupyx as cpx
    cupy_available = True
except:
    cupy_available = False


class ProbabilisticOptimizer(PlanOptimizer):
    """
    Probabilistic optimizer for plan optimization. Inherits from PlanOptimizer (OpenTPS).

    Attributes
    ----------
    plan : Plan
        The plan to optimize.
    CTD: CTDImage
        CTD image
    """

    def __init__(self, plan, CTD,**kwargs):
        super().__init__(plan,**kwargs)
        self.CTD = CTD

    def initializeFidObjectiveFunction(self):
        """
        Initialize the dose fidelity objective function.
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

        beamletMatrix = sp.csc_matrix.dot(sp.diags(roiObjectives.astype(np.float32), format='csc'),self.plan.planDesign.beamlets.toSparseMatrix())

        self.plan.planDesign.beamlets.setUnitaryBeamlets(beamletMatrix)

        if robust:
            for s in range(len(self.plan.planDesign.robustness.scenarios)):
                beamletMatrix = sp.csc_matrix.dot(sp.diags(roiRobustObjectives.astype(np.float32), format='csc'),self.plan.planDesign.robustness.scenarios[s].toSparseMatrix())
            self.plan.planDesign.robustness.scenarios[s].setUnitaryBeamlets(beamletMatrix)

        objectiveFunction = ProbabilisticFidelity(self.plan,self.CTD,self.xSquared,GPU_acceleration=self.GPU_acceleration)
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
        self.initializeFidObjectiveFunction()
        x0 = self.initializeWeights()
        # Optimization
        result = self.solver.solve(self.functions, x0)
        if self.GPU_acceleration:
            logger.info('abnormal used memory: {}'.format(cp.get_default_memory_pool().used_bytes()))
            self.functions[0].unload_blGPU()
            cp._default_memory_pool.free_all_blocks()

        return self.postProcess(result)