from opentps.core.data.plan import FidObjective,ObjectivesList
from opentps.core.data.images import ROIMask

class BiologicalObjectiveList(ObjectivesList):
    """
    List of Biological Objectives. Inherits from ObjectivesList (OpenTPS).
    """
    def __init__(self):
        super().__init__()

    def addBiologicalObjective(self, roi=None, weight=1., radiosensitivity =0.04, cellDensity= 1e7, beta = 0):
        """
        Add a Biological Objective to the list.

        Parameters
        ----------
        radiosensitivity: float
            radiosensitivity
        cellDensity: float
            cell density
        relatedFidObjective: FidObjective
            related Fidelity Objective
        roi: ROIMask
            ROI mask
        weight: float
            weight
        radiosensitivity: float
            radiosensitivity
        cellDensity: float
            cell density
        beta: float
            beta
        """
        objective = BiologicalObjective(roi, weight,radiosensitivity, cellDensity, beta)
        self.fidObjList.append(objective)

    def setTarget(self, roiName, prescription = 0.0):
        """
        Set the target name and prescription dose.

        Parameters
        ----------
        roiName: str
            name of the target
        prescription: float
            prescription dose
        """
        self.targetName = roiName
        if prescription == 0:
            prescription = 1
        self.targetPrescription = prescription


class BiologicalObjective(FidObjective):
    """
    Biological Objective. Inherits from FidObjective (OpenTPS).

    Attributes:
    ----------
    radiosensitivity: float
        radiosensitivity
    cellDensity: float
        cell density per voxel
    partialVolume: float
        partial volume of the ROI (used only in NTCP calculation)
    """
    def __init__(self,roi=None, weight=1., radiosensitivity= 0.4, cellDensity= 1e7, beta=0):
        super().__init__(roi = roi,metric = None,limitValue= 0, weight = weight)
        self.radiosensitivity = radiosensitivity
        self.cellDensity = cellDensity
        self.partialVolume = 1.0
        self.beta = beta



