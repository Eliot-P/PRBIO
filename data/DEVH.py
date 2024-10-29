from opentps.core.data import DVH
from typing import Union

import numpy as np

from opentps.core.data.images import DoseImage
from opentps.core.data.images import ROIMask
from opentps.core.data import ROIContour
from opentps.core.processing.imageProcessing import resampler3D
from PRBIO.data.image.CTDImage import CTDImage


class DEVH(DVH) :
    """
    Class for the Dose-expected Volume Histogram. Inherits from DVH (OpenTPS).
    """

    def __init__(self, roiMask:Union[ROIContour, ROIMask], CTD:CTDImage, dose:DoseImage=None, prescription=None):
        """
        Constructor of the DEVH class.

        Parameters
        ----------
        roiMask: Union[ROIContour, ROIMask]
            ROI mask
        CTD: CTDImage
            CTD image
        dose: DoseImage
            Dose image
        prescription: float
            prescription dose
        """
        self.CTD = CTD

        super().__init__(roiMask,dose=dose, prescription=prescription)

    def computeDVH(self,maxDVH:float=100.0):
        """
        Compute the Dose-expected Volume Histogram. Overwrite computeDVH from DVH (OpenTPS).
        The method to compute de DEVH is described in Buti et al. 2022 (DOI:10.1002/mp.16097)

        Parameters
        ----------
        maxDVH: float (default = 100.0)
            Maximum value of the DVH
        """
        if (self._doseImage is None):
            return

        self._convertContourToROI()
        roiMask = self._roiMask
        if not(self._doseImage.hasSameGrid(self._roiMask)):
            roiMask = resampler3D.resampleImage3DOnImage3D(self._roiMask, self._doseImage, inPlace=False, fillValue=0.)
            roiMask.patient = None
        dose = self._doseImage.imageArray
        mask = np.flip(roiMask.imageArray.astype(bool), (0, 1))
        spacing = self._doseImage.spacing
        number_of_bins = 4096
        DVH_interval = [0, maxDVH]
        bin_size = (DVH_interval[1] - DVH_interval[0]) / number_of_bins
        bin_edges = np.arange(DVH_interval[0], DVH_interval[1] + 0.5 * bin_size, bin_size)
        bin_edges[-1] = maxDVH + dose.max()
        self._dose = bin_edges[:number_of_bins] + 0.5 * bin_size
        ctd = self.CTD.imageArray.reshape(mask.shape,order='F')


        d =  np.flip(dose, (0, 1))[mask]
        h, _ = np.histogram(d, bin_edges,weights=ctd[mask])
        h = np.flip(h, 0)
        h = np.cumsum(h)
        h = np.flip(h, 0)
        self._volume = h * 100 / np.sum(ctd[mask])  # volume in %
        self._volume_absolute = h * spacing[0] * spacing[1] * spacing[2] / 1000  # volume in cm3





        # compute metrics
        self._Dmean = np.mean(d)
        self._Dstd = np.std(d)
        self._Dmin = d.min() if len(d) > 0 else 0
        self._Dmax = d.max() if len(d) > 0 else 0
        self._D98 = self.computeDx(98)
        self._D95 = self.computeDx(95)
        self._D50 = self.computeDx(50)
        self._D5 = self.computeDx(5)
        self._D2 = self.computeDx(2)
        self.dataUpdatedEvent.emit()