# This file is part of obs_subaru.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np

import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.display as afwDisplay
from lsstDebug import getDebugFrame

from lsst.meas.algorithms import (SourceDetectionTask,
                                  SubtractBackgroundTask,
                                  SubtractBackgroundConfig)


class HscSubtractBackgroundConfig(SubtractBackgroundConfig):
    """Config for HscSubtractBackgroundTask.
    """
    doPatConCorr = pexConfig.Field(
        default=True,
        dtype=bool,
        doc="Incorporate pattern continuity corrections into the background model?",
    )
    ignoredPixelMask = pexConfig.ListField(
        default=["BAD", "SAT", "INTRP", "CR", "EDGE", "DETECTED",
                 "DETECTED_NEGATIVE", "SUSPECT", "NO_DATA"],
        dtype=str,
        doc="Names of mask planes to ignore while estimating the background.",
        itemCheck=lambda x: x in afwImage.Mask().getMaskPlaneDict().keys(),
    )
    doDetection = pexConfig.Field(
        default=True,
        dtype=bool,
        doc="Detect sources prior to pattern continuity estimation?",
    )
    ampEdgeInset = pexConfig.Field(
        default=5,
        dtype=int,
        doc="Number of pixels amp edge strip inset from amp edge, used to calculate amp edge flux values.",
    )
    ampEdgeWidth = pexConfig.Field(
        default=64,
        dtype=int,
        doc="Width of amp edge strip, used to calculate amp edge flux values.",
    )


class HscSubtractBackgroundTask(SubtractBackgroundTask):
    """Calculate amp-edge pattern continuity corrections and apply to background.
    """
    ConfigClass = HscSubtractBackgroundConfig
    _DefaultName = "hscSubtractBackground"

    def run(self, exposure, background=None, stats=True, statsKeys=None):
        """Calculate amp-edge pattern continuity corrections and update the
        background model accordingly.

        Parameters
        ----------
        exposure: `lsst.afw.image.Exposure`
            calibrated exposure whose background is to be subtracted.
        background: `lsst.afw.math.BackgroundList` or None
            initial background model already subtracted from the exposure. May
            be None if no background has been subtracted.
        stats: `bool` or None
            if True, measure the mean and variance of the full background model
            and record the results in the exposures metadata.
        statsKeys: `(string, string)` or None
            key names used to store the mean and variance of the background in
            the exposures metadata. If None then use ('BGMEAN', 'BGVAR').

        Returns
        -------
        `lsst.pipe.base.Struct`
            a container containing 'background', a full and updated background
            model for the given exposure (type lsst.afw.math.BackgroundList).
        """
        if background is None:
            background = afwMath.BackgroundList()

        maskedImage = exposure.getMaskedImage()
        im = maskedImage.getImage().clone()

        # apply pattern continuity corrections
        if self.config.doPatConCorr:
            # set up ignored pixel bit mask
            bitMask = maskedImage.mask.getPlaneBitMask(self.config.ignoredPixelMask)
            sctrl = afwMath.StatisticsControl()
            sctrl.setAndMask(bitMask)
            self.log.debug(f"Ignoring mask planes: {', '.join(self.config.ignoredPixelMask)}")
            if (maskedImage.getMask().getArray() & bitMask).all():
                raise pipeBase.TaskError("All pixels masked. Cannot apply pattern continuity corrections.")

            # add detection mask plane
            if self.config.doDetection and ('DETECTED' in self.config.ignoredPixelMask or
                                            'DETECTED_NEGATIVE' in self.config.ignoredPixelMask):
                # fit and subtract temporary background to facilitate source detection
                tempFitBg = self.fitBackground(maskedImage)
                maskedImage -= tempFitBg.getImageF(self.config.algorithm, self.config.undersampleStyle)

                # run source detection task and update mask planes in place
                config = SourceDetectionTask.ConfigClass()
                config.reEstimateBackground = False
                schema = afwTable.SourceTable.makeMinimalSchema()
                detectionTask = SourceDetectionTask(config=config, schema=schema)
                table = afwTable.SourceTable.make(schema)
                _ = detectionTask.run(table, exposure, sigma=2).sources

                # restore original image in mask-updated exposure
                maskedImage.setImage(im)

            im.getArray()[(maskedImage.getMask().getArray() & bitMask) > 0] = np.nan
            amps = exposure.getDetector().getAmplifiers()

            # extract amp-edge pixel step values
            deltas = []
            ampEdgeOuter = self.config.ampEdgeInset + self.config.ampEdgeWidth
            for ii in range(1, len(amps)):
                ampA = im[amps[ii - 1].getBBox()].getArray()
                ampB = im[amps[ii].getBBox()].getArray()
                stripA = ampA[:, -ampEdgeOuter:][:, :self.config.ampEdgeWidth]
                stripB = ampB[:, :ampEdgeOuter][:, -self.config.ampEdgeWidth:]
                arrayA = np.ma.median(np.ma.array(stripA, mask=np.isnan(stripA)), axis=1)
                arrayB = np.ma.median(np.ma.array(stripB, mask=np.isnan(stripB)), axis=1)
                ampDiff = arrayA.data[~arrayA.mask & ~arrayB.mask] - arrayB.data[~arrayA.mask & ~arrayB.mask]
                ampStep = np.median(ampDiff) if len(ampDiff) > 0 else 0
                deltas.append(ampStep)

            # solve for piston values and update masked image
            A = np.array([[1.0, -1.0, 0.0, 0.0],
                          [-1.0, 2.0, -1.0, 0.0],
                          [0.0, -1.0, 2.0, -1.0],
                          [0.0, 0.0, -1.0, 1.0]])
            B = np.array([deltas[0],
                          deltas[1] - deltas[0],
                          deltas[2] - deltas[1],
                          -deltas[2]])
            pistons = np.nan_to_num(np.linalg.lstsq(A, B, rcond=None)[0])
            for amp, piston in zip(amps, pistons):
                ampIm = maskedImage.getImage()[amp.getBBox()].getArray()
                ampIm -= piston

            self.log.info(f"pattern continuity corrections: {pistons[0]:.2f}, {pistons[1]:.2f}, "
                          f"{pistons[2]:.2f}, {pistons[3]:.2f}")

        fitBg = self.fitBackground(maskedImage)
        maskedImage -= fitBg.getImageF(self.config.algorithm, self.config.undersampleStyle)

        actrl = fitBg.getBackgroundControl().getApproximateControl()
        background.append((fitBg, getattr(afwMath.Interpolate, self.config.algorithm),
                           fitBg.getAsUsedUndersampleStyle(), actrl.getStyle(),
                           actrl.getOrderX(), actrl.getOrderY(), actrl.getWeighting()))

        if stats:
            self._addStats(exposure, background, statsKeys=statsKeys)

        subFrame = getDebugFrame(self._display, "subtracted")
        if subFrame:
            subDisp = afwDisplay.getDisplay(frame=subFrame)
            subDisp.mtv(exposure, title="subtracted")

        bgFrame = getDebugFrame(self._display, "background")
        if bgFrame:
            bgDisp = afwDisplay.getDisplay(frame=bgFrame)
            bgImage = background.getImage()
            bgDisp.mtv(bgImage, title="background")

        return pipeBase.Struct(
            background=background,
        )
