"""Set color terms for HSC"""

from lsst.meas.photocal.colorterms import Colorterm
from lsst.obs.hsc.colorterms import colortermsData
Colorterm.setColorterms(colortermsData)
Colorterm.setActiveDevice("Hamamatsu")
