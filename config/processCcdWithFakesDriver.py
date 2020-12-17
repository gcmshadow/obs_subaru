"""
HSC-specific overrides for ProcessCcdWithFakesDriverTask
"""
import os.path

config.processCcdWithFakes.load(os.path.join(os.path.dirname(__file__), "processCcdWithFakes.py"))
config.ccdKey = 'ccd'

config.doMakeSourceTable = True
config.doSaveWideSourceTable = True
config.transformSourceTable.load(os.path.join(os.path.dirname(__file__), "transformSourceTable.py"))
config.ignoreCcdList = [9, 104, 105, 106, 107, 108, 109, 110, 111]
