"""
Subaru-specific overrides for ProcessCcdWithFakesDriverTask
(applied before HSC-specific overrides).
"""
import os.path


config.processCcdWithFakes.load(os.path.join(os.path.dirname(__file__), "processCcdWithFakes.py"))
config.ccdKey = 'ccd'
