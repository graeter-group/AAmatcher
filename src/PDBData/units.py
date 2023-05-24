from openmm import unit

DISTANCE_UNIT = unit.angstrom
ENERGY_UNIT = unit.kilocalorie_per_mole
FORCE_UNIT = ENERGY_UNIT / DISTANCE_UNIT
ANGLE_UNIT = unit.radian
CHARGE_UNIT = unit.elementary_charge

FORCE_CONSTANT_UNIT = ENERGY_UNIT / (DISTANCE_UNIT ** 2)
ANGLE_FORCE_CONSTANT_UNIT = ENERGY_UNIT / (ANGLE_UNIT ** 2)