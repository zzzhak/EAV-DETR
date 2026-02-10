
from .solver import BaseSolver
from .det_solver import DetSolver
from .det_solver_codrone import DetSolverCODrone


from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
    'codrone_detection': DetSolverCODrone,        
    'uav_rod_detection_oriented': DetSolverCODrone, 
}