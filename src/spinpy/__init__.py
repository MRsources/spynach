"""spinpy: a Python reimplementation of the Spinach spin dynamics kernel."""

from spinpy.assume import assume
from spinpy.basis import basis
from spinpy.contexts import crystal, liquid, load_grid, powder, singlerot
from spinpy.filters import coherence, correlation, decouple, homospoil
from spinpy.hamiltonian import frqoffset, hamiltonian, orientation
from spinpy.kinetics import kinetics
from spinpy.operators import operator
from spinpy.propagation import evolution, krylov, propagator, step
from spinpy.reduction import path_trace, reduce, zte
from spinpy.relaxation import equilibrium, relaxation, thermalize
from spinpy.processing import apodisation
from spinpy.sequences import acquire, cosy, hsqc, noesy
from spinpy.states import state
from spinpy.system import Chemistry, SpinSystem, create, dilute, kill_spin
from spinpy.utilities import hdot

__all__ = [
    "Chemistry",
    "SpinSystem",
    "acquire",
    "apodisation",
    "assume",
    "basis",
    "coherence",
    "correlation",
    "cosy",
    "create",
    "crystal",
    "decouple",
    "dilute",
    "equilibrium",
    "evolution",
    "frqoffset",
    "hamiltonian",
    "hdot",
    "homospoil",
    "hsqc",
    "kill_spin",
    "kinetics",
    "krylov",
    "liquid",
    "load_grid",
    "noesy",
    "operator",
    "orientation",
    "path_trace",
    "powder",
    "propagator",
    "reduce",
    "relaxation",
    "singlerot",
    "state",
    "step",
    "thermalize",
    "zte",
]
