"""respinach: a Python reimplementation of the Spinach spin dynamics kernel."""

from respinach.assume import assume
from respinach.basis import basis
from respinach.contexts import crystal, liquid, load_grid, powder, singlerot
from respinach.filters import coherence, correlation, decouple, homospoil
from respinach.hamiltonian import frqoffset, hamiltonian, orientation
from respinach.kinetics import kinetics
from respinach.operators import operator
from respinach.propagation import evolution, krylov, propagator, step
from respinach.reduction import path_trace, reduce, zte
from respinach.relaxation import equilibrium, relaxation, thermalize
from respinach.processing import apodisation
from respinach.sequences import acquire, cosy, hsqc, noesy
from respinach.states import state
from respinach.system import Chemistry, SpinSystem, create, dilute, kill_spin
from respinach.utilities import hdot

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
