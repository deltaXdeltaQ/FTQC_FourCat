import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import scipy as sp
from matplotlib import cm
from qutip.measurement import measure, measurement_statistics
import multiprocessing as mp
from multiprocessing import Pool

import copy
import time
import random

# Modify the multiprocessing functions
def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()

    proc = [mp.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


## Initialization
# Paramter initialization
f_dim = 56   # cavity truncation dim

chi_f = 2*np.pi*1e6
Omega = 0.3*chi_f # SNAP gate driving strengths
Delta_chi = 0*Omega # chi mismatch
chi_e = chi_f - Delta_chi
gBS = 2*chi_f # BS strength


Eps = 1e-5      # for XXrot gate: the Schmidt decompsition sigular value threshold
s = 1           # the Xrot and XXrot gate SNAP gate phase-modulated photon number truncation
# alpha = 2.34
alpha = 2.93 # amplitude of the cat

# basic operator definition
a = destroy(f_dim)
a_dagger = a.dag()
iden = identity(f_dim)

def CodeWords(alpha):
    logical_zero = ((coherent(f_dim, alpha) + coherent(f_dim, - alpha)) + (coherent(f_dim, 1j*alpha) + coherent(f_dim, - 1j*alpha))).unit()
    logical_one = ((coherent(f_dim, alpha) + coherent(f_dim, - alpha)) - (coherent(f_dim, 1j*alpha) + coherent(f_dim, - 1j*alpha))).unit()
    return logical_zero, logical_one

def EncodingMap(alpha):
    logical_zero, logical_one = CodeWords(alpha)
    return logical_zero*basis(2, 0).dag() + logical_one*basis(2,1).dag()

logical_zero, logical_one = CodeWords(alpha)
logical_p = (logical_zero + logical_one).unit()
logical_m = (logical_zero - logical_one).unit()

# encoding and decoding map
encoding = EncodingMap(alpha)
decoding = encoding.dag()

# logical operation
logical_Z = (1j*num(f_dim)*np.pi/2).expm()

# (normalized) jump operators
J_fe = tensor(projection(3,1,2),identity(f_dim))   # assume perfect chi-matching: rotating frame no extra phase term
J_eg = tensor(projection(3,0,1),(1j*num(f_dim)*np.pi/4).expm())   # apprximate the dephasing effect of decay on the bosonic system by a fixed large rotation
J_eg_bare = tensor(projection(3,0,1),identity(f_dim))
J_ph = tensor(projection(3,1,1) + 2*projection(3,2,2),identity(f_dim))
J_a = tensor(identity(3),a)

# measurement of the ancilla
Pg, Pe, Pf = tensor(ket2dm(basis(3, 0)), identity(f_dim)), tensor(ket2dm(basis(3, 1)), identity(f_dim)), tensor(ket2dm(basis(3, 2)), identity(f_dim)) 

# X-basis measurement of the ancilla
PsiPlus, PsiMinus = (basis(3, 0) + basis(3, 2)).unit(), (basis(3, 0) - basis(3, 2)).unit()
Pplus, Pminus, Pe = tensor( ket2dm(PsiPlus), identity(f_dim) ), tensor( ket2dm(PsiMinus), identity(f_dim) ), tensor(ket2dm(basis(3, 1)), identity(f_dim))

# measurement of the photon number parity of the cavity
Peven = qzero(f_dim)
Podd = qzero(f_dim)
for k1 in range(f_dim):
    if k1%2 == 0:
        Peven = Peven + projection(f_dim,k1,k1)
    else:
        Podd = Podd + projection(f_dim,k1,k1)
        
# Z-basis measurement of a cavity
P03 = qzero(f_dim)
P12 = qzero(f_dim)
for k1 in range(f_dim):
    if k1%4 == 0 or k1%4 == 3:
        P03 = P03 + projection(f_dim,k1,k1)
    else:
        P12 = P12 + projection(f_dim,k1,k1)
        

## Function definition

# single-shot (non-FT) parity measurement
def ParityMeasurement1shot(Psi, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph):
    '''
    Single-shot parity measurement.
    Parameters:
    -----------------------------------
    Input:
    Psi: f_dim*1 pure-state vector: initial state
    f_dim: dimension of each bosonic cavity
    chi_e, chi_f: the dispersive coupling strength 
    kappa0: the ancilla dissipation rate
    
    Output:
    PsiOut: f_dim*1 pure-state vector
    parity: 3-outcome indicator (0,1,2)=(p,m,e)
    '''
    
    # interaction Hamiltonian in the rotating frame
#     chi_e = chi_f
    H_intp = tensor(chi_e*projection(3,1,1) + chi_f*projection(3,2,2), num(f_dim))
    
    # dissipation rate
    kappa_fe, kappa_eg, kappa_ph = kappa_decay, kappa_decay, kappa_ph
    # kappa_a = kappa0/20
    kappa_a = kappa0
    
    # initial state
    Psi0 = tensor( (basis(3,0)+basis(3,2)).unit(), Psi)
    
    # time slices
    tmin = 0
    tmax = np.pi/chi_f
    Nt = 1000
    times = np.linspace(tmin, tmax, Nt)
    
    # Monte-Carlo simulation
    resultMC = mcsolve(H_intp, Psi0, times, [np.sqrt(kappa_eg)*J_eg_bare, np.sqrt(kappa_fe)*J_fe, \
                                        np.sqrt(kappa_ph)*J_ph, np.sqrt(kappa_a)*J_a], [], ntraj=1, \
                   progress_bar=False)
    
    rhoFinalMC = resultMC.states[0][-1]    # Size: NumShots*1
    
    # Measure the final state
    parity, new_state = measure(rhoFinalMC, [Pplus, Pminus, Pe])
    
    if parity == 2:
        PsiOut = tensor(basis(3,1).dag(), identity(f_dim))*new_state  # trace out the ancillary system
        PsiOut = (1j*(chi_e - chi_f)*tmax/2*a_dagger*a).expm()*PsiOut
    elif parity == 0:
        PsiOut = tensor(PsiPlus.dag(), identity(f_dim))*new_state
    else:  # parity == 1
        PsiOut = tensor(PsiMinus.dag(), identity(f_dim))*new_state
    
    return PsiOut, parity
 
 
# FT photon-loss correction
def FTLossQEC(Psi, f_dim, chi_e, chi_f, kappa0):
    '''
    FT photon-loss correction.
    Parameters:
    -----------------------------------
    Input:
    Psi: f_dim*1 pure-state vector: initial state
    f_dim: dimension of each bosonic cavity
    chi_e, chi_f: the dispersive coupling strength 
    kappa0: the ancilla dissipation rate
    
    Output:
    PsiOut: f_dim*1 pure-state vector
    parity: 3-outcome indicator (0,1,2)=(p,m,e)
    '''
    
    parityList = np.ones(3,dtype=np.int64)*2  # parity outcome storage
    
    for i in range(3):
        Psi1, parity = ParityMeasurement1shot(Psi, f_dim, chi_e, chi_f, kappa0)
        Psi = Psi1
        if parity == 2:  # ancilla dephased: repeat
            Psi1, parity = ParityMeasurement1shot(Psi, f_dim, chi_e, chi_f, kappa0)
            Psi = Psi1
        if parity == 2:  # dephased again: randomly assign 0/1
            parity = np.random.randint(2)
        parityList[i] = parity
    
    # majority vote to determine the parity
    counts = np.bincount(parityList)
    parity = np.argmax(counts)
    
    return Psi, parity
    
    
# FT plus state preparation by parity measurement
def FTPlusStatePrep(alpha, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph):
    '''
    FT + state preparation.
    Parameters:
    -----------------------------------
    Input:
    parity: 0: even; 1: odd
    f_dim: dimension of each bosonic cavity
    chi_e, chi_f: the dispersive coupling strength 
    kappa0: the ancilla dissipation rate
    
    Output:
    PsiOut: f_dim*1 pure-state vector: should be |+>
    '''
    
    Psi = coherent(f_dim, alpha)
    
    # if ancilla measurement outcome is e, then repeat the measurement
    PsiOut, parity = ParityMeasurement1shot(Psi, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph)    
    if parity == 2:  # the ancilla decays
        PsiOut, parity = ParityMeasurement1shot(Psi, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph)
    
    if parity == 2:  # fail again: randomly assign 0/1
        parity = np.random.randint(2)
    
    if parity == 1:  # odd parity: flip it to even parity state
        PsiOut = (a*PsiOut).unit()
        
    return PsiOut
 
 
# Logical Z measurement (non-FT) 
def LogicalZMeasurement1shot(Psi, parity, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph):
    '''
    Single-shot non-FT Z measurement.
    Parameters:
    -----------------------------------
    Input:
    Psi: f_dim*1 pure-state vector: initial state
    parity: 0: even; 1: odd
    f_dim: dimension of each bosonic cavity
    chi_e, chi_f: the dispersive coupling strength 
    kappa0: the ancilla dissipation rate
    
    Output:
    PsiOut: f_dim*1 pure-state vector
    Zvalue: 3-outcome indicator (0,1,2)=(0,1,e)
    '''
    
    # interaction Hamiltonian in the rotating frame
    # chi_e = chi_f
    H_intp = tensor(chi_e*projection(3,1,1) + chi_f*projection(3,2,2), num(f_dim))
    
    # dissipation rate
    kappa_fe, kappa_eg, kappa_ph = kappa_decay, kappa_decay, kappa_ph
    # kappa_a = kappa0/20
    kappa_a = kappa0
    
    # initial state
    if parity == 0:
        Psi0 = tensor( (basis(3,0)+basis(3,2)).unit(), Psi)
    else:
        Psi0 = tensor( (basis(3,0)+ np.exp(-1j*np.pi/2)*basis(3,2)).unit(), Psi)    
    
    # time slices
    tmin = 0
    tmax = np.pi/(2*chi_f)
    Nt = 2000
    times = np.linspace(tmin, tmax, Nt)
    
    # Monte-Carlo simulation
    resultMC = mcsolve(H_intp, Psi0, times, [np.sqrt(kappa_eg)*J_eg_bare, np.sqrt(kappa_fe)*J_fe, \
                                        np.sqrt(kappa_ph)*J_ph, np.sqrt(kappa_a)*J_a], [], ntraj=1, \
                   progress_bar=False)
    
    rhoFinalMC = resultMC.states[0][-1]    # Size: NumShots*1
    
    # Measure the final state
    Zvalue, new_state = measure(rhoFinalMC, [Pplus, Pminus, Pe])
    
    if Zvalue == 2:
        PsiOut = tensor(basis(3,1).dag(), identity(f_dim))*new_state  # trace out the ancillary system
    elif Zvalue == 0:
        PsiOut = tensor(PsiPlus.dag(), identity(f_dim))*new_state
    else:  # Zvalue == 1
        PsiOut = tensor(PsiMinus.dag(), identity(f_dim))*new_state
    
    return PsiOut, Zvalue
    
    
# FT Z-basis measurement
def FTLogicalZmeasurement(Psi, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph):
    '''
    FT Z-basis measurement.
    Parameters:
    -----------------------------------
    Input:
    Psi: f_dim*1 pure-state vector: initial state
    f_dim: dimension of each bosonic cavity
    chi_e, chi_f: the dispersive coupling strength 
    kappa0: the ancilla dissipation rate
    
    Output:
    PsiOut: f_dim*1 pure-state vector
    Zvalue: 2-value outcome (0,1)
    '''
    
    parityList = np.ones(3,dtype=np.int64)*2  # parity outcome storage
    ZvalueList = np.ones(3,dtype=np.int64)*2  # Z-basis outcome storage
    
    for i in range(3):
        # first measure parity
        parity = 2  # (0,1,2) : (p,m,e)
        while parity == 2:
            Psi1, parity = ParityMeasurement1shot(Psi, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph)
            Psi = Psi1
 
        # then Z-basis measurement
        Zvalue = 2  # (0,1,2) : (0,1,e)
        while Zvalue == 2:
            Psi2, Zvalue = LogicalZMeasurement1shot(Psi1, parity, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph)
            Psi1 = Psi2
        
        parityList[i] = parity
        ZvalueList[i] = Zvalue
    
    # majority vote to determine the Z-basis value
    counts = np.bincount(ZvalueList)
    Zvalue = np.argmax(counts)
    
    return Psi2, Zvalue
 
# idling gadget for a single bosonic mode
def TwoModeIdling(Psi, f_dim, kappa0, T):
    H = tensor(qzero(f_dim), qzero(f_dim))
     # time slices
    Nt = 2000
    times = np.linspace(0, T, Nt)
    
    kappa_a = kappa0
    # Monte-carlo simulation
    resultMC = mcsolve(H, Psi, times, [np.sqrt(kappa_a)*tensor(iden, a), np.sqrt(kappa_a)*tensor(a, iden)], [], ntraj=1, \
                   progress_bar=False) 
    PsiOut = resultMC.states[0][-1]
    return PsiOut


# SNAP gate gadget: basic element for single cavity gadgets
def SNAP(Psi, PhiVec, f_dim, Omega, chi_e, chi_f, kappa0, kappa_decay, kappa_ph, seedin1=-1, seedin2=-1):
    '''
    SNAP gate element.
    Parameters:
    -----------------------------------
    Input:
    Psi: f_dim*1 pure-state vector: initial state
    PhiVec: f_dim*1 ndarray, the phase vector
    f_dim: dimension of each bosonic cavity
    Omega: the Rabi driving strength 
    kappa0: the ancilla dissipation rate
    seed1in: random seed used for the first mcsolve
    seed2in: random seed used for the second mcsolve
    
    Output:
    PsiOut: f_dim*1 pure-state vector
    seed1: random seed used for the first mcsolve (default: None)
    seed2: random seed used for the second mcsolve (default: None)
    '''
    
    # interaction Hamiltonian in the rotating frame
    H_int = tensor(qzero(3), qzero(f_dim))
    for k in range(f_dim):
        Hk = Omega*np.exp(1j*PhiVec[k])*tensor(projection(3,2,0), projection(f_dim,k,k) )
        H_int = H_int + Hk + dag(Hk)
    H_Delta_chi = (chi_e - chi_f)*tensor(projection(3,1,1), a_dagger*a)
    H_int += H_Delta_chi
    
    # dissipation rate
    kappa_fe, kappa_eg, kappa_ph = kappa_decay, kappa_decay, kappa_ph
    # kappa_a = kappa0/20
    kappa_a = kappa0
    
    # initial state
    Psi0 = tensor(basis(3,0), Psi)

    # time slices
    tmin = 0
    tmax = np.pi/(2*np.abs(Omega))
    Nt = 1000
    times = np.linspace(tmin, tmax, Nt)
    
    if seedin1 == -1:
        option = Options(seeds=None)   # set the random seed to be None
    else:
        option = Options(seeds=seedin1)   # set the random seed to be the input one
    
    # Monte-carlo simulation
    resultMC = mcsolve(H_int, Psi0, times, [np.sqrt(kappa_eg)*J_eg, np.sqrt(kappa_fe)*J_fe, \
                                        np.sqrt(kappa_ph)*J_ph, np.sqrt(kappa_a)*J_a], [], ntraj=1, \
                   progress_bar=False, options = option)    
    # resultMC = mcsolve(H_int, Psi0, times, [J_fe, J_eg, J_ph, J_a], [], ntraj=1, progress_bar=False)
 
    # store the random seed for this round
    seed1 = resultMC.seeds
    seed2 = 0
    
    rhoFinalMC = resultMC.states[0][-1]    # Size: NumShots*1
 
    # Measure the final state; if get g, repeat the protocol
    value, new_state = measure(rhoFinalMC, [Pg, Pe, Pf])
    if value == 0:  
        Psi1 = new_state
        
        if seedin2 == -1:
            option = Options(seeds=None)   # set the random seed to be None
        else:
            option = Options(seeds=seedin2)   # set the random seed to be the input one
 
 
        resultMC1 = mcsolve(H_int, Psi1, times, [np.sqrt(kappa_eg)*J_eg, np.sqrt(kappa_fe)*J_fe, \
                                            np.sqrt(kappa_ph)*J_ph, np.sqrt(kappa_a)*J_a], [], ntraj=1, \
                            progress_bar=False, options = option)
 
        # store the random seed for this round
        seed2 = resultMC1.seeds
        
        rhoFinalMC1 = resultMC1.states[0][-1]
        value1, new_state1 = measure(rhoFinalMC1, [Pg, Pe, Pf])
        PsiOut = tensor(basis(3,value1).dag(), identity(f_dim))*new_state1 # trace out the ancillary system
    else:
        PsiOut = tensor(basis(3,value).dag(), identity(f_dim))*new_state  # trace out the ancillary system
        if value == 1:
            PsiOut = (1j*(chi_e - chi_f)*tmax/2*a_dagger*a).expm()*PsiOut
        
    return PsiOut, seed1, seed2

# BS gate gadget: basic element for XXrot gates
def BS(PsiPsi, Phi, f_dim, gBS, kappa0):
    '''
    BS gate element: H = i(a b^\dag - a^\dag b)
    U = exp(-i\Phi H)
    Typical BS: set Phi = np.pi/4.
    BS^dag: set Phi = np.pi/4, gBS -> -gBS
    
    Parameters:
    -----------------------------------
    Input:
    PsiPsi: (f_dim*f_dim)*1 pure-state vector: initial state
    Phi: 
    f_dim: dimension of each bosonic cavity
    gBS: the beam-splitter coupling strength 
    kappa0: the ancilla dissipation rate
    
    Output:
    PsiPsiOut: (f_dim*f_dim)*1 pure-state vector
    '''
    
    # define the BS Hamiltonian
    HBS = gBS*(1j*tensor(a, a_dagger) - 1j*tensor(a_dagger, a))
    
    # dissipation rate
    # kappa_fe, kappa_eg, kappa_ph = kappa0, kappa0, kappa0
    # kappa_a = kappa0/20
    kappa_a = kappa0
    
    # time slices
    tmin = 0
    tmax = Phi*1/np.abs(gBS)
    Nt = 1000  
    times = np.linspace(tmin, tmax, Nt)
    
    # Monte-Carlo simulation
    resultMC = mcsolve(HBS, PsiPsi, times, [np.sqrt(kappa_a)*tensor(a,identity(f_dim)), \
                                            np.sqrt(kappa_a)*tensor(identity(f_dim), a)], [], ntraj=1, progress_bar=False)
    Options(seeds=None)   # reset the random seeds. IMPORTANT!!
    PsiPsiOut = resultMC.states[0][-1]    # Size: NumShots*1
    
    return PsiPsiOut


# Z-rotation Z(Phi) gadget
def Zrot(Psi, Phi, f_dim, Omega, chi_e, chi_f, kappa0, kappa_decay, kappa_ph):
    '''
    Z(Phi) gate element.
    Parameters:
    -----------------------------------
    Input:
    Psi: f_dim*1 pure-state vector: initial state
    Phi: Z rotation angle
    f_dim: dimension of each bosonic cavity
    Omega: the Rabi driving strength 
    kappa0: the ancilla dissipation rate
    
    Output:
    PsiOut: f_dim*1 pure-state vector
    '''
    
    # define phase vector to be implement
    PhiVec = np.zeros(f_dim)
    for k in range(f_dim):
        if (k%4==1) or (k%4==2):
            PhiVec[k] = Phi
    
    PsiOut = SNAP(Psi, PhiVec, f_dim, Omega, chi_e, chi_f, kappa0, kappa_decay, kappa_ph)[0]
    
    return PsiOut
 
 
# X-rotation X(Phi) gadget
def Xrot(Psi, Phi, s, alpha, f_dim, Omega, chi_e, chi_f, kappa0, kappa_decay, kappa_ph):
    '''
    X(Phi) gate element.
    Parameters:
    -----------------------------------
    Input:
    Psi: f_dim*1 pure-state vector: initial state
    Phi: X rotation angle
    s: the SNAP gate phase truncation number
    alpha: the leg length of the cat
    f_dim: dimension of each bosonic cavity
    Omega: the Rabi driving strength  
    kappa0: the ancilla dissipation rate
    
    Output:
    PsiOut: f_dim*1 pure-state vector
    '''
    
    PhiVec = np.zeros(f_dim)
    for i in range(s):
        PhiVec[i] = -Phi
    
    Psi1 = displace(f_dim, alpha)*Psi
    Psi2 = SNAP(Psi1, PhiVec, f_dim, Omega, chi_e, chi_f, kappa0, kappa_decay, kappa_ph)[0]
    Psi3 = displace(f_dim, -2*alpha)*Psi2
    Psi4 = SNAP(Psi3, PhiVec, f_dim, Omega, chi_e, chi_f, kappa0, kappa_decay, kappa_ph)[0]
    PsiOut = displace(f_dim, alpha)*Psi4
    
    return PsiOut

# XX-rotation XX(Phi) gadget: Schmidt decomposition
def XXrot(PsiPsi, Phi, s, f_dim, Omega, chi_e, chi_f, gBS, kappa0, kappa_decay, kappa_ph, Eps):
    '''
    XX(Phi) gate element by Schmidt decomposition.
    Parameters:
    -----------------------------------
    Input:
    PsiPsi: (f_dim*f_dim)*1 pure-state vector: initial state
    Phi: XX rotation angle
    s: the truncation of SNAP phase modulation
    f_dim: dimension of each bosonic cavity
    Omega: the Rabi driving strength (should be positive!)
    gBS: the beam-splitter coupling strength 
    kappa0: the ancilla dissipation rate
    Eps: sigular value non-zero threshold: e.g. 1e-10
    
    Output:
    PsiOut: f_dim*1 pure-state vector
    ''' 
    
    # first get through a BS
    PsiPsi1 = BS(PsiPsi, np.pi/4, f_dim, gBS, kappa0)
    
    # then get through two SNAP gates
    # SNAP phases
    PhiVec1 = np.zeros(f_dim)
    PhiVec2 = np.zeros(f_dim)
    for i in range(s):
        PhiVec1[i] = -Phi
        PhiVec2[i] = -Phi
    
    # Schmidt decomposition for the entangled state input 
    PsiMat = np.array(PsiPsi1).reshape((f_dim, f_dim))
    U,sVec,Vh = sp.linalg.svd(PsiMat, full_matrices=False)
    V = Vh.T
    # the column vectors of UNzero and VNzero are the state components we want
    NzeroArg = np.where(np.abs(sVec) > Eps)[0]
    sNzero = sVec[NzeroArg]
    UNzero = U[:,NzeroArg]
    VNzero = V[:,NzeroArg]
    
    # run SNAP gates (without photon losses) in parallel and add them together
    PsiPsiCal = tensor(Qobj(np.zeros((UNzero.shape[0],1))), Qobj(np.zeros((VNzero.shape[0],1))) )
    # seed1, seed2 = np.random.randint(2**12), np.random.randint(2**12)
    # seed3, seed4 = np.random.randint(2**12), np.random.randint(2**12)
    seed1, seed2 = random.randint(1,2**14), random.randint(1,2**14)
    seed3, seed4 = random.randint(1,2**14), random.randint(1,2**14)
    for k in range(sNzero.size): # first go through the SNAP gate 1
        Psi1 = SNAP(Qobj(UNzero[:,k]), PhiVec1, f_dim, Omega, chi_e, chi_f, 0*kappa0, 1*kappa_decay, 1*kappa_ph, seedin1=seed1, seedin2=seed2)[0]
        Psi2 = SNAP(Qobj(VNzero[:,k]), PhiVec2, f_dim, Omega, chi_e, chi_f, 0*kappa0, 1*kappa_decay, 1*kappa_ph, seedin1=seed3, seedin2=seed4)[0]
        PsiPsiCal = PsiPsiCal + sNzero[k]*tensor(Psi1, Psi2)
    
    # # add photon losses for the SNAP gates
    PsiPsiCal = TwoModeIdling(PsiPsiCal, f_dim, kappa0, np.pi/(2*np.abs(Omega)))
    
    # finally get through BS^\dag
    PsiPsiOut = BS(PsiPsiCal, np.pi/4, f_dim, -gBS, kappa0)  # BS inversed by change the sign of driving
    
    return PsiPsiOut

# Ideal error correction gadget
def IdealCatQEC(rho, alpha, f_dim):
    '''
    Ideal QEC scheme for 4-legged cat.
    Phase EC is based on engineered dissipation.
    Photon number EC is based on parity tracking.
    
    Parameters:
    -----------------------------------
    Input:
    rho: f_dim*f_dim density matrix: initial state
    alpha: the leg length of the cat
    f_dim: dimension of each bosonic cavity
    
    Output:
    PsiOut: f_dim*1 pure-state vector
    '''
    
    # calculate the steady state
    # rho = ket2dm((a*(1j*(np.pi/10)*a_dagger*a).expm()*logical_p).unit())
 
    ## engineered dissipation to the steady state space
    L4 = a**4 - alpha**4   # 4-photon dissipation
    H0 = qzero(f_dim)
 
    # time slices
    tmin = 0
    tmax = 0.009
    Nt = 10000
    times = np.linspace(tmin, tmax, Nt)

    options = Options(atol=1e-6, rtol=1e-6)
    result = mesolve(H0, rho, times, [L4], [], options=options)
    rhoFinal = result.states[-1]

    rhoFinal = (rhoFinal + rhoFinal.dag())/2
    rhoFinal = rhoFinal/rhoFinal.tr()
 
    # then do ideal parity measurement
    parity, rhoOut = measure(rhoFinal, [Peven, Podd])

    # rhoList, prob = measurement_statistics(rhoFinal, [Peven, Podd]) # measurement statistics
    
    return rhoOut, parity


# single-shot photon-loss QEC gadgets
def Teleportation(Psi_qubit, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph):
    '''
    Simulate the concatenated photon-loss QEC circuit.
    Parameters:
    -----------------------------------
    Input:
    Psi: f_dim*1 pure-state vector: initial state
    f_dim: dimension of each bosonic cavity
    chi_e, chi_f: dispersive coupling strength
    N_seq: sequence length
    
    Output:
    FideList: N_seq*1 vector: fidelity sampling result
    AvgFockList: N_seq*1 vector: photon number sampling result
    '''
    Psi = encoding*Psi_qubit
    s = 1

    initial_state = copy.deepcopy(Psi)

    input_state = Psi
    
    # first simulate if there will be a measurement error of Z-basis measurement
    rand_s = np.random.randint(2) # randomly sample an input state as logical zero or one
    if rand_s == 0 :
        PsiRand_ini = logical_zero
    else: 
        PsiRand_ini = logical_one 
    PsiOut, Zvalue = FTLogicalZmeasurement(PsiRand_ini, f_dim, chi_e, chi_f, 1*kappa0, 1*kappa_decay, 1*kappa_ph)
    if_meas_error = int(Zvalue != rand_s)
        
    # then do the simulation of the teleportation circuit
    PsiB_ini = FTPlusStatePrep(alpha, f_dim, chi_e, chi_f, 1*kappa0, 1*kappa_decay, 1*kappa_ph)  # FT state preparation

    # single-cavity operation procedure
    PsiA1 = Xrot(input_state, np.pi/2, s, alpha, f_dim, Omega, chi_e, chi_f, 1*kappa0, 1*kappa_decay, 1*kappa_ph)
    PsiA2 = Zrot(PsiA1, np.pi/2, f_dim, Omega, chi_e, chi_f, 1*kappa0, 1*kappa_decay, 1*kappa_ph)
    PsiB1 = Zrot(PsiB_ini, np.pi/2, f_dim, Omega, chi_e, chi_f, 1*kappa0, 1*kappa_decay, 1*kappa_ph)

    # XXrot procedure
    PsiPsi1 = XXrot(tensor(PsiA2,PsiB1), np.pi/2, s, f_dim, Omega, chi_e, chi_f, gBS, 1*kappa0, 1*kappa_decay, 1*kappa_ph, Eps)

    # measurement procedure
    valueZ, new_state = measure(PsiPsi1, [tensor(P03, identity(f_dim)), tensor(P12, identity(f_dim))])
    
#     if (np.random.rand() < ErrorRate):   # simulate the Z-basis measurment error
#         valueZ = 1 - valueZ 
    if if_meas_error: # add measurement error, if present
        valueZ = 1 - valueZ 
    
    rhoOut = new_state.ptrace(1)  # trace out the ancillary system
    if valueZ == 1:
        rhoOut = logical_Z*rhoOut*(logical_Z.dag()) 
    
    alpha_decayed = np.sqrt(np.abs(expect(a_dagger*a, rhoOut)))
    decayed_encoding = EncodingMap(alpha_decayed)
    decayed_decoding = decayed_encoding.dag()
    # ideal QEC procedure
    rhoOut2, parity = IdealCatQEC(rhoOut, alpha_decayed, f_dim)
    final_corr = a**parity
     
    target_state = (final_corr*decayed_encoding*Psi_qubit).unit()
    final_fid = fidelity(target_state, rhoOut2)
    
    return 1 - np.abs(final_fid)


def SweepInfid_SingleShot(kappa0, kappa_decay, kappa_ph):
    PsiRand = np.array(rand_ket(2))
    input_state = (PsiRand[0,0]*basis(2,0) + PsiRand[1,0]*basis(2,1)).unit()

    infid = Teleportation(input_state, f_dim, chi_e, chi_f, kappa0, kappa_decay, kappa_ph)

    return infid

def SweepInfid(kappa0, kappa_decay, kappa_ph, nshot):
    eval_func = lambda _: SweepInfid_SingleShot(kappa0, kappa_decay, kappa_ph)
    infids = parmap(eval_func, [0]*nshot, nprocs = mp.cpu_count())
    
    # return np.average(infids)
    return infids

## Run the main program
nshot = 10
eta_decay = 10 # ancilla decay rate = photon loss rate * eta_decay
eta_dephasing = 2.5 # ancilla dephasing rate = photon loss rate * eta_dephasing

log_kappa0min = np.log(5e-4*Omega)
log_kappa0max = np.log(1e-2*Omega)
Nkappa = 6
kappa0List = np.exp(np.linspace(log_kappa0min, log_kappa0max, Nkappa))[:3] # photon loss rates to sweep

start_time = time.time()
infids_twoD = []
for kappa0 in kappa0List:
    infids = SweepInfid(kappa0, eta_decay*kappa0, eta_dephasing*kappa0, nshot)
    infids_twoD.append(infids)

file_path = './tel_sweep_chi_0.txt'
np.savetxt(file_path, np.array(infids_twoD))
end_time = time.time()
print('time elapsed:', end_time - start_time)
