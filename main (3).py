# calculate the equation of states (EoS) for nuclear matter using RMF theory
import numpy as np
import math

# -----------------------------Module parameters-----------------------------------------
sti = 5  # standard input file
sto = 6  # standard output file
file_id_in = 15
file_id_out = 25
EPSI = 1.0e-50
# Natural units conversion parameter (1 fm^{-1} = tran MeV)
tran = 197.32858145060818247
Mn = 938.9
m_e = 0.511
m_mu = 105.66

as_param, das_param, aV_param, daV_param, aTV_param, daTV_param = 0, 0, 0, 0, 0, 0

# ----------------------------------------PARAMETERS----------------------------------
t_asy = 0.0  # Asymetric strength of nuclear matter
rho_stp = 0.0  # Interval of baryon number densities in our calculation
eta = 0.0  # accuracy for iteration fields (in MeV)
eosin = np.zeros(13)


# --------Module functions for the calculation of EoS for nuclear matter----------------
# Fermi moumentum for a given number density
def nuu(nb):
    return 3 ** (1 / 3) * (nb * np.pi ** 2) ** (1 / 3)


# Baryon number density
def nn(nu):
    return nu ** 3 / np.pi ** 2 / 3


# Scalar density
def nns(nu, m):
    return m / np.pi ** 2 * (np.sqrt(nu ** 2 + m ** 2) * nu - np.log((nu + np.sqrt(nu ** 2 + m ** 2)) / m) * m ** 2) / 2


# Energy density
def ee(nu, m):
    return 1 / np.pi ** 2 * (np.sqrt(nu ** 2 + m ** 2) * nu * (2 * nu ** 2 + m ** 2) -
                             np.log((nu + np.sqrt(nu ** 2 + m ** 2)) / m) * m ** 4) / 8


# Pressure
def pp(nu, m):
    return 1 / np.pi ** 2 * (np.sqrt(nu ** 2 + m ** 2) * nu * (2 * nu ** 2 - 3 * m ** 2) +
                             3 * np.log((nu + np.sqrt(nu ** 2 + m ** 2)) / m) * m ** 4) / 24


# Density dependent coupling constants
def apoly(nb, nfix, a0, a1, a2, a3):
    return a0 + a1 * (nb - nfix) + a2 * (nb - nfix) ** 2 / 2.0 + a3 * (nb - nfix) ** 3 / 6.0


def dapoly(nb, nfix, a0, a1, a2, a3):
    return a1 + a2 * (nb - nfix) + a3 * (nb - nfix) ** 2 / 2.0


# Subroutine Field
def Field(rho_inp):
    global as_param, das_param, aV_param, daV_param, aTV_param, daTV_param
    nV = rho_inp  # Total baryon number density
    nch0 = nV
    nch0 = nV
    Mns = Mn - as_param * nV
    DMns = 0.7 * Mn
    dlt = 0.1e-2  # Accuracy for iteration fields (in MeV)
    t_asy = 0.0

    dlt_asy = 0.8
    while dlt_asy > 1e-6:
        rho_n = float(nV * (1 + t_asy)) / 2  # Neutron number density
        rho_p = float(nV * (1 - t_asy)) / 2  # Proton number density
        nTV = rho_n - rho_p
        nun = nuu(rho_n)  # Fermi moumentum for neutrons
        nup = nuu(rho_p)  # Fermi moumentum for protons
        i = 0
        # -------------------------------ITERATION STRATS------------------------------------------------
        while abs(DMns) > dlt:
            nS = nns(nun, Mns) + nns(nup, Mns)  # Scalar density
            DMns = Mn - as_param * nS - Mns
            Mns = 0.9 * (Mn - as_param * nS) + 0.1 * Mns  # Effective nucleon mass
            i += 1
            if i > 500000:
                print("Cant find convergent fields!")
                return

        SigmaR = 0.5 * (daV_param * nV ** 2 - das_param * nS ** 2 + daTV_param * nTV ** 2)
        mun = np.sqrt(nun ** 2 + Mns ** 2) + SigmaR + aV_param * nV + aTV_param * nTV
        mup = np.sqrt(nup ** 2 + Mns ** 2) + SigmaR + aV_param * nV - aTV_param * nTV
        mue = mun - mup
        nue = 0
        if mue > m_e:
            nue = np.sqrt(mue ** 2 - m_e ** 2)
        numu = 0
        if mue > m_mu:
            numu = np.sqrt(mue ** 2 - m_mu ** 2)
        rho_e = nn(nue)
        rho_mu = nn(numu)
        nch = rho_p - rho_e - rho_mu
        if nch * nch0 < 0:
            dlt_asy = 0.1 * dlt_asy
        nch0 = nch
        # t_asy = t_asy + np.sign(dlt_asy, nch)
        t_asy = t_asy + (1 if nch > 0 else -1) * dlt_asy

    Ek = ee(nun, Mns) + ee(nup, Mns) + ee(nue, m_e) + ee(numu, m_mu)
    EB = Ek + 0.5 * (as_param * nS ** 2 + aV_param * nV ** 2 + aTV_param * nTV ** 2)
    EpA = EB / nV
    PB = mun * rho_n + mup * rho_p + mue * rho_e + mue * rho_mu - EB

    eosin[0] = rho_inp
    eosin[1] = (1.0 - t_asy) / 2
    eosin[2] = EpA
    eosin[3] = EB
    eosin[4] = PB
    eosin[5] = mup
    eosin[6] = mun
    eosin[7] = mue


# print('eosin')
# print(eosin)
# change it

# with open("EoS_bstb1.dat", "a") as f:
# file_id_out.write(f"    {nV/tran**3:.10E}      {EB/tran**3:.10E}      {PB/tran**3:.10E}      {EpA:.10E}      {eosin[2]:.10E}      {Ek/tran**3:.10E}      {rho_n/tran**3:.10E}      {rho_p/tran**3:.10E}      {rho_e/tran**3:.10E}      {rho_mu/tran**3:.10E}      {nS/tran**3:.10E}      {Mns:.10E}      {mun:.10E}      {mup:.10E}      {mue:.10E}      {ass:.10E}      {aV:.10E}      {aTV:.10E}      {das:.10E}      {daV:.10E}      {daTV:.10E}      {SigmaR:.10E}\n")

# exit()


# RMF_EoS_beta program
aall = np.zeros((1, 10))
err = False

# read RMF parameters from paramet_RMF.dat
with open(r"c:\Users\huawei\Desktop\work\Paramet_RMF.dat", "r") as f:
    f.readline()  # Skip a line
    for i in range(1):
        line = f.readline().split()
        # print(line)
        #  print(aall)
        aall[i] = [float(x) for x in line]
    # print(aall[0])
prmset = 0
non = 0.1 * tran ** 3
print(non)
print(tran)
# second part
asd0 = aall[prmset, 1]
asd1 = aall[prmset, 4]
aVd0 = aall[prmset, 2]
aVd1 = aall[prmset, 5]
aTVd0 = aall[prmset, 3]
aTVd1 = aall[prmset, 6]
asd2 = aall[prmset, 7]
asd3 = 0.0
aVd2 = aall[prmset, 8]
aTVd2 = aall[prmset, 9]
n0 = 0.16 * tran ** 3
nu0 = 263.041037587349514814175442513

a0sd2 = -0.1 * asd2
a0Vd2 = 0.1 * aVd2
a0TVd2 = -0.01 * aTVd2

# set random parameters at n0
n0 = 0.16 * tran ** 3
nu0 = 263.041037587349514814175442513
a0sd2 = -0.1 * asd2
a0Vd2 = 0.1 * aVd2
a0TVd2 = -0.01 * aTVd2

a0sd0 = apoly(n0, non, asd0, asd1, asd2, 0.0)
a0sd1 = dapoly(n0, non, asd0, asd1, asd2, 0.0)
a0Vd0 = apoly(n0, non, aVd0, aVd1, aVd2, 0.0)
a0Vd1 = dapoly(n0, non, aVd0, aVd1, aVd2, 0.0)
a0TVd0 = apoly(n0, non, aTVd0, aTVd1, aTVd2, 0.0)
a0TVd1 = dapoly(n0, non, aTVd0, aTVd1, aTVd2, 0.0)

# set random parameters at n1
n1 = 2.0 * n0
a1sd2 = 0.05 * asd2
a1Vd2 = 0.05 * aVd2
a1TVd2 = 0.05 * aTVd2

a1sd0 = apoly(n1, n0, a0sd0, a0sd1, a0sd2, 0.0)
a1sd1 = dapoly(n1, n0, a0sd0, a0sd1, a0sd2, 0.0)
a1Vd0 = apoly(n1, n0, a0Vd0, a0Vd1, a0Vd2, 0.0)
a1Vd1 = dapoly(n1, n0, a0Vd0, a0Vd1, a0Vd2, 0.0)
a1TVd0 = apoly(n1, n0, a0TVd0, a0TVd1, a0TVd2, 0.0)
a1TVd1 = dapoly(n1, n0, a0TVd0, a0TVd1, a0TVd2, 0.0)
# read crust EOSs
# Assuming prmset is already defined
RMF_param = str(prmset).rjust(17)

with open(r"C:\Users\huawei\Desktop\work\Crust.dat", "r") as file_id_in:
    # Skip a line
    file_id_in.readline()

    DEN = []
    EDEN = []
    PRES = []

    for i in range(1, 10001):
        line = file_id_in.readline().split()
        if not line or float(line[0]) >= 0.1:
            break

        eosin = [float(x) for x in line]
        DEN.append(eosin[0] * tran ** 3)
        EDEN.append(eosin[3] * tran ** 3)
        PRES.append(eosin[4] * tran ** 3)

# Assuming the value of i is needed later


# ... (your existing code)

# Output the results
with open("EoS_bstb1.dat", "a") as file_id_out:
    header_line = "rho  E   P   EpA  Yp  Ek  rhon rhop  rhoe rhomu rhos  ms    mun mup mue as   aV   aTV    das   daV  daTV  SigmaR  Psy"
    file_id_out.write(header_line + "\n")

    # Calculate EOSs at non<nV<n0
    rho_stp = (n1 - n0) / 100.0
    rho_B = non
    ass = apoly(rho_B, non, asd0, asd1, asd2, 0.0)
    das = dapoly(rho_B, non, asd0, asd1, asd2, 0.0)
    aV = apoly(rho_B, non, aVd0, aVd1, aVd2, 0.0)
    daV = dapoly(rho_B, non, aVd0, aVd1, aVd2, 0.0)
    aTV = apoly(rho_B, non, aTVd0, aTVd1, aTVd2, 0.0)
    daTV = dapoly(rho_B, non, aTVd0, aTVd1, aTVd2, 0.0)
    Field(rho_B)

    while rho_B < n0:
        ass = apoly(rho_B, non, asd0, asd1, asd2, 0.0)
        das = dapoly(rho_B, non, asd0, asd1, asd2, 0.0)
        aV = apoly(rho_B, non, aVd0, aVd1, aVd2, 0.0)
        daV = dapoly(rho_B, non, aVd0, aVd1, aVd2, 0.0)
        aTV = apoly(rho_B, non, aTVd0, aTVd1, aTVd2, 0.0)
        daTV = dapoly(rho_B, non, aTVd0, aTVd1, aTVd2, 0.0)
        Field(rho_B)
        # file_id_out.write(f'    {eosin[0]/tran**3:.10E}      {eosin[3]/tran**3:.10E}      {eosin[4]/tran**3:.10E}\n')
        rho_B += rho_stp

    # Calculate EOSs at n0<nV<n1
    while rho_B < n1:
        ass = apoly(rho_B, n0, a0sd0, a0sd1, a0sd2, 0.0)
        das = dapoly(rho_B, n0, a0sd0, a0sd1, a0sd2, 0.0)
        aV = apoly(rho_B, n0, a0Vd0, a0Vd1, a0Vd2, 0.0)
        daV = dapoly(rho_B, n0, a0Vd0, a0Vd1, a0Vd2, 0.0)
        aTV = apoly(rho_B, n0, a0TVd0, a0TVd1, a0TVd2, 0.0)
        daTV = dapoly(rho_B, n0, a0TVd0, a0TVd1, a0TVd2, 0.0)
        Field(rho_B)
        # file_id_out.write(f'    {eosin[0]/tran**3:.10E}      {eosin[3]/tran**3:.10E}      {eosin[4]/tran**3:.10E}\n')
        rho_B += rho_stp

    # Calculate EOSs at nV>n1
    i = 0
    while rho_B < tran ** 3:
        ass = apoly(rho_B, n1, a1sd0, a1sd1, a1sd2, 0.0)
        das = dapoly(rho_B, n1, a1sd0, a1sd1, a1sd2, 0.0)
        aV = apoly(rho_B, n1, a1Vd0, a1Vd1, a1Vd2, 0.0)
        daV = dapoly(rho_B, n1, a1Vd0, a1Vd1, a1Vd2, 0.0)
        aTV = apoly(rho_B, n1, a1TVd0, a1TVd1, a1TVd2, 0.0)
        daTV = dapoly(rho_B, n1, a1TVd0, a1TVd1, a1TVd2, 0.0)
        Field(rho_B)
        # file_id_out.write(f'    {eosin[0]/tran**3:.10E}      {eosin[3]/tran**3:.10E}      {eosin[4]/tran**3:.10E}\n')
        rho_B += rho_stp
        i += 1

    XI = 125 - 1
    for i in range(XI):
        file_id_out.write(f'    {DEN[i] / tran ** 3}      {EDEN[i] / tran ** 3}      {PRES[i] / tran ** 3:.10E}\n')
