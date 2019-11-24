#!/usr/bin/env python3.5

import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table as table
from astropy import units as u
from astropy import constants
from scipy.integrate import odeint
import argparse,os
import mpmath
from astropy.io import ascii
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import matplotlib.colors as colors


def astronomical_constants():
    global c, G, M_sun, T_sun
    c = constants.c.value
    G = constants.G.value
    M_sun = constants.M_sun.value
    T_sun = constants.GM_sun.value/((constants.c.value)**3)


def n_dot_to_pb_dot(n,n_dot):
    pb_dot = -2.0*(np.pi)*n_dot/(n**2)

    return pb_dot


# Advance of periastron per orbit


def adv_per(m,M,ecc,n,eta):

    #xi = ((G*m*n/(c**3))**(2.0/3)); OTS = ((1 - (ecc**2))**(0.5))

    k_1PN = 3*((G*m*n/(c**3))**(2.0/3))*(((1 - (ecc**2))**(0.5))**-2.0)

    k_2PN = (0.25)*(((1 - (ecc**2))**(0.5))**-4.0)*(((G*m*n/(c**3))**(2.0/3))**2.0)*((51-26*eta)*(ecc**2) - 28*eta + 78.0)

    k_3PN = (1.0/128)*(((1 - (ecc**2))**(0.5))**-6.0)*(((G*m*n/(c**3))**(2.0/3))**3.0)*(((-1536*eta + 3840)*(ecc**2) + 1920 - 768*eta)*((1 - (ecc**2))**(0.5)) +\
        (1040*(eta**2) + 2496 - 1760*eta)*(ecc**4.0) + (5120*(eta**2) + 28128 - 27840*eta + 123*((np.pi)**2)*eta)*(ecc**2) + 18240 - 25376*eta + 492*((np.pi)**2)*eta + 896*(eta**2))

    return k_1PN+k_2PN+k_3PN

# Variation of Frequency


def freq_var(m,M,ecc,n,eta):
    
    pre_factor_for_lead_1PN = (1.0/(M*T_sun*(((1 - (ecc**2))**(0.5))**7.0)))*(((G*m*n/(c**3))**(2.0/3))**4.0)*eta

    pre_factor_ecc_for_lead_1PN = -1.0*(((1 - (ecc**2))**(0.5))**-5.0)*eta*ecc*(((G*m*n/(c**3))**(2.0/3))**2.5)

    n_dot_lead = n*((37.0/5)*(ecc**4.0) + (292.0/5)*(ecc**2)+ (96.0/5))

    ecc_dot_lead = n*((121.0/15)*(ecc**2) + (304.0/15))

    n_dot_1PN = n*(-1.0/280)*(((1 - (ecc**2))**(0.5))**-2)*((G*m*n/(c**3))**(2.0/3))*\
        (8288*(ecc**6)*eta - 11717*(ecc**6) + 141708*(ecc**4)*eta - 197022*(ecc**4)
        - 219880*(ecc**2) + 159600*(ecc**2)*eta + 14787*eta - 20368)

    n_dot_tail = n*(384.0/5)*(((G*m*n/(c**3))**(2.0/3))**5.5)*eta*(np.pi)*((M*T_sun)**-1)*\
        (1.0 + 7.260831042*(ecc**2) + 5.844370473*(ecc**4)+ 0.8452020270*(ecc**6)
        + 0.07580633432*(ecc**8) + 0.002034045037*(ecc**10))/(1.0 - 4.900627291*(ecc**2)
        + 9.512155497*(ecc**4) - 9.051368575*(ecc**6) + 4.096465525*(ecc**8)-
        0.5933309609*(ecc**10) - 0.05427399445*(ecc**12)- 0.009020225634*(ecc**14))

    dn_dot_dn = eta*((G*m*n/c**3)**(8.0/3))*(407*ecc**4 + 3212*ecc**2 + 1056)/(15*M*T_sun*((1 - ecc**2)**(7/2.0)))

    dn_dot_de = ecc*eta*n*((G*m*n/c**3)**(8.0/3))*(111*ecc**4 + 1608*ecc**2 + 1256)/(5*M*T_sun*((1 - ecc**2)**(9.0/2)))

    nddot = dn_dot_dn*(pre_factor_for_lead_1PN*n_dot_lead) + dn_dot_de*(pre_factor_ecc_for_lead_1PN*ecc_dot_lead)

    nddot_check = (G**4)*(eta**2)*(m**4)*(n**6)*((G*m*n/c**3)**(1.0/3))*\
    	(-11*G*m*((37*ecc**4 + 292*ecc**2 + 96)**2) + M*T_sun*(c**3)*(ecc**2)*(13431*ecc**6 + 228312*ecc**4 + 640808*ecc**2 + 
    		381824))/(75*(M**2)*(T_sun**2)*(c**15)*((ecc**2 - 1)**7.0))

    if abs(nddot- nddot_check)>1e-15:
    	print('Expressions do not match')
    	raise SystemExit

    pb_dot_lead = n_dot_to_pb_dot(n, pre_factor_for_lead_1PN*n_dot_lead)
    pb_dot_lead_1PN = n_dot_to_pb_dot(n, pre_factor_for_lead_1PN*(n_dot_lead+n_dot_1PN))
    pb_dot_lead_1PN_tail = n_dot_to_pb_dot(n, (pre_factor_for_lead_1PN*(n_dot_lead+n_dot_1PN)+n_dot_tail))
    pb_dot_1PN = pb_dot_lead_1PN - pb_dot_lead
    pb_dot_tail = pb_dot_lead_1PN_tail - pb_dot_lead_1PN

    #return pb_dot_lead, pb_dot_1PN, pb_dot_tail
    return pre_factor_for_lead_1PN*n_dot_lead, pre_factor_for_lead_1PN*n_dot_1PN, n_dot_tail, nddot


# Variation of Eccentricity


def ecc_var(m,M,ecc,n,eta):
    
    pre_factor_ecc_for_lead_1PN = -1.0*(((1 - (ecc**2))**(0.5))**-5.0)*eta*ecc*(((G*m*n/(c**3))**(2.0/3))**2.5)

    # Remember !!!!! pre_factor_for_lead_1PN has to be multiplied to lead and 1PN

    ecc_dot_lead = n*((121.0/15)*(ecc**2) + (304.0/15))

    ecc_dot_1PN = n*(-1.0/2520)*(((1 - (ecc**2))**(0.5))**-2)*((G*m*n/(c**3))**(2.0/3))*(
    	93184*(ecc**4)*eta - 125361*(ecc**4) + 651252*(ecc**2)*eta - 880632*(ecc**2) 
    	+ 228704*eta - 340968)

    ecc_dot_tail = n*(-1.0/ecc)*(25.6*eta*(np.pi)*(((G*m*n/(c**3))**(2.0/3))**4.0)*((1 - (ecc**2))**(0.5)))*\
    	(((((1 - (ecc**2))**(0.5)))*(1.0 + 7.260831042*(ecc**2) + 5.844370473*(ecc**4) + 
    		0.8452020270*(ecc**6) + 0.07580633432*(ecc**8) + 0.002034045037*(ecc**10))/
    		(1.0 - 4.900627291*(ecc**2) + 9.512155497*(ecc**4) - 9.051368575*(ecc**6) + 
    			4.096465525*(ecc**8) - 0.5933309609*(ecc**10) - 0.05427399445*(ecc**12) - 0.009020225634*(ecc**14))) - 
    		((1.0 + 1.893242666*(ecc**2) - 2.708117333*(ecc**4) + 0.6192474531*(ecc**6) + 
    			0.05008474620*(ecc**8) - 0.01059040781*(ecc**10))/(1.0 - 4.638007334*(ecc**2) 
    			+ 8.716680569*(ecc**4) - 8.451197591*(ecc**6) + 4.435922348*(ecc**8)
    			- 1.199023304*(ecc**10) + 0.1398678608*(ecc**12) - 0.004254544193*(ecc**14))))

    return pre_factor_ecc_for_lead_1PN*ecc_dot_lead, pre_factor_ecc_for_lead_1PN*ecc_dot_1PN, ecc_dot_tail


def check():

    n = 2*(np.pi)/(7.751938773864*(u.hour.to(u.second)))
    ecc = 0.6171334
    eta = 0.24984192988081588
    M = 2.8283787
    m = 2.8283787*M_sun
    
    k = adv_per(m,M,ecc,n,eta)
    pb_dot_lead, pb_dot_1PN, pb_dot_tail = freq_var(m,M,ecc,n,eta)
    ecc_dot_lead, ecc_dot_1PN, ecc_dot_tail = ecc_var(m,M,ecc,n,eta)

    print(pb_dot_lead, pb_dot_1PN, pb_dot_lead+pb_dot_1PN, pb_dot_tail, pb_dot_lead+pb_dot_1PN+pb_dot_tail)
    print(n*k*u.rad.to(u.degree)*(u.year.to(u.second)))
    print(ecc_dot_lead, ecc_dot_1PN, ecc_dot_lead+ecc_dot_1PN, ecc_dot_tail, ecc_dot_lead+ecc_dot_1PN+ecc_dot_tail)


def lead_1PN_tail(y,t):

    m, M, eta = inputs(args)

    n, ecc, phase = y

    pre_factor_for_lead_1PN = (1.0/(M*T_sun*(((1 - (ecc**2))**(0.5))**7.0)))*(((G*m*n/(c**3))**(2.0/3))**4.0)*eta

    n_dot_lead = n*((37.0/5)*(ecc**4.0) + (292.0/5)*(ecc**2)+ (96.0/5))

    n_dot_1PN = n*(-1.0/280)*(((1 - (ecc**2))**(0.5))**-2)*((G*m*n/(c**3))**(2.0/3))*\
        (8288*(ecc**6)*eta - 11717*(ecc**6) + 141708*(ecc**4)*eta - 197022*(ecc**4)
        - 219880*(ecc**2) + 159600*(ecc**2)*eta + 14787*eta - 20368)

    n_dot_tail = n*(384.0/5)*(((G*m*n/(c**3))**(2.0/3))**5.5)*eta*(np.pi)*((M*T_sun)**-1)*\
        (1.0 + 7.260831042*(ecc**2) + 5.844370473*(ecc**4)+ 0.8452020270*(ecc**6)
        + 0.07580633432*(ecc**8) + 0.002034045037*(ecc**10))/(1.0 - 4.900627291*(ecc**2)
        + 9.512155497*(ecc**4) - 9.051368575*(ecc**6) + 4.096465525*(ecc**8)-
        0.5933309609*(ecc**10) - 0.05427399445*(ecc**12)- 0.009020225634*(ecc**14))

    n_dot = pre_factor_for_lead_1PN*(n_dot_lead+n_dot_1PN) + n_dot_tail

    pre_factor_ecc_for_lead_1PN = -1.0*(((1 - (ecc**2))**(0.5))**-5.0)*eta*ecc*(((G*m*n/(c**3))**(2.0/3))**2.5)

    ecc_dot_lead = n*((121.0/15)*(ecc**2) + (304.0/15))

    ecc_dot_1PN = n*(-1.0/2520)*(((1 - (ecc**2))**(0.5))**-2)*((G*m*n/(c**3))**(2.0/3))*(
        93184*(ecc**4)*eta - 125361*(ecc**4) + 651252*(ecc**2)*eta - 880632*(ecc**2)
        + 228704*eta - 340968)

    ecc_dot_tail = n*(-1.0/ecc)*(25.6*eta*(np.pi)*(((G*m*n/(c**3))**(2.0/3))**4)*((1 - (ecc**2))**(0.5)))*\
        (((((1 - (ecc**2))**(0.5)))*(1.0 + 7.260831042*(ecc**2) + 5.844370473*(ecc**4) +
        0.8452020270*(ecc**6) + 0.07580633432*(ecc**8) + 0.002034045037*(ecc**10))/
        (1.0 - 4.900627291*(ecc**2) + 9.512155497*(ecc**4) - 9.051368575*(ecc**6) +
        4.096465525*(ecc**8) - 0.5933309609*(ecc**10) - 0.05427399445*(ecc**12) - 0.009020225634*(ecc**14))) -
        ((1.0 + 1.893242666*(ecc**2) - 2.708117333*(ecc**4) + 0.6192472531*(ecc**6) +
        0.05008474620*(ecc**8) - 0.01059040781*(ecc**10))/(1.0 - 4.638007334*(ecc**2) 
        + 8.716680569*(ecc**4) - 8.451197591*(ecc**6) + 4.435922348*(ecc**8) 
        - 1.199023304*(ecc**10) + 0.1398678608*(ecc**12) - 0.004254544193*(ecc**14))))

    ecc_dot = pre_factor_ecc_for_lead_1PN*(ecc_dot_lead+ecc_dot_1PN) + ecc_dot_tail

    k_1PN = 3*((G*m*n/(c**3))**(2.0/3))*(((1 - (ecc**2))**(0.5))**-2)

    k_2PN = (0.25)*(((1 - (ecc**2))**(0.5))**-4)*(((G*m*n/(c**3))**(2.0/3))**2)*\
        ((51-26*eta)*(ecc**2) - 28*eta + 78.0)

    k_3PN = (1.0/128)*(((1 - (ecc**2))**(0.5))**-6.0)*(((G*m*n/(c**3))**
                                                        (2.0/3))**3.0)*\
        (((-1536*eta + 3840)*(ecc**2) + 1920 - 768*eta)*((1 - (ecc**2))**(0.5)) +\
            (1040*(eta**2) + 2496 - 1760*eta)*(ecc**4) + (5120*(eta**2) + 28128 -
            27840*eta + 123*((np.pi)**2)*eta)*(ecc**2) + 18240 - 25376*eta +
            492*((np.pi)**2)*eta + 896*(eta**2))

    k = k_1PN+k_2PN+k_3PN

    dydt = [n_dot, ecc_dot, n*(1+ k)]

    return dydt


def lead_1PN(y,t):

    m, M, eta = inputs(args)

    n, ecc, phase = y

    pre_factor_for_lead_1PN = (1.0/(M*T_sun*(((1 - (ecc**2))**(0.5))**7.0)))*(((G*m*n/(c**3))**(2.0/3))**4.0)*eta

    n_dot_lead = n*((37.0/5)*(ecc**4.0) + (292.0/5)*(ecc**2)+ (96.0/5))

    n_dot_1PN = n*(-1.0/280)*(((1 - (ecc**2))**(0.5))**-2)*((G*m*n/(c**3))**(2.0/3))*\
        (8288*(ecc**6)*eta - 11717*(ecc**6) + 141708*(ecc**4)*eta - 197022*(ecc**4)
        - 219880*(ecc**2) + 159600*(ecc**2)*eta + 14784*eta - 20368)

    n_dot = pre_factor_for_lead_1PN*(n_dot_lead+n_dot_1PN)

    pre_factor_ecc_for_lead_1PN = -1.0*(((1 - (ecc**2))**(0.5))**-5.0)*eta*ecc*(((G*m*n/(c**3))**(2.0/3))**2.5)

    ecc_dot_lead = n*((121.0/15)*(ecc**2) + (304.0/15))

    ecc_dot_1PN = n*(-1.0/2520)*(((1 - (ecc**2))**(0.5))**-2)*((G*m*n/(c**3))**(2.0/3))*(
        93184*(ecc**4)*eta - 125361*(ecc**4) + 651252*(ecc**2)*eta - 880632*(ecc**2)
        + 228704*eta - 340968)

    ecc_dot = pre_factor_ecc_for_lead_1PN*(ecc_dot_lead+ecc_dot_1PN)

    k_1PN = 3*((G*m*n/(c**3))**(2.0/3))*(((1 - (ecc**2))**(0.5))**-2)

    k_2PN = (0.25)*(((1 - (ecc**2))**(0.5))**-4)*(((G*m*n/(c**3))**(2.0/3))**2)*\
        ((51-26*eta)*(ecc**2) - 28*eta + 78.0)

    k_3PN = (1.0/128)*(((1 - (ecc**2))**(0.5))**-6.0)*(((G*m*n/(c**3))**
                                                        (2.0/3))**3.0)*\
        (((-1536*eta + 3840)*(ecc**2) + 1920 - 768*eta)*((1 - (ecc**2))**(0.5)) +\
            (1040*(eta**2) + 2496 - 1760*eta)*(ecc**4) + (5120*(eta**2) + 28128 -
            27840*eta + 123*((np.pi)**2)*eta)*(ecc**2) + 18240 - 25376*eta +
            492*((np.pi)**2)*eta + 896*(eta**2))

    k = k_1PN+k_2PN+k_3PN

    dydt = [n_dot, ecc_dot, n*(1+ k)]

    return dydt


def lead(y,t):

    m, M, eta = inputs(args)

    n, ecc, phase = y

    pre_factor_for_lead_1PN = (1.0/(M*T_sun*(((1 - (ecc**2))**(0.5))**7.0)))*(((G*m*n/(c**3))**(2.0/3))**4.0)*eta

    n_dot_lead = n*((37.0/5)*(ecc**4.0) + (292.0/5)*(ecc**2)+ (96.0/5))

    n_dot = pre_factor_for_lead_1PN*(n_dot_lead)

    pre_factor_ecc_for_lead_1PN = -1.0*(((1 - (ecc**2))**(0.5))**-5.0)*eta*ecc*(((G*m*n/(c**3))**(2.0/3))**2.5)

    ecc_dot_lead = n*((121.0/15)*(ecc**2) + (304.0/15))

    ecc_dot = pre_factor_ecc_for_lead_1PN*(ecc_dot_lead)

    k_1PN = 3*((G*m*n/(c**3))**(2.0/3))*(((1 - (ecc**2))**(0.5))**-2)

    k_2PN = (0.25)*(((1 - (ecc**2))**(0.5))**-4.0)*(((G*m*n/(c**3))**(2.0/3))**2)*\
    	((51-26*eta)*(ecc**2) - 28*eta + 78.0)

    k_3PN = (1.0/128)*(((1 - (ecc**2))**(0.5))**-6.0)*(((G*m*n/(c**3))**(2.0/3))**3.0)*\
        (((-1536*eta + 3840)*(ecc**2) + 1920 - 768*eta)*((1 - (ecc**2))**(0.5)) +\
            (1040*(eta**2) + 2496 - 1760*eta)*(ecc**4) + (5120*(eta**2) + 28128 -
            27840*eta + 123*((np.pi)**2)*eta)*(ecc**2) + 18240 - 25376*eta +
            492*((np.pi)**2)*eta + 896*(eta**2))

    k = k_1PN + k_2PN + k_3PN

    dydt = [n_dot, ecc_dot, n*(1+ k)]

    return dydt


def inputs(args):
    M = args.m1 + args.m2
    m = M*M_sun
    eta = (args.m1)*(args.m2)/(M**2)

    return m, M, eta


def odeint_solver(y0,t):

    lead_sol = odeint(lead, y0, t)
    lead_1PN_sol = odeint(lead_1PN, y0, t)
    lead_1PN_tail_sol = odeint(lead_1PN_tail, y0, t)

    return lead_sol, lead_1PN_sol, lead_1PN_tail_sol


def sol_spilt(sol):
    freq = sol[:,0]
    ecc = sol[:,1]
    phase = sol[:,2]

    return freq, ecc, phase


def const_P(pb,ecc,m,m1):
    p = pb*u.hour.to(u.second)
    n = 2*np.pi/p
    m2 = m-m1
    eta = m1*m2/(m**2)
    T_sun = 4.9254909476412675e-06
    chirp_mass = (eta**(3.0/5))*m*T_sun
    A = 0.2*(chirp_mass)**(5.0/3)
    res = ((n**(8.0/3))*(ecc**(48.0/19))*((121*(ecc**2) + 304.0)**(3480.0/2299)))/(3*((1 - ecc**2)**4))
    P = A*res
    return chirp_mass,P


def appelf1(a,b,c,d,x,y):
    return mpmath.hyper2d({'m+n':[a], 'm':[b], 'n':[c]}, {'m+n':[d]}, x, y)


def tau_fn(ecc):
    return (ecc**(48.0/19))*(appelf1(24.0/19, -1181.0/2299, 3.0/2, 43.0/19, -121.0/304*(ecc**2), ecc**2))/768.0


def orbits(t):

    pb = np.linspace(0.5, 24, 50)
    ecc = np.linspace(0.05,0.95,50)
    n = 2*np.pi/(pb*u.hour.to(u.second))

    N, ECC = np.meshgrid(n,ecc)

    orbits_lead = np.zeros((len(N), len(N[0])))
    orbits_lead_1PN = np.zeros((len(N), len(N[0])))
    orbits_lead_1PN_tail = np.zeros((len(N), len(N[0])))
    orbits_lead_taylor_res = np.zeros((len(N), len(N[0])))
    orbits_lead_taylor_res1 = np.zeros((len(N), len(N[0])))
    orbits_lead_taylor_ratio1 = np.zeros((len(N), len(N[0])))
    orbits_lead_taylor_ratio2 = np.zeros((len(N), len(N[0])))


    for i in range(len(N)):
        for j in range(len(ECC[0])):

            y0 = [N[i][j], ECC[i][j], 0.0]

            m, M, eta = inputs(args)

            chirp_mass, const_merger_time = const_P(2*np.pi*u.second.to(u.hour)/y0[0], y0[1], M, 1.4)

            merger_time_yrs = tau_fn(y0[1])*u.second.to(u.year)/const_merger_time

            print(merger_time_yrs)

            
            lead_sol, lead_1PN_sol, lead_1PN_tail_sol = odeint_solver(y0,t)

            lead_freq, lead_ecc, lead_phase = sol_spilt(lead_sol)
            lead_1PN_freq, lead_1PN_ecc, lead_1PN_phase = sol_spilt(lead_1PN_sol)
            lead_1PN_tail_freq, lead_1PN_tail_ecc, lead_1PN_tail_phase = sol_spilt(lead_1PN_tail_sol)

            m, M, eta = inputs(args)
            n_dot_lead, n_dot_1PN, n_dot_tail, nddot = freq_var(m,M,ECC[i][j],N[i][j],eta)
            
            taylor_phase = N[i][j]*t[-1] + 0.5*n_dot_lead*((t[-1])**2)
            taylor_phase_ext = N[i][j]*t[-1] + 0.5*n_dot_lead*((t[-1])**2) + (1.0/6)*nddot*((t[-1])**3)

            print(N[i][j]*t[-1] , 0.5*n_dot_lead*((t[-1])**2), (1.0/6)*nddot*((t[-1])**3))
            print(lead_phase[-1] - taylor_phase, lead_phase[-1] - taylor_phase_ext)
            

            orb_lead = lead_phase[-1]/(2*np.pi)
            orb_lead_1PN = lead_1PN_phase[-1]/(2*np.pi)
            orb_lead_1PN_tail = lead_1PN_tail_phase[-1]/(2*np.pi)
            orb_res = orb_lead - (taylor_phase/(2*np.pi))
            

            orbits_lead[i][j] = orb_lead
            orbits_lead_1PN[i][j] = orb_lead_1PN
            orbits_lead_1PN_tail[i][j] = orb_lead_1PN_tail
            orbits_lead_taylor_res[i][j] = orb_res
            orbits_lead_taylor_res1[i][j] = (taylor_phase_ext- taylor_phase)/(2*np.pi)
            orbits_lead_taylor_ratio1[i][j] = (N[i][j]*t[-1])/(0.5*n_dot_lead*((t[-1])**2))
            orbits_lead_taylor_ratio2[i][j] = (0.5*n_dot_lead*((t[-1])**2))/((1.0/6)*nddot*((t[-1])**3))
            #print(2*np.pi*u.second.to(u.hour)/N[i][j], ECC[i][j], orb_lead_1PN-orb_lead)
            print('Simulating the phase accumulated for the binary with period {p} and eccentricity {e}'.format(p=(2*u.second.to(u.hour)*np.pi/(N[i][j])), e = ECC[i][j]))

    return orbits_lead, orbits_lead_1PN, orbits_lead_1PN_tail, orbits_lead_taylor_res, orbits_lead_taylor_res1, orbits_lead_taylor_ratio1, orbits_lead_taylor_ratio2


def contour_plot(M,orbits_lead, orbits_lead_1PN, orbits_lead_1PN_tail, orbits_lead_taylor_res, orbits_lead_taylor_res1, orbits_lead_taylor_ratio1, orbits_lead_taylor_ratio2):

    pb = np.linspace(0.5, 24, 50)
    ecc = np.linspace(0.05,0.95,50)
    n = 2*np.pi/(pb*u.hour.to(u.second))

    N, ECC = np.meshgrid(n,ecc)

    if not os.path.isdir('Plots/Mass_{m:4.2f}'.format(m=M)):
        os.mkdir('Plots/Mass_{m:4.2f}'.format(m=M))


    plotfile = PdfPages("./Plots/Mass_{m:4.2f}/1PN_alone_{m:4.2f}.pdf".format(m=M))
    levels= [1e-8, 1e-6, 1e-3, 1, 10, 1e2]
    plt.clf()
    plt.suptitle("For a binary system of massses 1.4 and {m:4.2f}".format(m=M-1.4))
    #plt.contour(2*np.pi*u.second.to(u.hour)/N, ECC, 2*np.pi*abs(orbits_lead_1PN-orbits_lead), norm=LogNorm(), origin='lower', colors='red', linestyles='dashed')
    plt.contour(2*np.pi*u.second.to(u.hour)/N, ECC, 2*np.pi*abs(orbits_lead_1PN-orbits_lead), levels, norm=LogNorm(), origin='lower', colors='red', linewidths=0.5)
    plt.imshow(2*np.pi*abs(orbits_lead_1PN-orbits_lead),extent=[0.5, 24.0, 0.05, 0.95], norm=LogNorm(), aspect='auto', interpolation='gaussian', origin='lower')
    plt.xlabel('Period of the binary in hours')
    plt.ylabel('Eccentricity of the orbit')
    plt.title('Orbits contributed by 1PN alone')
    plt.colorbar()
    plotfile.savefig()
    plotfile.close()
    
    plotfile = PdfPages("./Plots/Mass_{m:4.2f}/tail_alone_{m:4.2f}.pdf".format(m=M))
    plt.clf()
    plt.suptitle("For a binary system of massses 1.4 and {m:4.2f}".format(m=M-1.4))
    plt.contour(2*np.pi*u.second.to(u.hour)/N, ECC, 2*np.pi*abs(orbits_lead_1PN_tail-orbits_lead_1PN), levels, norm=LogNorm(), origin='lower', colors='red', linewidths=0.5)
    plt.imshow(abs(orbits_lead_1PN_tail-orbits_lead_1PN),extent=[0.5, 24.0, 0.05, 0.95], norm=LogNorm(), interpolation='gaussian', aspect='auto', origin='lower')
    plt.xlabel('Period of the binary in hours')
    plt.ylabel('Eccentricity of the orbit')
    plt.title('Orbits contributed by tail alone')
    plt.colorbar()
    plotfile.savefig()
    plotfile.close()

    plotfile = PdfPages("./Plots/Mass_{m:4.2f}/taylor_residual_{m:4.2f}.pdf".format(m=M))
    plt.clf()
    levels= [1e0, 1e1, 1e2]
    plt.contour(2*np.pi*u.second.to(u.hour)/N, ECC, 2*np.pi*orbits_lead_taylor_res, levels, norm=LogNorm(), origin='lower', colors='red', linewidths=0.5)
    plt.imshow(2*np.pi*orbits_lead_taylor_res,extent=[0.5, 24.0, 0.05, 0.95], norm=LogNorm(), aspect='auto', interpolation='gaussian', origin='lower')
    plt.xlabel('Period of the binary in hours')
    plt.ylabel('Eccentricity of the orbit')
    plt.title('Residuals due to Taylor approximation')
    plt.colorbar()
    plotfile.savefig()
    plotfile.close()

    plotfile = PdfPages("./Plots/Mass_{m:4.2f}/taylor_residual_ddot_{m:4.2f}.pdf".format(m=M))
    plt.clf()
    levels= [1e-2, 1e-1, 1e0, 1e1]
    plt.contour(2*np.pi*u.second.to(u.hour)/N, ECC, 2*np.pi*orbits_lead_taylor_res1, levels, norm=LogNorm(), origin='lower', colors='red', linewidths=0.5)
    plt.imshow(2*np.pi*orbits_lead_taylor_res1,extent=[0.5, 24.0, 0.05, 0.95], norm=LogNorm(), aspect='auto', interpolation='gaussian', origin='lower')
    plt.xlabel('Period of the binary in hours')
    plt.ylabel('Eccentricity of the orbit')
    plt.title('Residuals due to Taylor approximation')
    plt.colorbar()
    plotfile.savefig()
    plotfile.close()

    plotfile = PdfPages("./Plots/Mass_{m:4.2f}/taylor_residual_ratio_{m:4.2f}.pdf".format(m=M))
    plt.clf()
    levels= [1e0, 1e1, 1e2, 1e3]
    plt.contour(2*np.pi*u.second.to(u.hour)/N, ECC, orbits_lead_taylor_ratio1, levels, norm=LogNorm(), origin='lower', colors='red', linewidths=0.5)
    plt.imshow(orbits_lead_taylor_ratio1,extent=[0.5, 24.0, 0.05, 0.95], norm=LogNorm(), aspect='auto', interpolation='gaussian', origin='lower')
    plt.xlabel('Period of the binary in hours')
    plt.ylabel('Eccentricity of the orbit')
    plt.title('Residuals due to Taylor approximation')
    plt.colorbar()
    plotfile.savefig()
    plotfile.close()

    plotfile = PdfPages("./Plots/Mass_{m:4.2f}/taylor_residual_ratio2_{m:4.2f}.pdf".format(m=M))
    plt.clf()
    levels= [1e0, 1e1, 1e2, 1e3]
    plt.contour(2*np.pi*u.second.to(u.hour)/N, ECC, orbits_lead_taylor_ratio2, levels, norm=LogNorm(), origin='lower', colors='red', linewidths=0.5)
    plt.imshow(orbits_lead_taylor_ratio2,extent=[0.5, 24.0, 0.05, 0.95], norm=LogNorm(), aspect='auto', interpolation='gaussian', origin='lower')
    plt.xlabel('Period of the binary in hours')
    plt.ylabel('Eccentricity of the orbit')
    plt.title('Residuals due to Taylor approximation')
    plt.colorbar()
    plotfile.savefig()
    plotfile.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Code used to solve coupled ODEs to solve for phase')
    parser.add_argument('m2', help='Companion mass', type=float)
    parser.add_argument('--m1', help='Pulsar mass, default=1.4',type=float, default=1.4)
    parser.add_argument('--pb', help='Binary Period in hrs',type=float)
    parser.add_argument('--ecc', help='Eccentricity of the orbit',type=float)
    parser.add_argument('--t', help='Span of interest, default= 20yrs', default=20,type=float)

    args = parser.parse_args()

    astronomical_constants()

    t = np.linspace(0, (args.t)*u.year.to(u.second), u.year.to(u.day)*args.t)

    orbits_lead, orbits_lead_1PN, orbits_lead_1PN_tail, orbits_lead_taylor_res, orbits_lead_taylor_res1, orbits_lead_taylor_ratio1, orbits_lead_taylor_ratio2 = orbits(t)

    contour_plot((args.m1+args.m2),orbits_lead, orbits_lead_1PN, orbits_lead_1PN_tail, orbits_lead_taylor_res, orbits_lead_taylor_res1, orbits_lead_taylor_ratio1, orbits_lead_taylor_ratio2)