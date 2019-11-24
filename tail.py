#!/usr/bin/env python3.5

import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table as table
from astropy import units as u
from astropy import constants
from scipy.integrate import odeint
import argparse
from astropy.io import ascii
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import matplotlib.colors as colors


# Advance of Periastron per orbit


# k = ( (3*(((G*m*n/c**3)**(2.0/3))**(2/3))/((1 - et**2)**0.5)**2) + ((1.0/4)*((51-26*eta)*(et**2) - 28*eta + 78)*(((G*m*n/c**3)**(2.0/3))**(4.0/3))/((1 - et**2)**0.5)**4) + ((1.0/128)*(((G*m*n/c**3)**(2.0/3))**2)*(((-1536*eta + 3840)*(et**2) + 1920 - 768*eta)*((1 - et**2)**0.5) + (1040*(eta**2) + 2496-1760*eta)*(et**4) + (5120*(eta**2) + 28128 - 27840*eta + 123*(np.pi**2)*eta)*(et**2) +18240-25376*eta + 492*(np.pi**2)*eta + 896*(eta**2))/((1 - et**2)**0.5)**6) )
# Frequncy variation (dn_dl) :

# Leading Order (0 PN) = (1/(M*T_sun*((1 - et**2)**0.5)**7))*(((G*m*n/c**3)**(2.0/3))**4)*eta*((37.0/5)*(et**4) + (292.0/5)*(et**2) + 96.0/5)

# Post-Newtonian (1 PN) = (1/(M*T_sun*((1 - et**2)**0.5)**7))*((1/((1 - et**2)**0.5)**2)*(-1.0/280)*((G*m*n/c**3)**(2.0/3))*(8288*(et**6)*eta - 11717*(et**6) + 141708*(et**4)*eta - 197022*(et**4)-219880*(et**2) + 159600*(et**2)*eta + 14784*eta - 20368))

# Tail Term (1.5 PN)  = (((384.0/5)*(((G*m*n/c**3)**(2.0/3))**(11.0/2))*eta*pi*(1.0 + 7.260831042*(et**2) + 5.844370473*(et**4) + 0.8452020270*(et**6) + 0.7580633432e-1*(et**8) + 0.2034045037e-2*(et**10)))/(T_sun*M*(1.0 - 4.900627291*(et**2) + 9.512155497*(et**4) - 9.051368575*(et**6) + 4.096465525*(et**8) - 0.5933309609*(et**10) - 0.5427399445e-1*(et**12) - 0.9020225634e-2*(et**14))))

# Eccentricity Variation (de_dl) : 

# Leading Order (0 PN) = ((1/((1 - et**2)**0.5)**5)*-1*eta*((G*m*n/c**3)**(2.0/3))**(5.0/2)*et*((121.0/15)*et**2 + 304.0/15))

# Post-Newtonian (1 PN) = ((1/((1 - et**2)**0.5)**5)*(1/((1 - et**2)**0.5)**2)*(1.0/2520)*((G*m*n/c**3)**(2.0/3))*(93184*(et**4)*eta - 125361*(et**4) + 651252*(et**2)*eta - 880632*(et**2) + 228704*eta - 340968))

# Tail Term (1.5 PN) = ((-1/et)*25.6*eta*(np.pi)*(((G*m*n/c**3)**(2.0/3))**4)*((1 - et**2)**0.5)*((((1 - et**2)**0.5)*(1.0 + 7.260831042*(et**2) + 5.844370473*(et**4) + 0.8452020270*(et**6) + 0.7580633432e-1*(et**8) + 0.2034045037e-2*(et**10))/(1.0 - 4.900627291*(et**2) + 9.512155497*(et**4) - 9.051368575*(et**6) + 4.096465525*(et**8) - 0.5933309609*(et**10) - 0.5427399445e-1*(et**12) -0.9020225634e-2*(et**14)))    -    (1.0*(1.0 + 1.893242666*(et**2) - 2.708117333*(et**4) + 0.6192474531*(et**6) + 0.5008474620e-1*(et**8) - 0.1059040781e-1*(et**10))/(1.0 - 4.638007334*(et**2) + 8.716680569*(et**4) - 8.451197591*(et**6) + 4.435922348*(et**8) -1.199023304*(et**10) + 0.1398678608*(et**12) - 0.4254544193e-2*(et**14)))))


def astronomical_constants():
    c = constants.c.value
    G = constants.G.value
    M_sun = constants.M_sun.value
    T_sun = constants.GM_sun.value/((constants.c.value)**3)
    return c, G, M_sun, T_sun


def parameters(args):

    mass = args.Mass
    eta = mass*1.4/((mass+1.4)**2)
    return eta, mass


def initial_conditions(args):
    ecc_0 = args.e

    if args.P_units == 'hrs':
        P_0 = (args.P)*u.hour.to(u.second)
    elif args.P_units == 'min':
        P_0 = (args.P)*u.minute.to(u.second)
    elif args.P_units == 'days':
        P_0 = (args.P)*u.day.to(u.second)
    elif args.P_units == 'yrs':
        P_0 = (args.P)*u.year.to(u.second)
    else:
        P_0 = args.P

    n_0 = 2*np.pi/P_0
    y0 = [ecc_0, n_0, 0.0]
    return ecc_0, n_0, y0


def lead(y, t):
    c, G, M_sun, T_sun = astronomical_constants()
    eta, mass = parameters(args)
    #print(eta,mass)
    M = mass
    m = mass*M_sun    
    et, n, phase = y
    dydt = [ (n*(1/(((1 - et**2)**0.5)**5))*-1*eta*((G*m*n/c**3)**(2.0/3))**(5.0/2)*et*((121.0/15)*et**2 + 304.0/15))  ,   (n*(1/(M*T_sun*((1 - et**2)**0.5)**7))*(((G*m*n/c**3)**(2.0/3))**4)*eta*((37.0/5)*(et**4) + (292.0/5)*(et**2) + 96.0/5))  ,   n*(1 + ((3*(((G*m*n/c**3)**(2.0/3)))/(((1 - et**2)**0.5)**2)))) ]
    return dydt


def lead_1PN(y,t):
    c, G, M_sun, T_sun = astronomical_constants()
    eta, mass = parameters(args)
    M = mass
    m = mass*M_sun
    et, n, phase = y 
    dydt = [((1/((1 - et**2)**0.5)**5)*-1*n*eta*((G*m*n/c**3)**(2.0/3))**(5.0/2)*et*(((121.0/15)*et**2 + 304.0/15) - ((1/((1 - et**2)**0.5)**5)*(1/((1 - et**2)**0.5)**2)*(1.0/2520)*((G*m*n/c**3)**(2.0/3))*(93184*(et**4)*eta - 125361*(et**4) + 651252*(et**2)*eta - 880632*(et**2) + 228704*eta - 340968)))) ,  (1/(M*T_sun*((1 - et**2)**0.5)**7))*n*((G*m*n/c**3)**(2.0/3))**4*eta*(((37.0/5)*et**4 + (292.0/5)*et**2 + 96.0/5)  +   ((1/((1 - et**2)**0.5)**2)*(-1.0/280)*((G*m*n/c**3)**(2.0/3))*(8288*(et**6)*eta - 11717*(et**6) + 141708*(et**4)*eta - 197022*(et**4)-219880*(et**2) + 159600*(et**2)*eta + 14784*eta - 20368)))  ,   (n*(1 + ((3*(((G*m*n/c**3)**(2.0/3))**(2/3))/((1 - et**2)**0.5)**2)+((1.0/4)*((51-26*eta)*(et**2) - 28*eta + 78)*(((G*m*n/c**3)**(2.0/3))**(4.0/3))/((1 - et**2)**0.5)**4) + ((1.0/128)*(((G*m*n/c**3)**(2.0/3))**2)*(((-1536*eta + 3840)*(et**2) + 1920 - 768*eta)*((1 - et**2)**0.5) + (1040*(eta**2) + 2496-1760*eta)*(et**4) + (5120*(eta**2) + 28128 - 27840*eta + 123*(np.pi**2)*eta)*(et**2) +18240-25376*eta + 492*(np.pi**2)*eta + 896*(eta**2))/((1 - et**2)**0.5)**6))))]
    return dydt


def lead_1PN_tail(y,t):
    c, G, M_sun, T_sun = astronomical_constants()
    eta, mass = parameters(args)
    M = mass
    m = mass*M_sun
    et, n, phase = y 
    dydt = [((1/((1 - et**2)**0.5)**5)*-1*n*eta*((G*m*n/c**3)**(2.0/3))**(5.0/2)*et*(((121.0/15)*et**2 + 304.0/15) - ((1/((1 - et**2)**0.5)**5)*(1/((1 - et**2)**0.5)**2)*(1.0/2520)*((G*m*n/c**3)**(2.0/3))*(93184*(et**4)*eta - 125361*(et**4) + 651252*(et**2)*eta - 880632*(et**2) + 228704*eta - 340968)))) +  ((-1/et)*25.6*eta*(np.pi)*n*(((G*m*n/c**3)**(2.0/3))**4)*((1 - et**2)**0.5)*((((1 - et**2)**0.5)*(1.0 + 7.260831042*(et**2) + 5.844370473*(et**4) + 0.8452020270*(et**6) + 0.7580633432e-1*(et**8) + 0.2034045037e-2*(et**10))/(1.0 - 4.900627291*(et**2) + 9.512155497*(et**4) - 9.051368575*(et**6) + 4.096465525*(et**8) - 0.5933309609*(et**10) - 0.5427399445e-1*(et**12) - 0.9020225634e-2*(et**14))) - (1.0*(1.0 + 1.893242666*(et**2) - 2.708117333*(et**4) + 0.6192474531*(et**6) + 0.5008474620e-1*(et**8) - 0.1059040781e-1*(et**10))/(1.0 - 4.638007334*(et**2) + 8.716680569*(et**4) - 8.451197591*(et**6) + 4.435922348*(et**8) -1.199023304*(et**10) + 0.1398678608*(et**12) - 0.4254544193e-2*(et**14))))),  (1/(M*T_sun*((1 - et**2)**0.5)**7))*n*((G*m*n/c**3)**(2.0/3))**4*eta*(((37.0/5)*et**4 + (292.0/5)*et**2 + 96.0/5)  +   ((1/((1 - et**2)**0.5)**2)*(-1.0/280)*((G*m*n/c**3)**(2.0/3))*(8288*(et**6)*eta - 11717*(et**6) + 141708*(et**4)*eta - 197022*(et**4)-219880*(et**2) + 159600*(et**2)*eta + 14784*eta - 20368)))  +   (((384.0/5)*n*(((G*m*n/c**3)**(2.0/3))**(11.0/2))*eta*np.pi*(1.0 + 7.260831042*(et**2) + 5.844370473*(et**4) + 0.8452020270*(et**6) + 0.7580633432e-1*(et**8) + 0.2034045037e-2*(et**10)))/(T_sun*M*(1.0 - 4.900627291*(et**2) + 9.512155497*(et**4) - 9.051368575*(et**6) + 4.096465525*(et**8) - 0.5933309609*(et**10) - 0.5427399445e-1*(et**12) - 0.9020225634e-2*(et**14)))),   (n*(1 + ((3*(((G*m*n/c**3)**(2.0/3))**(2/3))/((1 - et**2)**0.5)**2)+((1.0/4)*((51-26*eta)*(et**2) - 28*eta + 78)*(((G*m*n/c**3)**(2.0/3))**(4.0/3))/((1 - et**2)**0.5)**4) + ((1.0/128)*(((G*m*n/c**3)**(2.0/3))**2)*(((-1536*eta + 3840)*(et**2) + 1920 - 768*eta)*((1 - et**2)**0.5) + (1040*(eta**2) + 2496-1760*eta)*(et**4) + (5120*(eta**2) + 28128 - 27840*eta + 123*(np.pi**2)*eta)*(et**2) +18240-25376*eta + 492*(np.pi**2)*eta + 896*(eta**2))/((1 - et**2)**0.5)**6))))]
    return dydt


'''
def n_dot_lead(freq,ecc):
    c, G, M_sun, T_sun = astronomical_constants()
    eta, mass = parameters(args)
    M = mass
    m = mass*M_sun    
    n=freq
    et=ecc
    dydt = [ (n*(1/((1 - et**2)**0.5)**5)*-1*eta*((G*m*n/c**3)**(2.0/3))**(5.0/2)*et*((121.0/15)*et**2 + 304.0/15))  ,   (n*(1/(M*T_sun*((1 - et**2)**0.5)**7))*(((G*m*n/c**3)**(2.0/3))**4)*eta*((37.0/5)*(et**4) + (292.0/5)*(et**2) + 96.0/5))  ,   n*(1 + ((3*(((G*m*n/c**3)**(2.0/3))**(2/3))/((1 - et**2)**0.5)**2) +  ((1.0/4)*((51-26*eta)*(et**2) - 28*eta + 78)*(((G*m*n/c**3)**(2.0/3))**(4.0/3))/((1 - et**2)**0.5)**4)   +   ((1.0/128)*(((G*m*n/c**3)**(2.0/3))**2)*(((-1536*eta + 3840)*(et**2) + 1920 - 768*eta)*((1 - et**2)**0.5) + (1040*(eta**2) + 2496-1760*eta)*(et**4) + (5120*(eta**2) + 28128 - 27840*eta + 123*(np.pi**2)*eta)*(et**2) +18240-25376*eta + 492*(np.pi**2)*eta + 896*(eta**2))/((1 - et**2)**0.5)**6))) ]
    return dydt[1]
'''
'''
def n_dot_lead_1PN_tail(freq_tail,ecc_tail):
    c, G, M_sun, T_sun = astronomical_constants()
    eta, mass = parameters(args)
    M = mass
    m = mass*M_sun
    n=freq_tail
    et=ecc_tail
    dydt = [((1/((1 - et**2)**0.5)**5)*-1*n*eta*((G*m*n/c**3)**(2.0/3))**(5.0/2)*et*(((121.0/15)*et**2 + 304.0/15) - ((1/((1 - et**2)**0.5)**5)*(1/((1 - et**2)**0.5)**2)*(1.0/2520)*((G*m*n/c**3)**(2.0/3))*(93184*(et**4)*eta - 125361*(et**4) + 651252*(et**2)*eta - 880632*(et**2) + 228704*eta - 340968)))) +  ((-1/et)*25.6*eta*(np.pi)*n*(((G*m*n/c**3)**(2.0/3))**4)*((1 - et**2)**0.5)*((((1 - et**2)**0.5)*(1.0 + 7.260831042*(et**2) + 5.844370473*(et**4) + 0.8452020270*(et**6) + 0.7580633432e-1*(et**8) + 0.2034045037e-2*(et**10))/(1.0 - 4.900627291*(et**2) + 9.512155497*(et**4) - 9.051368575*(et**6) + 4.096465525*(et**8) - 0.5933309609*(et**10) - 0.5427399445e-1*(et**12) - 0.9020225634e-2*(et**14))) - (1.0*(1.0 + 1.893242666*(et**2) - 2.708117333*(et**4) + 0.6192474531*(et**6) + 0.5008474620e-1*(et**8) - 0.1059040781e-1*(et**10))/(1.0 - 4.638007334*(et**2) + 8.716680569*(et**4) - 8.451197591*(et**6) + 4.435922348*(et**8) -1.199023304*(et**10) + 0.1398678608*(et**12) - 0.4254544193e-2*(et**14))))),  (1/(M*T_sun*((1 - et**2)**0.5)**7))*n*((G*m*n/c**3)**(2.0/3))**4*eta*(((37.0/5)*et**4 + (292.0/5)*et**2 + 96.0/5)  +   ((1/((1 - et**2)**0.5)**2)*(-1.0/280)*((G*m*n/c**3)**(2.0/3))*(8288*(et**6)*eta - 11717*(et**6) + 141708*(et**4)*eta - 197022*(et**4)-219880*(et**2) + 159600*(et**2)*eta + 14784*eta - 20368)))  +   (((384.0/5)*n*(((G*m*n/c**3)**(2.0/3))**(11.0/2))*eta*np.pi*(1.0 + 7.260831042*(et**2) + 5.844370473*(et**4) + 0.8452020270*(et**6) + 0.7580633432e-1*(et**8) + 0.2034045037e-2*(et**10)))/(T_sun*M*(1.0 - 4.900627291*(et**2) + 9.512155497*(et**4) - 9.051368575*(et**6) + 4.096465525*(et**8) - 0.5933309609*(et**10) - 0.5427399445e-1*(et**12) - 0.9020225634e-2*(et**14)))),   (n*(1 + ((3*(((G*m*n/c**3)**(2.0/3))**(2/3))/((1 - et**2)**0.5)**2)+((1.0/4)*((51-26*eta)*(et**2) - 28*eta + 78)*(((G*m*n/c**3)**(2.0/3))**(4.0/3))/((1 - et**2)**0.5)**4) + ((1.0/128)*(((G*m*n/c**3)**(2.0/3))**2)*(((-1536*eta + 3840)*(et**2) + 1920 - 768*eta)*((1 - et**2)**0.5) + (1040*(eta**2) + 2496-1760*eta)*(et**4) + (5120*(eta**2) + 28128 - 27840*eta + 123*(np.pi**2)*eta)*(et**2) +18240-25376*eta + 492*(np.pi**2)*eta + 896*(eta**2))/((1 - et**2)**0.5)**6))))]
    return dydt[1]
'''
'''
def n_dot_diff(freq,ecc):
    c, G, M_sun, T_sun = astronomical_constants()
    eta, mass = parameters(args)
    M = mass
    m = mass*M_sun    
    n = freq
    et = ecc
    n_dot_1PN_tail  = (1/(M*T_sun*((1 - et**2)**0.5)**7))*n*((G*m*n/c**3)**(2.0/3))**4*eta*( ((1/((1 - et**2)**0.5)**2)*(-1.0/280)*((G*m*n/c**3)**(2.0/3))*(8288*(et**6)*eta - 11717*(et**6) + 141708*(et**4)*eta - 197022*(et**4)-219880*(et**2) + 159600*(et**2)*eta + 14784*eta - 20368)))  +   (((384.0/5)*n*(((G*m*n/c**3)**(2.0/3))**(11.0/2))*eta*np.pi*(1.0 + 7.260831042*(et**2) + 5.844370473*(et**4) + 0.8452020270*(et**6) + 0.7580633432e-1*(et**8) + 0.2034045037e-2*(et**10)))/(T_sun*M*(1.0 - 4.900627291*(et**2) + 9.512155497*(et**4) - 9.051368575*(et**6) + 4.096465525*(et**8) - 0.5933309609*(et**10) - 0.5427399445e-1*(et**12) - 0.9020225634e-2*(et**14)))) 
    n_dot_lead=  n*(1/(M*T_sun*((1 - et**2)**0.5)**7))*(((G*m*n/c**3)**(2.0/3))**4)*eta*((37.0/5)*(et**4) + (292.0/5)*(et**2) + 96.0/5)
    return n_dot_1PN_tail
'''


def sim_sol(arr):
    ecc = arr[:,0]
    freq = arr[:,1]
    phase = arr[:,2]
    orbits = phase/(2*np.pi)
    return ecc, freq, phase, orbits


def plots(var,t, plotfile, label, leg):
    plt.plot(t, var, 'b')
    plt.xlabel('Time')
    plt.ylabel('{yname}'.format(yname=label))
    plt.title('{name}'.format(name=leg))
    plotfile.savefig()
    plt.clf()


def solver(e_ini, n_ini, t):
    ini = [e_ini,n_ini,0]
    solution_lead  = odeint(lead,ini,t)
    solution_lead_1PN  = odeint(lead_1PN, ini, t)
    solution_lead_1PN_tail = odeint(lead_1PN_tail, ini, t)
    return solution_lead, solution_lead_1PN, solution_lead_1PN_tail


def x_dot(m,P,ecc,mask=None):
    c, G, M_sun, T_sun = astronomical_constants()
    M = m+1.4
    eta = m*1.4/((m+1.4)**2)
    m = (m+1.4)*M_sun
    n = 2*np.pi/(P*u.hour.to(u.second))
    et=ecc
    dx_dt_lead_1PN_tail = [((1/((1 - et**2)**0.5)**5)*-1*n*eta*((G*m*n/c**3)**(2.0/3))**(5.0/2)*et*(((121.0/15)*et**2 + 304.0/15) - ((1/((1 - et**2)**0.5)**5)*(1/((1 - et**2)**0.5)**2)*(1.0/2520)*((G*m*n/c**3)**(2.0/3))*(93184*(et**4)*eta - 125361*(et**4) + 651252*(et**2)*eta - 880632*(et**2) + 228704*eta - 340968)))) +  ((-1/et)*25.6*eta*(np.pi)*n*(((G*m*n/c**3)**(2.0/3))**4)*((1 - et**2)**0.5)*((((1 - et**2)**0.5)*(1.0 + 7.260831042*(et**2) + 5.844370473*(et**4) + 0.8452020270*(et**6) + 0.7580633432e-1*(et**8) + 0.2034045037e-2*(et**10))/(1.0 - 4.900627291*(et**2) + 9.512155497*(et**4) - 9.051368575*(et**6) + 4.096465525*(et**8) - 0.5933309609*(et**10) - 0.5427399445e-1*(et**12) - 0.9020225634e-2*(et**14))) - (1.0*(1.0 + 1.893242666*(et**2) - 2.708117333*(et**4) + 0.6192474531*(et**6) + 0.5008474620e-1*(et**8) - 0.1059040781e-1*(et**10))/(1.0 - 4.638007334*(et**2) + 8.716680569*(et**4) - 8.451197591*(et**6) + 4.435922348*(et**8) -1.199023304*(et**10) + 0.1398678608*(et**12) - 0.4254544193e-2*(et**14))))),  (1/(M*T_sun*((1 - et**2)**0.5)**7))*n*((G*m*n/c**3)**(2.0/3))**4*eta*(((37.0/5)*et**4 + (292.0/5)*et**2 + 96.0/5)  +   ((1/((1 - et**2)**0.5)**2)*(-1.0/280)*((G*m*n/c**3)**(2.0/3))*(8288*(et**6)*eta - 11717*(et**6) + 141708*(et**4)*eta - 197022*(et**4)-219880*(et**2) + 159600*(et**2)*eta + 14784*eta - 20368)))  +   (((384.0/5)*n*(((G*m*n/c**3)**(2.0/3))**(11.0/2))*eta*np.pi*(1.0 + 7.260831042*(et**2) + 5.844370473*(et**4) + 0.8452020270*(et**6) + 0.7580633432e-1*(et**8) + 0.2034045037e-2*(et**10)))/(T_sun*M*(1.0 - 4.900627291*(et**2) + 9.512155497*(et**4) - 9.051368575*(et**6) + 4.096465525*(et**8) - 0.5933309609*(et**10) - 0.5427399445e-1*(et**12) - 0.9020225634e-2*(et**14))))]
    dx_dt_lead_1PN = [((1/((1 - et**2)**0.5)**5)*-1*n*eta*((G*m*n/c**3)**(2.0/3))**(5.0/2)*et*(((121.0/15)*et**2 + 304.0/15) - ((1/((1 - et**2)**0.5)**5)*(1/((1 - et**2)**0.5)**2)*(1.0/2520)*((G*m*n/c**3)**(2.0/3))*(93184*(et**4)*eta - 125361*(et**4) + 651252*(et**2)*eta - 880632*(et**2) + 228704*eta - 340968)))) ,  (1/(M*T_sun*((1 - et**2)**0.5)**7))*n*((G*m*n/c**3)**(2.0/3))**4*eta*(((37.0/5)*et**4 + (292.0/5)*et**2 + 96.0/5)  +   ((1/((1 - et**2)**0.5)**2)*(-1.0/280)*((G*m*n/c**3)**(2.0/3))*(8288*(et**6)*eta - 11717*(et**6) + 141708*(et**4)*eta - 197022*(et**4)-219880*(et**2) + 159600*(et**2)*eta + 14784*eta - 20368))) ]
    dx_dt_lead = [ (n*(1/((1 - et**2)**0.5)**5)*-1*eta*((G*m*n/c**3)**(2.0/3))**(5.0/2)*et*((121.0/15)*et**2 + 304.0/15))  ,   (n*(1/(M*T_sun*((1 - et**2)**0.5)**7))*(((G*m*n/c**3)**(2.0/3))**4)*eta*((37.0/5)*(et**4) + (292.0/5)*(et**2) + 96.0/5))]
    pb_dot_lead = dx_dt_lead[1]*-2*np.pi/(n**2)
    pb_dot_lead_1PN = dx_dt_lead_1PN[1]*-2*np.pi/(n**2)
    pb_dot_lead_1PN_tail = dx_dt_lead_1PN_tail[1]*-2*np.pi/(n**2)
    '''if mask==None:
        print("pb_dot_lead is "+ str(pb_dot_lead))
        print("pb_dot_lead_1PN is "+ str(pb_dot_lead_1PN))
        print("pb_dot_lead_1PN_tail is "+ str(pb_dot_lead_1PN_tail))
        print("e_dot_lead is "+ str(dx_dt_lead[0]))
        print("e_dot_lead_1PN is "+ str(dx_dt_lead_1PN[0]))
        print("e_dot_lead_1PN_tail is "+ str(dx_dt_lead_1PN_tail[0]))
    #return dx_dt_lead, dx_dt_lead_1PN, dx_dt_lead_1PN_tail  '''
    return pb_dot_lead_1PN , pb_dot_lead



def contours(m,gs,pos,pb_arr=None,ecc_arr=None):
    plt.clf()
    if pb_arr==None:
        pb_arr = np.linspace(0.1,50,100)
    else:
        pb_arr = pb_arr
    if ecc_arr ==None:
        ecc_arr = np.linspace(0.1,0.95,20)
    else:
        ecc_arr = ecc_arr
    P,ECC = np.meshgrid(pb_arr,ecc_arr)
    dx_dt_lead, dx_dt_lead_1PN, dx_dt_lead_1PN_tail = x_dot(m,P,ECC)
    #plt.subplot(gs[pos])
    plt.imshow(dx_dt_lead_1PN_tail[1],extent=[pb_arr[0],pb_arr[-1],ecc_arr[0],ecc_arr[-1]],interpolation='gaussian',norm=LogNorm(),origin='lower',aspect='auto')
    plt.xlabel('Period in hours')
    plt.ylabel('Eccentricity')
    plt.title('Time Variation of Frequency for binary system(masses of 1.4 and {m})'.format(m=m),fontsize='small')
    plt.colorbar()  
    #plotfile.savefig()


#check the code for Huse-Taylor binay
def check():
    c, G, M_sun, T_sun = astronomical_constants()
    m=1.4
    M = m+1.4
    eta = m*1.4/((m+1.4)**2)
    ecc_0 = 0.6171334
    pb_0 = 7.751938773864
    dx_dt_lead, dx_dt_lead_1PN, dx_dt_lead_1PN_tail = x_dot(m,pb_0,ecc_0)
    pb_dot = -2*np.pi*dx_dt_lead_1PN_tail[1]/(2*np.pi/(pb_0*u.hour.to(u.second)))**2 
    return pb_dot


def const_P(p0,ecc0,M,eta): 
    n0 = 2*np.pi/p0 
    chirp_mass = G*(M*M_sun*(eta**0.6)/(c**3)) 
    A = 0.2*(chirp_mass)**(5.0/3) 
    res = ((n0**(8.0/3))*(ecc0**(48.0/19))*((121*(ecc0**2) + 304.0)**(3480.0/2299)))/(3*(1 - ecc0**2)**4) 
    P = A*res 
    return chirp_mass,P



def appelf1(a,b,c,d,x,y):
    return mpmath.hyper2d({'m+n':[a], 'm':[b], 'n':[c]}, {'m+n':[d]}, x, y)


def tau_fn(e):
    return (e**(48./19))/(768 * appelf1(24./19, -1181./2299, 3./2, 43./19, -121/304*e**2, e**2))


def orbit_simulator(m,t,pb_arr=None,ecc_arr=None):
    #pos = GridSpec(1,2)
    c, G, M_sun, T_sun = astronomical_constants()

    tab = table(names=['Initial_Eccentricity', 'Initial Period', 'Orbits_lead', 'Orbits_lead_1PN', 'Orbits_lead_1PN_tail', 'Orbits_1PN_tail','Orbits_tail'])

    if pb_arr==None:
        pb_arr = np.linspace(0.25,18,35)
    else:
        pb_arr = pb_arr

    n_arr = 2*np.pi/(pb_arr*u.hour.to(u.second))

    if ecc_arr ==None:
        ecc_arr = np.linspace(0.05,0.85,16)
    else:
        ecc_arr = ecc_arr
    
    N,ECC = np.meshgrid(n_arr,ecc_arr)
    Orbits_contour_lead = np.zeros((len(N),len(N[0])))
    Orbits_contour_lead_1PN = np.zeros((len(N),len(N[0])))
    Orbits_contour_lead_1PN_tail = np.zeros((len(N),len(N[0])))
    Orbits_contour_1PN_tail = np.zeros((len(N),len(N[0])))
    Orbits_contour_tail = np.zeros((len(N),len(N[0])))

    for i in range(len(N)):
        for j in range(len(ECC[0])):
            gridpoint = (N[i][j], ECC[i][j])
            solution_lead, solution_lead_1PN, solution_lead_1PN_tail = solver(gridpoint[1],gridpoint[0],t)
            ecc, freq, phase, orbits = sim_sol(solution_lead)
            ecc_1PN, freq_1PN, phase_1PN, orbits_lead_1PN = sim_sol(solution_lead_1PN)
            ecc_tail, freq_tail, phase_tail, orbits_lead_1PN_tail = sim_sol(solution_lead_1PN_tail)

            #tab.add_row([np.round(ECC[i][j],3),np.round(2*np.pi*u.second.to(u.hour)/N[i][j],3), orbits[-1], orbits_lead_1PN[-1], orbits_lead_1PN_tail[-1], orbits_lead_1PN_tail[-1] - orbits[-1], orbits_lead_1PN_tail[-1] - orbits_lead_1PN[-1]])

            Orbits_contour_lead[i][j] = orbits[-1]
            Orbits_contour_lead_1PN[i][j] = orbits_lead_1PN[-1]
            Orbits_contour_lead_1PN_tail[i][j] = orbits_lead_1PN_tail[-1]
            Orbits_contour_1PN_tail[i][j] = orbits_lead_1PN_tail[-1] - orbits[-1]
            Orbits_contour_tail[i][j] = orbits_lead_1PN_tail[-1] - orbits_lead_1PN[-1]
            print('Simulating done for initial condition P = {x} and Eccentricity = {y} and mass = {m}'.format(m=m,x=2*np.pi*u.second.to(u.hour)/N[i][j],y=ECC[i][j]))
            #print('Ratio of lead_1PN_tail to lead_1PN is {r} for mass = {m}, period = {p} and eccentricity ={e}'.format(m=m,p=2*np.pi*u.second.to(u.hour)/N[i][j], e=ECC[i][j], r = 1.0*Orbits_contour_tail[i][j]/Orbits_contour_lead_1PN[i][j]))


    
    
    #tab.write('Orbits_Simulated.txt',format='ascii',overwrite=True)
    #print(Orbits_contour_1PN_tail)
    plotfile1 = PdfPages('./Plots/1PN_Tail_{m}.pdf'.format(m=m))
    plt.clf()
    plt.suptitle("For a binary system with masses 1.4 and {m}, orbits contributed by".format(m=m),fontsize='small')
    plt.imshow(abs(Orbits_contour_1PN_tail),extent=[pb_arr[0],pb_arr[-1],ecc_arr[0],ecc_arr[-1]],norm=LogNorm(),interpolation='gaussian',origin='lower',aspect='auto')
    plt.xlabel('Period in hours')
    plt.ylabel('Eccentricity')
    plt.title('1PN and tail',fontsize='small')
    plt.colorbar()
    plt.savefig('./Plots/1PN_Tail_{m}.png'.format(m=m))
    plotfile1.savefig()
    plotfile1.close()
    plotfile2 = PdfPages('./Plots/Tail_{m}.pdf'.format(m=m))
    plt.clf()
    plt.imshow(abs(Orbits_contour_tail),extent=[pb_arr[0],pb_arr[-1],ecc_arr[0],ecc_arr[-1]],norm=LogNorm(),interpolation='gaussian',origin='lower',aspect='auto')
    plt.xlabel('Period in hours')
    plt.ylabel('Eccentricity')
    plt.title('Tail alone',fontsize='small')
    plt.colorbar()
    plt.savefig("./Plots/Tail_{m}.png".format(m=m))
    plotfile2.savefig()
    plotfile2.close()
    #plotfile.close()

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Used to find the binary systems in which the effect of  1PN+tail can be measured")
    parser.add_argument("--Mass", help="Mass of the companion in the binary system, default is 1.4", type=float, default=9.0)
    #parser.add_argument("--MR", help="Mass Ratio os the binary system, default is 0.25", type=float, default=0.25)
    parser.add_argument("--P", help="Initial Period of the binary system, default is 24 hrs", type=float, default=8)
    parser.add_argument("--P_units", help="Unit of initial Period of the binary system, default is hrs, please use the notation sec, min, hrs, days, yrs to specify", type=str, default='hrs')
    parser.add_argument("--e", help="Initial eccentricity of the binary system, default is 0.5", type=float, default=0.7)
    parser.add_argument("--t", help="Observing time for the binary system, default is 10 yrs", type=float, default=20)
    parser.add_argument("--order", help="Prioritize the order in which we want to tweak the system, default is PMR i.e Period, then Mass, then Mass Ratio", type=str, default='PMR')

    args = parser.parse_args()
    t = np.linspace(0, args.t*u.year.to(u.second), args.t)
    masses = np.array([50.0])
    for m in masses:
        args.Mass = m
        orbit_simulator(m,t)
    
    '''y0 = [args.e,2*(np.pi)/((args.P)*u.hour.to(u.second)), 0]

    sol = odeint(lead, y0, t)

    res = sol[:,2] - (2*(np.pi)/(args.P*u.hour.to(u.second)))*t

    plt.plot(t, res)
    plt.show()'''
        

    