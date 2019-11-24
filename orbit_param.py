#!/usr/bin/env python2.7

import argparse
from astropy import constants
import numpy as np
from astropy import units as u
from astropy.io import ascii
import mpmath
from astropy.table import Table, Column


def astronomical_constants():
    global c, G, M_sun, T_sun
    c = constants.c.value
    G = constants.G.value
    M_sun = constants.M_sun.value
    T_sun = constants.GM_sun.value/((constants.c.value)**3)


def semi_major_axis(pb,ecc,m,m1):
	p = pb*u.hour.to(u.second)
	axis_cube = (p**2)*con.G.value*m*con.M_sun.value/(4*(np.pi)**2)
	axis = axis_cube**(1.0/3)
	pulsar_axis = axis*((m-m1)/m)
	pulsar_axis_lts = pulsar_axis/con.c.value
	period_ephimeris = p*u.second.to(u.day)

	return pulsar_axis_lts, period_ephimeris


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


def freq_var(m,M,ecc,n,eta):
    
    n_dot_lead = (((1.0/(M*T_sun*(((1 - (ecc**2))**(0.5))**7.0)))*(((G*m*n/(c**3))**(2.0/3))**4.0)*eta)*n*((37.0/5)*(ecc**4.0) + (292.0/5)*(ecc**2)+ (96.0/5)))

    n_dot_1PN = (((1.0/(M*T_sun*(((1 - (ecc**2))**(0.5))**7.0)))*(((G*m*n/(c**3))**(2.0/3))**4.0)*eta))*n*(-1.0/280)*(((1 - (ecc**2))**(0.5))**-2)*\
        ((G*m*n/(c**3))**(2.0/3))*(8288*(ecc**6)*eta - 11717*(ecc**6) + 141708*(ecc**4)*eta - 197022*(ecc**4)\
            - 219880*(ecc**2) + 159600*(ecc**2)*eta + 14784*eta - 20368)

    n_dot_tail = n*(384.0/5)*(((G*m*n/(c**3))**(2.0/3))**5.5)*eta*(np.pi)*((M*T_sun)**-1)*(1.0 + 7.260831042*(ecc**2) + 5.844370473*(ecc**4)+ 0.8452020270*(ecc**6)+\
        0.07580633432*(ecc**8) + 0.002034045037*(ecc**10))/(1.0 - 4.900627291*(ecc**2) + 9.512155497*(ecc**4) - 9.051368575*(ecc**6) + 4.096465525*(ecc**8)-\
            0.5933309609*(ecc**10) - 0.05427399445*(ecc**12)- 0.009020225634*(ecc**14))

    pb_dot_lead = n_dot_to_pb_dot(n, n_dot_lead)
    pb_dot_lead_1PN = n_dot_to_pb_dot(n, (n_dot_lead+n_dot_1PN))
    pb_dot_lead_1PN_tail = n_dot_to_pb_dot(n, (n_dot_lead+n_dot_1PN+n_dot_tail))
    pb_dot_1PN = pb_dot_lead_1PN - pb_dot_lead
    pb_dot_tail = pb_dot_lead_1PN_tail - pb_dot_lead_1PN

    return pb_dot_lead
    #return n_dot_lead, n_dot_1PN, n_dot_tail, pb_dot_lead, pb_dot_1PN, pb_dot_tail


def n_dot_to_pb_dot(n,n_dot):
    pb_dot = -2.0*(np.pi)*n_dot/(n**2)

    return pb_dot


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Snippet to calculate initial semimajpr axis and the time befor merger of a binary system')
	parser.add_argument('--pb', help='Initial Period of the binary in hrs',type=float)
	parser.add_argument('--ecc', help='Initial eccentricity of the binary',type=float)
	parser.add_argument('mass', help='Total mass of the binary',type=float)
	parser.add_argument('--mass_pulsar', help='Total mass of the binary',default=1.4,type=float)

	args = parser.parse_args()

	#period = args.pb
	#eccentricity = args.ecc
	m_tot = args.mass
	m_pul = args.mass_pulsar
	eta = m_pul*(m_tot-m_pul)/(m_tot**2)
	astronomical_constants()
	tab = Table(names=['Period', 'Eccentricity',  'Pb_dot','Merger_time in yrs', 'Linear_merger_time in Myr', 'Ratio(Linear_merger_time/Merger_time)'])

	#pul_semi_major_axis, ephimeris_period = semi_major_axis(period, eccentricity, m_tot, m_pul)

	#Time before merger

	pb_arr = np.arange(0.5,24,1.0)
	ecc_arr = np.append(np.linspace(1e-6,0.099,20),np.linspace(0.1,0.95,20))

	for i in pb_arr:
		for j in ecc_arr:
			m_chirp, constant_P = const_P(i, j, m_tot, m_pul)
			life_time = tau_fn(j)/constant_P
			life_time_yrs = life_time*(u.second.to(u.year))
			pb_der = -1*freq_var(m_tot*M_sun, m_tot, j, 2*np.pi/(i*u.hour.to(u.second)), eta)
			linear_time = i*(u.hour.to(u.second))/pb_der
			linear_time_Myr = linear_time*(u.second.to(u.year))*1e-6
			tab.add_row([i,j,pb_der,life_time_yrs, linear_time_Myr, linear_time/life_time])

	print(tab)
	tab.write('Timescales.txt', format='ascii',delimiter='\t', overwrite=True)
'''
	print"\nSemi major axis of the pulsar is {a} lt-sec\n".format(a=pul_semi_major_axis)
	print"The time period of the binary is {p} days\n".format(p = ephimeris_period)
	print"Time before merger from the epoch is {t} yrs\n".format(t = life_time*u.second.to(u.year))
'''
