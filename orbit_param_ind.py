#!/usr/bin/env python2.7

import argparse
from astropy import constants as con
import numpy as np
from astropy import units as u
from astropy.io import ascii
import mpmath
from astropy.table import Table, Column


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


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Snippet to calculate initial semimajpr axis and the time befor merger of a binary system')
	parser.add_argument('pb', help='Initial Period of the binary in hrs',type=float)
	parser.add_argument('ecc', help='Initial eccentricity of the binary',type=float)
	parser.add_argument('mass', help='Total mass of the binary',type=float)
	parser.add_argument('--mass_pulsar', help='Total mass of the binary',default=1.4,type=float)

	args = parser.parse_args()

	period = args.pb
	eccentricity = args.ecc
	m_tot = args.mass
	m_pul = args.mass_pulsar
	
	pul_semi_major_axis, ephimeris_period = semi_major_axis(period, eccentricity, m_tot, m_pul)

	#Time before merger

	m_chirp, constant_P = const_P(period, eccentricity, m_tot, m_pul)
	life_time = tau_fn(eccentricity)/constant_P
	
	print"\nSemi major axis of the pulsar is {a} lt-sec\n".format(a=pul_semi_major_axis)
	print"The time period of the binary is {p} days\n".format(p = ephimeris_period)
	print"Time before merger from the epoch is {t} yrs\n".format(t = life_time*u.second.to(u.year))

