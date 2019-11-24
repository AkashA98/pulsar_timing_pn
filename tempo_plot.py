#!/usr/bin/env python2.7

import numpy as np
from matplotlib import pyplot as plt
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Snippet used to plot tempo2 residuals')

	parser.add_argument('file', help='name of the text file containing residuals', type=str)
	parser.add_argument('P', help='period of the binary')
	parser.add_argument('e', help='Eccentricity of the orbit', type=str)

	args = parser.parse_args()

	data = np.genfromtxt(args.file)

	(_, caps, _) = plt.errorbar(data[:,4]-np.average(data[:,4]), data[:,2]*1e6, data[:,3], fmt='s', markersize=5, capsize=2, color='black')

	for cap in caps:
		cap.set_markeredgewidth(1)

#	plt.errorbar(data[:,4]-np.average(data[:,4]), data[:,2]*1e6, data[:,3], fmt='', mfc='black', mec='black')

	plt.xlabel('MJD - {:06.2f}'.format(np.average(data[:,4])))

	plt.ylabel(r'Post fit residuals ($\mu s$)')

	plt.savefig('P_{p}_e_{E}.pdf'.format(p=args.P, E=args.e))

	plt.show()	