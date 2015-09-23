import matplotlib.pyplot as plt

def plot_correlation(taxis, yarr, sym, lcol, fcol, ecol, ms, lab, xlab, ylab, 
					plotfile, dpi=300, save=False):
	fig = plt.figure(figsize=(8,8))
	plt.plot([-1e6, 1e6], [0,0], '-', color='gray', lw=2)
	
	for j in range(0, len(yarr)):
		plt.plot(taxis, yarr[j], sym[j], color=lcol[j], 
				 markerfacecolor=fcol[j],
				 markeredgecolor=ecol[j], markersize=ms[j], label=lab[j],
				 linewidth=2, linestyle='-')
	
	plt.legend(numpoints=1, markerscale=1.5, frameon=False, loc=0)
	plt.xlim(taxis[0], taxis[-1])
	plt.ylim(-0.1, 1.001)
	plt.xlabel(xlab, fontsize=18)
	plt.ylabel(ylab, fontsize=18)
	plt.tick_params(axis='both', labelsize=14)
	
	if save:
		plt.savefig(plotfile, dpi=dpi, bbox_inches='tight')
	else: 
		plt.show()
