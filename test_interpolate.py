import numpy as np
import matplotlib.pyplot as plt
import match_utils
from scipy.interpolate import interp1d
from sys import exit
from pdb import set_trace

infile = '/astro/store/phat/arlewis/SFH/project/B02/15x30/WFC/M31-B02_15x30-450.zc.fit_unbinned.final.sfh'

data = match_utils.read_zcombined_sfh(infile)

t0 = data['logagei']
tf = data['logagef']
sfr = data['sfr']

sfr[0] = sfr[0] * (10**tf[0] - 10**t0[0]) / 10*tf[0]
t0[0] = 0
mass = sfr * (10**tf - 10**t0)
cum_mass = np.cumsum(mass)

tmid = np.mean((t0, tf), axis=0)

t0_reverse = t0[::-1]
tf_reverse = tf[::-1]
tmid_reverse = tmid[::-1]
sfr_reverse = sfr[::-1]
mass_reverse = mass[::-1]
cum_mass_reverse = np.cumsum(mass_reverse)

sel = t0_reverse <= 8.0
t0_sel = t0_reverse[sel]
tf_sel = tf_reverse[sel]
tmid_sel = tmid_reverse[sel]
sfr_sel = sfr_reverse[sel]
mass_sel = mass_reverse[sel]
cum_mass_sel = np.cumsum(mass_sel)
norm_cum_mass_sel = cum_mass_sel / np.max(cum_mass_sel)

norm_cum_mass_sel = np.append(0.0, norm_cum_mass_sel)
time = np.append(tf_sel, 0)

#cum_mass_sel = np.append(cum_mass_sel, cum_mass_sel[-1])

new_x = []
new_cum_mass = []
new_sfr = []
for i in range(1, len(tf_sel)):
    x = [10**tf_sel[i]/1e6, 10**t0_sel[i]/1e6]
    y = [cum_mass_sel[i-1], cum_mass_sel[i]]
    f = interp1d(x, y, bounds_error=False)
    
    num = abs(int(np.around(x[0]) - np.around(x[1])))
    xx = np.linspace(np.around(x[0]), np.around(x[1]), num+1) # time in Myr
    yy = f(xx)  #cumulative mass
 
    tmp1 = (xx[:-1] - xx[1:]) * 1e6  # time diff in yr
    tmp2 = yy[1:] - yy[:-1]  # mass formed per time bin in M_sun
    yy2 = tmp2 / tmp1 # sfr per time bin in M_sun / yr

    if np.isnan(yy[-1]):
        yy[-1] = yy[-2]
    if np.isnan(yy[0]):
        yy[0] = yy[1]

    if i == len(tf_sel)-1:
        xx[-1] = 0
        [new_x.append(xxx) for xxx in xx]
        [new_cum_mass.append(yyy) for yyy in yy]
    else:
        [new_x.append(xxx) for xxx in xx[:-1]]
        [new_cum_mass.append(yyy) for yyy in yy[:-1]]
    [new_sfr.append(yyy2) for yyy2 in yy2]
   


plt.figure()
plt.plot(10**t0_sel/1e6, cum_mass_sel, 'k', lw=2)
plt.plot(10**t0_sel/1e6, cum_mass_sel, 'r.', ms=15)
plt.gca().invert_xaxis()
plt.xlim(105, -5)
plt.plot(new_x, new_cum_mass, 'b--', lw=2)
plt.plot(new_x, new_cum_mass, 'y.', ms=10)

plt.show()
exit()
x = 10**tf_sel/1e6
y = cum_mass_sel
f = interp1d(x, y, bounds_error=False)
xx = np.linspace(0, 100, 76)[::-1]
yy = f(xx)

subtract_array = np.append(0, yy[:-1])
mm = yy - subtract_array
ss = mm / 1e6

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,8), sharex=True)
ax1.plot(x, y, 'k', lw=2)
ax1.plot(x, y, 'r.', ms=15)
ax1.plot(xx, yy, 'b--', lw=2)

ax2.plot(x, sfr_sel*1e4, 'k', lw=2)
ax2.plot(x, sfr_sel*1e4, 'r.', ms=15)
ax2.plot(xx, ss*1e4, 'b--', lw=2)

ax1.set_xlim(105, -5)
ax2.set_ylim(-0.2, 3.7)
plt.subplots_adjust(hspace=0.01)


plt.show()
