import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,15,15)
ppo=[-0.05121775267199921, -0.0579021432610912, -0.06352966148271676, -0.061987859511637794, -0.05793473481251039, -0.06324444358381293, -0.06594541530750034, -0.06144300725105972, -0.0640431904712357, -0.06080681192612906, -0.058575912953051416, -0.061051457045183555, -0.06365670642181193, -0.0650338937865773, -0.06355239093763375]
ppo_upper_limit= [ele + ele * 0.1 for ele in ppo]
ppo_lower_limit= [ele - ele * 0.1 for ele in ppo]
sac=[-0.15, -0.08277777390633538, -0.057973059488687195, -0.04744855525038871, -0.04230242513066589, -0.037750882087645876, -0.03868267915425601, -0.038557030777465966, -0.03707205643913492, -0.03624830151713172, -0.036151682381675224, -0.03498693858281901, -0.03517617965900591, -0.03542881259767308, -0.035054092905816035]
sac_upper_limit= [ele + ele * 0.1 for ele in sac]
sac_lower_limit= [ele - ele * 0.1 for ele in sac]
nes=[-0.04647689231861361, -0.03713331259417363, -0.046693846907829184, -0.06142199780322986, -0.06641771768596269, -0.08948069217482746, -0.08386318530105866, -0.07346581120731586, -0.07678466097221176, -0.07965883762086115, -0.07606481582457317, -0.07725097973372001, -0.07734751067728451, -0.07899313417371835, -0.08706025981297474]
nes_upper_limit= [ele + ele * 0.1 for ele in nes]
nes_lower_limit= [ele - ele * 0.1 for ele in nes]
plt.plot(x, nes, color='green', label='NESS')
plt.fill_between(x, nes_upper_limit, nes_lower_limit, alpha = .1, color='green')
plt.plot(x, ppo, color='red', label='PPOS')
plt.fill_between(x, ppo_upper_limit, ppo_lower_limit, alpha = .1, color='red')
plt.plot(x, sac, color='blue', label='SEPS')
plt.fill_between(x, sac_upper_limit, sac_lower_limit, alpha = .1, color='blue')
ax = plt.axes()
#ax.set_facecolor('whitesmoke')
ax.set_xlim(left=0,right=14)
#plt.grid(alpha = .5, color='white')
ax.legend(loc='lower right')
plt.title("Comparison of Average returns vs Interactions")
plt.xlabel("Number of Interactions")
plt.ylabel("Average returns")
plt.savefig("Avg_returns_vs_interactions_combined.jpg",dpi=300)
plt.close()
