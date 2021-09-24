"""
Conventional moment-curvature analysis of rectangular sections.
Automating hand-calculations.
September 22, 2021.
"""

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import integrate

def b1_ACI(fcprime):
    """
    beta1 parameter per ACI 318 sec. 22.2.2.2.4.3
    units: psi
    """
    if fcprime < 2500:
        return None
    elif fcprime < 4000:
        return 0.85
    elif fcprime < 8000:
        return 0.85 - 0.05/1000*(fcprime-4000)
    else:
        return 0.65

def steel_stress_strain_bilin(e, fy):
    young = 29000000
    ey = fy/young
    if e < ey:
        return e * fy/ey
    else:
        return fy

def a706gr60_interp(strain):
    x = np.array(
        [-0.15, -0.1, -0.06, -0.04, -0.03, -0.0225, -0.0023828, 0.00, 0.0023828, 0.0225, 0.03, 0.04, 0.06, 0.1, 0.15])
    y = np.array([-95., -92., -86., -80., -74., -69.1, -69.1, 0., 69.1, 69.1, 74., 80., 86.,  92., 95.])
    y = y * 1000.
    f = interp1d(x, y, kind='linear')
    if strain > 0.15:
        return 95000.
    elif strain < -0.15:
        return -95000.
    else:
        return float(f(strain))
    

A706GR60 = {
    'fy': 60000,
    'fy_act': 69100,
    'interp_func': a706gr60_interp
}
    
@dataclass
class rect_sect:
    """
    Rectangular section.
    """
    bw: float = field() # web thickness (in)
    h: float = field() # section height (in)
    ast: float = field() # area of tention longitudinal reinforcement (in2)
    dt: float = field() # distance of that reinforcement from the extreme compression fiber (in)
    asc: float = field() # area of compression longitudinal reinforcement (in2)
    dc: float = field() # distance of that reinforcement from the extreme compression fiber (in)
    fcprime: float = field() # specified compressive strength of concrete (psi)
    steel : dict = field() # steel dictionary

    
    def get_resultant_forces(self, c, epsilon_c, procedure, args={}):
        """
        Resultant forces based on assumed
        linear stress profile
        """
        # resultant forces
        if procedure in ["ACI nominal", "ACI probable"]:
            # strains
            epsilon_st = epsilon_c / c * (self.dt - c)
            epsilon_sc = epsilon_c / c * (c - self.dc)

            b1 = b1_ACI(self.fcprime)
            if procedure == "ACI nominal":
                f_ts = self.ast * steel_stress_strain_bilin(epsilon_st, self.steel['fy'])
                f_cs = self.asc * (steel_stress_strain_bilin(epsilon_sc, self.steel['fy']) - 0.85 * self.fcprime)
            elif procedure == "ACI probable":
                f_ts = self.ast * steel_stress_strain_bilin(epsilon_st, self.steel['fy']*1.25)
                f_cs = self.asc * (steel_stress_strain_bilin(epsilon_sc, self.steel['fy']*1.25) - 0.85 * self.fcprime)
            f_c  = 0.85 * self.fcprime * (b1 * c * self.bw)

        elif procedure == "Expected Spalling State":
            # strains
            ecu = 0.004
            epsilon_st = ecu / c * (self.dt - c)
            epsilon_sc = ecu / c * (c - self.dc)

            fccprime = 0.85 * self.fcprime
            ecc = 0.002
            
            young_conc = 57000 * np.sqrt(self.fcprime)
            r = young_conc / (young_conc - fccprime / ecc)
            
            def fc(e, fccprime, ecc, r):
                return fccprime * e / ecc * r / (r - 1 + (e/ecc)**r)
            
            def integrand1(e, fccprime, ecc, r):
                return e * fc(e, fccprime, ecc, r)
            beta1 = 2. * (
                1. - 1./ecu * integrate.quad(integrand1, 0.00, ecu, args=(fccprime, ecc, r))[0] / \
                integrate.quad(fc, 0., ecu, args=(fccprime, ecc, r))[0] )
            alpha1 = 1./ecu * integrate.quad(fc, 0., ecu, args=(fccprime, ecc, r))[0] / (fccprime * beta1)

            f_ts = self.ast * self.steel['interp_func'](epsilon_st)
            if epsilon_sc > 0:
                f_cs = self.asc * (self.steel['interp_func'](epsilon_sc) - alpha1 * self.fcprime)
            else:
                f_cs = self.asc * self.steel['interp_func'](-epsilon_sc)
            f_c  = alpha1 * fccprime * self.bw * beta1 * c

        elif procedure == "Expected Max State":
            fccprime = args['fccprime']
            ecc = args['ecc']
            ecu = args['ecu']
            cover = args['cover']
            dt_spall = self.dt - cover
            dc_spall = self.dc - cover

            # strains
            epsilon_st = ecu / c * (dt_spall - c)
            epsilon_sc = ecu / c * (c - dc_spall)

            young_conc = 57000 * np.sqrt(self.fcprime)
            r = young_conc / (young_conc - fccprime / ecc)
            
            def fc(e, fccprime, ecc, r):
                return fccprime * e / ecc * r / (r - 1 + (e/ecc)**r)
            def integrand1(e, fccprime, ecc, r):
                return e * fc(e, fccprime, ecc, r)
            beta1 = 2. * (
                1. - 1./ecu * integrate.quad(integrand1, 0.00, ecu, args=(fccprime, ecc, r))[0] / \
                integrate.quad(fc, 0., ecu, args=(fccprime, ecc, r))[0] )
            alpha1 = 1./ecu * integrate.quad(fc, 0., ecu, args=(fccprime, ecc, r))[0] / (fccprime * beta1)

            f_ts = self.ast * self.steel['interp_func'](epsilon_st)
            if epsilon_sc > 0:
                f_cs = self.asc * (self.steel['interp_func'](epsilon_sc) - alpha1 * fccprime)
            else:
                f_cs = self.asc * self.steel['interp_func'](-epsilon_sc)
            f_c  = alpha1 * fccprime * (self.bw-2.*cover) * beta1 * c
        else:
            raise ValueError("Invalid procedure", procedure)
            
        return f_ts, f_cs, f_c

            
    
    def resultant_axial_force(self, c, epsilon_c, procedure, args={}):
        """
        Free-body-diagram axial force resultant based on assumed
        linear stress profile
        """
        f_ts, f_cs, f_c = self.get_resultant_forces(c, epsilon_c, procedure, args)
        return f_c + f_cs - f_ts

    def determine_compression_zone(self, external_axial_load, c_init, epsilon_c, procedure, args={}):
        """
        Determine the compression zone that
        balacnes the internal resultant forces with the
        externally applied compressive load (in lb).
        Start iterating from a value of c = c_init
        """
        def temp_obj_fun(x):
            return (self.resultant_axial_force(x, epsilon_c, procedure, args) - \
                    external_axial_load)**2
        res = minimize(temp_obj_fun, c_init)
        c_opt = res.x[0]
        assert(temp_obj_fun(c_opt) < 1e-2), "Error: Can't determine compression zone"
        return c_opt

    def resultant_moment(self, c, epsilon_c, procedure, args={}):
        f_ts, f_cs, f_c = self.get_resultant_forces(c, epsilon_c, procedure, args)
        p = self.resultant_axial_force(c, epsilon_c, procedure, args)
        if procedure in ["ACI nominal", "ACI probable"]:
            b1 = b1_ACI(self.fcprime)
            beta = b1
        elif procedure == "Expected Spalling State":
            ecu = 0.004
            ecc = 0.002
            fccprime = 0.85 * self.fcprime
            young_conc = 57000 * np.sqrt(self.fcprime)
            r = young_conc / (young_conc - fccprime / ecc)

            def fc(e, fccprime, ecc, r):
                return fccprime * e / ecc * r / (r - 1 + (e/ecc)**r)

            def integrand1(e, fccprime, ecc, r):
                return e * fc(e, fccprime, ecc, r)

            beta1 = 2. * (
                1. - 1./ecu * integrate.quad(
                    integrand1, 0.00, ecu, args=(fccprime, ecc, r))[0] /
                integrate.quad(fc, 0., ecu, args=(fccprime, ecc, r))[0])
            beta = beta1
        elif procedure == "Expected Max State":
            ecu = args['ecu']
            ecc = args['ecc']
            fccprime = args['fccprime']
            young_conc = 57000 * np.sqrt(self.fcprime)
            r = young_conc / (young_conc - fccprime / ecc)

            def fc(e, fccprime, ecc, r):
                return fccprime * e / ecc * r / (r - 1 + (e/ecc)**r)

            def integrand1(e, fccprime, ecc, r):
                return e * fc(e, fccprime, ecc, r)

            beta1 = 2. * (
                1. - 1./ecu * integrate.quad(
                    integrand1, 0.00, ecu, args=(fccprime, ecc, r))[0] /
                integrate.quad(fc, 0., ecu, args=(fccprime, ecc, r))[0])
            beta = beta1
        else:
            raise ValueError("Invalid procedure", procedure)
        return p * self.h/2. + \
            f_ts * self.dt - f_c * 0.5 * beta * c - f_cs * self.dc

    def moment_strength(self, axial_compression, procedure, args={}):
        c_opt = self.determine_compression_zone(axial_compression, 4.00, 0.003, procedure, args)
        if procedure in ["ACI nominal", "ACI probable"]:
            mom_str = self.resultant_moment(c_opt, 0.003, procedure, args)
        elif procedure == "Expected Spalling State":
            mom_str = self.resultant_moment(c_opt, 0.004, procedure, args)
        elif procedure == "Expected Max State":
            ecu = args['ecu']
            mom_str = self.resultant_moment(c_opt, ecu, procedure, args)
        return mom_str

    def moment_curv_crack(self):
        """
        Returns the cracking moment and curvature of the section.
        """
        # modulus of rupture
        fr = 7.5 * np.sqrt(self.fcprime)
        # moment of inertia around centroid
        ig = self.bw * self.h**3 / 12.0
        m_cr = fr * ig / (self.h/2.0)  # lb in
        young_conc = 57000 * np.sqrt(self.fcprime)
        phi_cr = m_cr / (young_conc * ig)  # 1/in
        return m_cr, phi_cr

    def moment_curv_yield(self):
        """
        Moment and curvature of the section at the onset of the longitudinal
        reinforcement yielding, under the assumption that the concrete stress-strain
        is in the linear range.
        Caution: uses closed-form solutions from the book, only works when
        no axial load is present.
        """
        young_conc = 57000 * np.sqrt(self.fcprime)
        n = 29000000/young_conc
        rho_t = self.ast/self.bw/self.dt
        rho_c = self.asc/self.bw/self.dt
        kapa = np.sqrt((rho_t+rho_c)**2 * n**2 + 2. * (rho_t + rho_c * self.dc/self.dt)*n) - \
            (rho_t + rho_c) * n
        i_cr = self.bw * (kapa*self.dt)**3/3. + \
            (n - 1.) * self.asc * (kapa*self.dt-self.dc)**2 + \
            n * self.ast * (self.dt - kapa * self.dt)**2
        m_y = 1 / n * self.steel['fy_act'] * i_cr / (self.dt - kapa * self.dt)
        phi_y = m_y / (young_conc * i_cr)  # 1/in
        fc = m_y * kapa * self.dt / i_cr
        assert fc < self.fcprime, "ERROR: Concrete has gone nonlinear. Results unreliable."
        return m_y, phi_y

    def moment_curv_spall(self, args):
        procedure = "Expected Spalling State"
        mu = self.moment_strength(0.00, procedure, args)
        ecu = 0.004
        c_opt = self.determine_compression_zone(0.00, 4.00, ecu, procedure, args)
        phi = ecu/c_opt
        return mu, phi

    def moment_curv_ult_tensile(self, axial_compression, args):
        fccprime = args['fccprime']
        ecc = args['ecc']
        ecu = args['ecu']
        esmax = args['esmax']
        cover = args['cover']
        dt_spall = self.dt - cover
        dc_spall = self.dc - cover
        def resultants(c):
            epsilon_st = esmax
            epsilon_sc = esmax / (dt_spall - c) * (c - dc_spall)
            epsilon_c = esmax / (dt_spall - c) * c
            young_conc = 57000 * np.sqrt(self.fcprime)
            r = young_conc / (young_conc - fccprime / ecc)
            
            def fc(e, fccprime, ecc, r):
                return fccprime * e / ecc * r / (r - 1 + (e/ecc)**r)
                
            def integrand1(e, fccprime, ecc, r):
                return e * fc(e, fccprime, ecc, r)
            
            beta1 = 2. * (
                1. - 1./epsilon_c * integrate.quad(
                    integrand1, 0.00, epsilon_c,
                    args=(fccprime, ecc, r))[0] /
                integrate.quad(fc, 0., epsilon_c, args=(fccprime, ecc, r))[0])
            alpha1 = 1./epsilon_c * integrate.quad(
                fc, 0., epsilon_c,
                args=(fccprime, ecc, r))[0] / (fccprime * beta1)
            f_ts = self.ast * self.steel['interp_func'](epsilon_st)
            if epsilon_sc > 0:
                f_cs = self.asc * (
                    self.steel['interp_func'](epsilon_sc) -
                    alpha1 * self.fcprime)
            else:
                f_cs = self.asc * self.steel['interp_func'](-epsilon_sc)
            f_c = alpha1 * fccprime * (self.bw-2.*cover) * beta1 * c
            return f_c, f_cs, f_ts, beta1, epsilon_c

        def objective(c, compression):
            f_c, f_cs, f_ts, _, _ = resultants(c)
            return (f_c + f_cs - f_ts - compression)**2
        res = minimize(objective, 2.00, args=(axial_compression))
        c_opt = res.x[0]
        assert(objective(c_opt, axial_compression) < 1e-2),\
            "Error: Can't determine compression zone"
        f_c, f_cs, f_ts, b1, ec = resultants(c_opt)
        mu = axial_compression * self.h/2. + f_ts * \
            self.dt - f_c * 0.5 * b1 * c_opt - f_cs * self.dc
        phi = ec/c_opt
        return mu, phi
    

    def moment_curv_ult(self, args):
        procedure = "Expected Max State"
        mu = self.moment_strength(0.00, procedure, args)
        ecu = args['ecu']
        c_opt = self.determine_compression_zone(0.00, 4.00, ecu, procedure, args)
        # check if tention steel exceeds strain capacity
        cover = args['cover']
        dt_spall = self.dt - cover
        epsilon_st = ecu / c_opt * (dt_spall - c_opt)
        esmax = args['esmax']
        if epsilon_st > esmax:
            # tention steel exceeds strain capacity, must redo calculations
            mu, phi = self.moment_curv_ult_tensile(0.00, args)
        else:
            # calculations ok
            phi = ecu/c_opt
        
        return mu, phi

    def print_envelope(self, args):
        print()
        print("Cracking moment and curvature (kip-in, 1/in):")
        print(self.moment_curv_crack()[0]/1000, self.moment_curv_crack()[1])
        print()

        print("Yield moment and curvature (kip-in, 1/in):")
        print(self.moment_curv_yield()[0]/1000, self.moment_curv_yield()[1])
        print()

        print("Spalling moment and curvature (kip-in, 1/in):")
        res_spall = self.moment_curv_spall(args)
        print(res_spall[0]/1000, res_spall[1])
        print()

        print("Ultimate moment and curvature (kip-in, 1/in):")
        res_ult = self.moment_curv_ult(args)
        print(res_ult[0]/1000, res_ult[1])
        print()

    def get_envelope(self, args):
        p0 = (0., 0.)
        p1 = self.moment_curv_crack()
        p2 = self.moment_curv_yield()
        p3 = self.moment_curv_spall(args)
        p4 = self.moment_curv_ult(args)
        res = np.array([p0, p1, p2, p3, p4])
        return res



    
## ~~~~~~~~~~~ %%
## APPLICATION %%
## ~~~~~~~~~~~ %%

# # book example - results OK

# example_sec = rect_sect(18.,
#                         24.,
#                         4*1.00,
#                         21.4,
#                         2*1.00,
#                         2.6,
#                         4000.,
#                         A706GR60)

# args = {
#     'fccprime': 5120.0,
#     'ecc': 0.0048,
#     'ecu': 0.015,
#     'cover': 1.5,
#     'esmax': 0.14
# }

# example_sec.moment_strength(0.00, "ACI nominal")/1000
# example_sec.moment_strength(0.00, "ACI probable")/1000



# example_sec.determine_compression_zone(
#     0.00,
#     4.00,
#     0.004,
#     "Expected Spalling State",
#     args)
# # (fail)

# example_sec.get_resultant_forces(
#     4.03, 0.004,
#     "Expected Spalling State", args)

# example_sec.resultant_axial_force(
#     4.035839861877991, 0.004,
#     "Expected Spalling State", args)



# res = example_sec.get_envelope(args)

# # spalling
# res[3,1]*1e5
# res[3,0]/1000

# # ultimate
# res[4,1]*1e5
# res[4,0]/1000


# plt.figure(figsize=(8,4))
# plt.plot(res[:,1], res[:,0]/1000., 'k')
# plt.scatter(res[:,1], res[:,0]/1000.,  s=80, facecolors='none', edgecolors='k')
# plt.grid()
# plt.xlabel("Curvature ($in^{-1}$)")
# plt.ylabel("Moment ($kip \cdot in$)")
# plt.show()
# plt.close()






my_sec = rect_sect(18.0,
                   24.0,
                   5*0.79,
                   24.-1.5-0.25,
                   3*0.79,
                   1.5+0.25,
                   4000.,
                   A706GR60)

args = {
    'fccprime': 6511.9,
    'ecc': 0.011153,
    'ecu': 0.05144,
    'cover': 1.5,
    'esmax': 0.16
}

print()
print(" ~~~~~~~~~ ")
print(" Problem 1 ")
print(" ~~~~~~~~~ ")
print()
print()

print("Nominal moment strength")
print()

c_opt = my_sec.determine_compression_zone(
    0.00, 3.00, 0.003, "ACI nominal")
m_nom = my_sec.moment_strength(0.0, "ACI nominal") / 1000.
print("  - optimum c value:", c_opt)
print()

print("  - moment (kip-in):", m_nom)
print()

print("Probable moment strength")
print()

c_opt = my_sec.determine_compression_zone(
    0.00, 3.00, 0.003, "ACI probable")
m_prob = my_sec.moment_strength(0.0, "ACI probable") / 1000.
print("  - optimum c value:", c_opt)
print()

print("  - moment (kip-in):", m_prob)
print()


print()
print(" ~~~~~~~~~ ")
print(" Problem 2 ")
print(" ~~~~~~~~~ ")
print()
print()

print("(a) Moment and Curvature at Cracking")
print()
res = my_sec.moment_curv_crack()
mom_crack = res[0] / 1000
curv_crack = res[1]
print("  - moment (kip-in):", mom_crack)
print()
print("  - curvature (in^-1):", curv_crack)
print()

print("(b) Moment and Curvature at onset of long. rein. yielding")
print()
res = my_sec.moment_curv_yield()
mom_yield = res[0] / 1000
curv_yield = res[1]
print("  - moment (kip-in):", mom_yield)
print()
print("  - curvature (in^-1):", curv_yield)
print()

print("(c) Moment and Curvature at onset of cover spalling")
print()
res = my_sec.moment_curv_spall(args)
mom_spall = res[0] / 1000
curv_spall = res[1]
print("  - moment (kip-in):", mom_spall)
print()
print("  - curvature (in^-1):", curv_spall)
print()

print("(e) Moment and Curvature at max. confined strength of concrete")
print()
res = my_sec.moment_curv_ult(args)
mom_ult = res[0] / 1000
curv_ult = res[1]
print("  - moment (kip-in):", mom_ult)
print()
print("  - curvature (in^-1):", curv_ult)
print()





res = my_sec.get_envelope(args)
xtract_data = np.genfromtxt('resources/hw4/xtract_output.txt', skip_header=16, delimiter='\t')
xtract_data = xtract_data[:,0:2]


plt.figure(figsize=(8,4))
plt.plot(-xtract_data[:,1], -xtract_data[:,0], 'red')
plt.plot(res[:,1], res[:,0]/1000., 'k')
plt.scatter(res[:,1], res[:,0]/1000.,  s=80, facecolors='none', edgecolors='k')
plt.plot(np.array([0., 0.01]), np.array([m_nom, m_nom]), "--")
plt.plot(np.array([0., 0.01]), np.array([m_prob, m_prob]), "--")
plt.grid()
plt.xlabel("Curvature ($in^{-1}$)")
plt.ylabel("Moment ($kip \cdot in$)")
plt.show()
plt.close()

