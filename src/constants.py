# Physics Constants
p_outlet = (101325 - 17825) / (10**5)
Tref = 273.15
T = 298.15
mu_ref = 1.716e-5
S = 110.4
#mu = round(mu_ref * (T / Tref) ** (1.5) * ((Tref + S) / (T + S)), 8)
M = 28.96 / 1000
R = 8.314
rho = ((p_outlet * 10**5) * M) / (R * T)
thermal_conductivity = 2.61E-02
specific_heat = 1.00E+03  #at constant pressure
density = 9.7118E-01  # kg/m^3
T_inlet = 293.15 #K
T_wall = 338.15 #K