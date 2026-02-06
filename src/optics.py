# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
optics.py
Created on Fri Mar 11 12:53:36 2022

"""
from sympy.abc import*
from sympy import*
from libreflection import *
import mpmath as mp
import numpy as np
import scipy as sp
import libphyscon as pc

class optics(branch):
    """

    """
    _name = "optics"
    
    def define_symbols(self):
        """
        Common global symbols, functions.
        a: 
        F: 
        """
        global k,m
        global x,y,z,x0,y0,z0
        global x0min,x0max,y0min,y0max
        global l,t
        global _El, Eli, _I, Uapr
        
        k,m = symbols('k m', real=True, positive=True)
        x,y,z,x0,y0,z0 = symbols('x y z x_0 y_0 z_0', nonzero=True)
        x0min,x0max,y0min,y0max = symbols('x_0min x_0max y_0min y_0max', nonzero=True)
        l,t = symbols('lambda t', real=True, positive=True)
        
        _El   = Function('E')(x,y,z)
        _I    = Function('I')(x,y,z)
        Eli   = Function('E_i')(x0,y0)
        Uapr  = Function('U_apr')(x0,y0)
    
    def __init__(self, class_type='default'):
        """
        a: 
        F:
            
        integrate(exp(I*k*y0**2/(2*z))*exp(-I*k*y*y0/z), y0)
        
        Fraunhofer_Diff_Int:    Fraunhofer diffraction integral.
        Fresnel_Diff_IntEx:     Fraunhofer diffraction integral expanded version.
        Fresnel_Diff_Int:       Fresnel diffraction integral.
        Fresnel_Diff_intgd:     Integrand of Fresnel diffraction integral.
        """
        super().__init__()
        self.class_type = class_type
        self.define_symbols()
        
        # File settings
        self.input_dir  = "input/optics"
        self.output_dir = "output/optics"
        
        
        class subformulary:
            """
            Sub formulary class.
            
            Define global symbols in the outer class.
            """
            def __init__(self):
                self.rect = Lambda(x, Piecewise( (1, abs(x)<=0.5), (0, True) ))
        self.subformulary = subformulary() 


#### --- CLASSES ---


#### abcd
        class abcd(branch):
            """
            Ghatak Chapter5. The Matrix Method in Paraxial Optics.
            
            D1 = u, D2 = v
            """
            global n1, n2, R1, R2       # Material specific parameters.
            global x1, x2, alpha1, alpha2, D1, D2    # Optic configuration parameters.
            global lambda1, lambda2
            n1, n2, R1, R2 = symbols('n_1, n_2, R_1, R_2')
            x1, x2, alpha1, alpha2, D1, D2 = symbols('x_1, x_2, alpha_1, alpha_2, D_1 D_2')
            lambda1, lambda2 = symbols('lambda_1, lambda_2')
            
            def __init__(self):
                super().__init__()
                self.name = "abcd Matrix Method"
                self.lambda1 = Eq(S('lambda_1'), n1*alpha1)
                self.lambda2 = Eq(S('lambda_2'), n2*alpha2)
                self.P       = Eq(S('P'), (n2-n1)/R1)
                self.T = lambda D1=D1, n1=n1: Eq(S('T'), 
                            UnevaluatedExpr(Matrix(((1,    0),
                                                    (D1/n1, 1)))))  # (21)
                self.R = lambda n1=n1, n2=n2, R1=R1: Eq(S('R'), 
                            UnevaluatedExpr(Matrix(((1, -(n2-n1)/R1),
                                                    (0,  1)))))     # (31)
                self.SM = Eq(S('SM'), 
                            UnevaluatedExpr(Matrix((( b, -a),
                                                    (-d,  c)))))
                self.single_spherical_refracting_surface = \
                    Eq(S('M'), self.T(D2,n2).rhs * self.R().rhs * self.T(-D1,n1).rhs) # (37)
                self.single_spherical_refracting_surface_matrix_eq = \
                    Eq(Matrix(((lambda2),
                               (x2))),
                       MatMul(Matrix(((1 + self.P.rhs*D1/n1, -self.P.rhs),
                               (0                   ,  1 - self.P.rhs*D2/n2 ))),
                       Matrix(((lambda1),
                                    (x1))))                    
                       ) # (40)
                self.single_spherical_refracting_surface_matrix_mag = Eq(S('m'), n1*D2/(n2*D1))
                # self.coaxial_optical_system = Eq(S('M'), ) # kaldik # (51)
                
                self.translation_matrix = self.T
                self.refraction_matrix  = self.R
                self.system_matrix      = self.SM
                
            @staticmethod
            def __doc__():
                return "Sub class with abcd  matrix method in paraxial optics formulas from Ajoy Ghatak's Optics"
        self.MatrixMethod = self.abcd = abcd() 
                
            
            

#### Fiber Bragg Grating
        class Fiber_Bragg_Grating(branch):
            """
            Fiber Bragg Grating formulas from Ajoy Ghatak's Optics (Appendix C)
            """
            global Gamma, lambda_B, lambda_0, lambda_, n0, Delta_n
            Gamma, lambda_B, lambda_0, lambda_, n0, Delta_n = symbols('Gamma lambda_B lambda_0 Lambda n_0 Delta_n')           
            
            def __init__(self):
                super().__init__()
                self.name = "Fiber Bragg Grating"
                
                # Fundamental parameters
                self.lambda_B = self.Bragg_wavelength = Eq(lambda_B, 2*lambda_*n0)
                self.kappa = self.coupling_coefficient = Eq(kappa, (pi*Delta_n)/lambda_0)
                self.Gamma = Eq(Gamma, 4*pi*n0*(1/lambda_0 - 1/lambda_B))
                self.alpha = Eq(alpha, sqrt(kappa**2 - Gamma**2/4))
                # Ajoy Ghatak uses the word reflectivity instead of reflectance. Hecht uses reflectance.
                self.R = self.Reflectance = Eq( S('R'), kappa**2*sinh(alpha*L)**2 / (kappa**2*cosh(alpha*L)**2 - Gamma**2/4) ) # Ghatak2009 Appendix C Eq.3
                self.R_peak = Eq(S('R_peak'), tanh(pi*Delta_n*L/lambda_B)**2)
                    
            @staticmethod
            def __doc__():
                return "Sub class with Fiber Bragg Grating formulas from Ajoy Ghatak's Optics"
        self.Fiber_Bragg_Grating = self.FBG = Fiber_Bragg_Grating()        


#### Fraunhofer Diffraction
        class Fraunhofer(branch):
            """
            Sub class for Fraunhofer Diffraction
            
            x0: x coordinate at the aperture at z=0.
            y0: y coordinate at the aperture at z=0.
            
            Reference: Degiorgio, Photonics a Short Course. (Eq.1.61)
            """
            def __init__(self):
                super().__init__()
                self.name = "Fraunhofer Diffraction"
                self.class_type = "Fraunhofer"
                self.integrand  = Uapr*Eli*exp(-I*2*pi/(l*z)*(x*x0+y*y0))
                self.integral   = Eq(_El, ( exp(I*k*z)/(I*l*z))*(exp(I*k/(2*z)*(x**2 + y**2)))*Integral(self.integrand, (y0, y0min, y0max), (x0,x0min,x0max)) )
                self.EField     = Eq(_El, self.integral.rhs)
                self.intensity  = Eq(_I, self.EField.rhs*conjugate(self.EField.rhs))
                # self.intensity= Eq(_I, abs(self.EField.rhs)**2)
        self.Fraunhofer = Fraunhofer()


#### Fresnel Diffraction
        class Fresnel(branch):
            """
            Sub class for Fresnel Diffraction
            """
            def __init__(self):
                super().__init__()
                self.name = "Fresnel Diffraction"
                self.class_type = "Fresnel"
                self.integrand  = Uapr*Eli*exp( I*k/(2*z)*((x-x0)**2+(y-y0)**2) )
                self.integral   = Eq(_El, (exp(I*k*z)/(I*l*z))*Integral(self.integrand, (y0, y0min, y0max), (x0,x0min,x0max)) )
                # Fresnel Diffraction Expanded Integral
                self.Fresnel_Diff_IntEx  = Eq(_El, ( exp(I*k*z)/(I*l*z))*(exp(I*k/(2*z)*(x**2 + y**2)))*Integral(Uapr*Eli*exp( I*k/(2*z)*(x0**2+y0**2))*exp(-I*2*pi/(l*z)*(x*x0+y*y0)), (y0, y0min, y0max), (x0,x0min,x0max)) )
                self.EField     = Eq(_El, self.integral.rhs)
                self.intensity  = Eq(_I, exp(I*k*z)/(I*l*z)*conjugate(exp(I*k*z)/(I*l*z))* \
                                     (Integral(re(self.integrand), (y0, y0min, y0max), (x0,x0min,x0max))**2+ \
                                      Integral(im(self.integrand), (y0, y0min, y0max), (x0,x0min,x0max))**2 ))
        self.Fresnel = Fresnel()
             

#### Rayleigh-Sommerfeld Diffraction Integral
        class Rayleigh_Sommerfeld(branch):
            """
            Sub class for Rayleigh-Sommerfeld Diffraction Integral.
            """
            def __init__(self):
                super().__init__()
                self.class_type = "Rayleigh_Sommerfeld"
                self.name = "Rayleigh-Sommerfeld Diffraction Integral"
                self.integrand  = Uapr*Eli*z*exp(I*k*sqrt((x-x0)**2+(y-y0)**2+z**2))/((x-x0)**2+(y-y0)**2+z**2)
                self.integral   = Eq(_El, 1/(I*l)*Integral(self.integrand, (y0, y0min, y0max), (x0,x0min,x0max)) )
                self.EField     = Eq(_El, self.integral.rhs)
                self.intensity  = Eq(_I, 1/(I*l)*conjugate(1/(I*l))* \
                                    (Integral(re(self.integrand), (y0, y0min, y0max), (x0,x0min,x0max))**2+ \
                                     Integral(im(self.integrand), (y0, y0min, y0max), (x0,x0min,x0max))**2 ))
        self.Rayleigh_Sommerfeld = Rayleigh_Sommerfeld()

        
    @staticmethod
    def __doc__():
        return("Document of optics class.")
        
oopti = optics()
