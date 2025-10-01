"""
Low-level Mie calculations (jitted or non-jitted)

Thew code here is develped and optimized for numba, however, an equivalent
python-only code is obtained by disabling the compilation of scalar
functions and replacing the numba vectorization with numpy's vectorization

Whether we use jitted versions is determind by the USE_JIT variable. 
To further tune the compilation process, we set NB_FASTMATH, and NB_TARGET options.
"""

import numpy as np
import os
import numba as nb


__all__ = (
    "_D_calc",
    "_an_bn",
    "_cn_dn",
    "_S1_S2",
    "_mie"
)

#: whether we use numba to compile the code
USE_JIT = os.environ.get("MIEPYTHON_USE_JIT", "1").lower() == "1"

#: whether to use fatsmath option in jitted finctions
NB_FASTMATH = os.environ.get("MIEPYTHON_FASTMATH", "1").lower() == "1"

#: numba target option for vectorize and guvectorize functions 
NB_TARGET = os.environ.get("MIEPYTHON_TARGET", "parallel").lower()

#:whether to cache compiled functions
NB_CACHE = os.environ.get("MIEPYTHON_CACHE", "1").lower() == "1"

#: whether we use double precision
USE_DOUBLE = os.environ.get("MIEPYTHON_USE_DOUBLE", "1").lower() == "1"

if USE_DOUBLE == True:
    np_complex = np.complex128
    np_float = np.float64
    nb_complex = nb.complex128
    nb_float = nb.float64
    nb_int = nb.int64
    
else:
    np_complex = np.complex64
    np_float = np.float32
    nb_complex = nb.complex64
    nb_float = nb.float32
    nb_int = nb.int64 # no need to complicate with int32
    
dt_float = np.dtype(np_float, align = True) 
dt_complex = np.dtype(np_complex, align = True)

def njit(*args,**kwargs):
    """Wrapper for numba's njit decorator. Based on the USE_JIT, we either
    return a numba jit decorator, or a do-nothing decorator"""
    if USE_JIT:
        # return a numba njit decorator
        return nb.njit(*args,**kwargs)
    else:
        #return a "do nothing" decorator
        def _njit(f):
            return f
        return _njit
    
#-----------------
# Scalar functions
#-----------------

# Scalar functions work with scalar arguments and return scalar values.
# An exception is the _S1_S2_scalar, which in addition to scalar argument, takes 
# an array for the angles argument, and returns arrays.

@njit((nb_complex, nb_int), cache=NB_CACHE, fastmath = NB_FASTMATH, error_model='numpy')
def _Lentz_Dn(z, N):
    """
    Compute the logarithmic derivative of the Ricatti-Bessel function.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function

    Returns:
        This returns the Ricatti-Bessel function of order N with argument z
        using the continued fraction technique of Lentz, Appl. Opt., 15,
        668-671, (1976).
    """
    Nf = np_float(N)
    
    two = np_float(2)
    one = np_float(1)
    
    zinv = two / z
    alpha = (Nf + np_float(0.5)) * zinv
    aj = -(Nf + np_float(1.5)) * zinv
    alpha_j1 = aj + one / alpha
    alpha_j2 = aj
    ratio = alpha_j1 / alpha_j2
    runratio = alpha * ratio

    while np.abs(np.abs(ratio) - one) > np_float(1e-12):
        aj = zinv - aj
        alpha_j1 = one / alpha_j1 + aj
        alpha_j2 = one / alpha_j2 + aj
        ratio = alpha_j1 / alpha_j2
        zinv *= -one
        runratio = ratio * runratio

    return -Nf / z + runratio


@njit((nb_complex, nb_int, nb_complex[:]), cache=NB_CACHE, fastmath = NB_FASTMATH, error_model='numpy')
def _D_downwards(z, N, D):
    """
    Compute the logarithmic derivative by downwards recurrence.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with ψ_k'(z)/ψ_k(z) for k=0 to N-1
    """
    one = np_float(1)
    last_D = _Lentz_Dn(z, N)
    for n in range(N, 0, -1):
        nf = np_float(n)
        nf_div_z = nf / z
        last_D = nf_div_z - one / (last_D + nf_div_z)
        D[n - 1] = last_D


@njit((nb_complex, nb_int, nb_complex[:]), cache=NB_CACHE, fastmath = NB_FASTMATH, error_model='numpy')
def _D_upwards(z, N, D):
    """
    Compute the logarithmic derivative by upwards recurrence.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with ψ_k'(z)/ψ_k(z) for k=0 to N-1
    """
    twoj = np_complex(2j)
    onej = np_complex(1j)
    one = np_float(1)
    
    exp = np.exp(-twoj * z)
    D[1] = - one / z + (one - exp) / ((one - exp) / z - onej * (one + exp))
    for n in range(2, N):
        nf = np_float(n)
        nf_div_z = nf / z
        D[n] = one / (nf_div_z - D[n - 1]) - nf_div_z
        

@njit((nb_complex, nb_float, nb_int), cache=NB_CACHE, fastmath = NB_FASTMATH, error_model='numpy')
def _D_calc(m, x, N):
    """
    Compute the logarithmic derivative of ψ_n(z) using the best method.

    D_n(z) = d[log ψ_n(z)] = ψ_n'(z)/ψ_n(z)

    here ψ_n(z) is the Riccati-Bessel function of the first kind ψ_n(z)=z*j_n(z)
    were j_n(z) is the spherical Bessel function of order n.

    The zero-based array, D[:], is shifted so that D[0] = D₁(z) = ψ₁'(z)/ψ₁(z)

    Args:
        m: the np_complex index of refraction of the sphere
        x: the size parameter of the sphere
        N: order of Ricatti-Bessel function

    Returns:
        Array of logarithmic derivatives D_k(z) for k=1 to N-1.
    """
    n = m.real
    kappa = np.abs(m.imag)
    D = np.zeros(N + 1, dtype = np_complex)
    mx = np_complex(m * x)  # ensure complex

    if n < 1 or n > 10 or kappa > 10 or x * kappa >= 3.9 - 10.8 * n + 13.78 * n**2:
       _D_downwards(mx, N, D)
    else:
       _D_upwards(mx, N, D)
    return D[1:]


@njit((nb_complex, nb_float, nb_int), cache=NB_CACHE, fastmath = NB_FASTMATH, error_model='numpy')
def _an_bn(m, x, n_pole):
    """
    Compute arrays of Mie coefficients a_n and b_n for a sphere.

    When n_pole=0, the routine estimates the size of the arrays based on Wiscombe's
    formula. The length of the arrays is chosen so that the error when the series
    is summed is around 1e-6.

    If n_pole>0, then the array sizes will be n_pole+1. This is useful when
    trying to isolate the behavior of a particular multipole.

    To support resonance calculations, one can specify the number of terms
    to be calculated.  In general, using too few or too many terms increases the
    error rate.  So if you specify the number of terms be aware that you are
    playing with fire.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        n_pole: the number of An and Bn terms (0 does autosizing)

    Returns:
        a, b: arrays of Mie coefficents An and Bn
    """

    
    two = np_float(2)
    one = np_float(1)
        
    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1
        
    a = np.zeros(nstop, dtype=np_complex)
    b = np.zeros(nstop, dtype=np_complex)
     
    sin_x = np.sin(x)
    cos_x = np.cos(x)

    psi_nm1 = sin_x  # nm1 = n-1 = 0
    psi_n = psi_nm1 / x - cos_x
    xi_nm1 = np_complex(psi_nm1 + 1j * cos_x)
    xi_n = np_complex(psi_n + 1j * (cos_x / x + sin_x))
    
    if m.real > 0.0:
        D = _D_calc(m, x, nstop + 1)

        for i in range(nstop-1):
            n = np_float(i + 1)
            two_times_n_plus_one = (two * n + one)
            n_div_x = n / x
            temp = D[i] / m + n_div_x
            a[i] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            temp = D[i] * m + n_div_x
            b[i] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            psi = two_times_n_plus_one * psi_n / x - psi_nm1
            xi =  two_times_n_plus_one * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = psi

    else:
        for n in range(1, nstop):
            a[n - 1] = (n * psi_n / x - psi_nm1) / (n * xi_n / x - xi_nm1)
            b[n - 1] = psi_n / xi_n
            xi = (two * n + one) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = xi_n.real

    if n_pole != 0:
        a = a[:-1]
        b = b[:-1]

    return np.conjugate(a), np.conjugate(b)


@njit((nb_complex, nb_float, nb_int), fastmath = NB_FASTMATH, error_model='numpy')
def _cn_dn(m, x, n_pole):
    """
    Calculate Mie coefficients c_n and d_n for the internal field of a sphere.

    Args:
        m (np_complex): Refractive index of the sphere relative to the surrounding medium.
        x (float): Size parameter of the sphere (2πr/λ).
        n_pole (int): Number of terms to calculate (n_pole).

    Returns:
        (np.ndarray, np.ndarray): Arrays of c_n and d_n coefficients.
    """
    # ensure imaginary part of refractive index is negative
    m = np.where(np.imag(m) > 0, np.conj(m), m)
    mx = m * x

    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1

    c = np.zeros(nstop, dtype=np_complex)
    d = np.zeros(nstop, dtype=np_complex)
    
    # so that we compute cos(x) and sin(x) only once
    sin_x = np.sin(x) 
    cos_x = np.cos(x)

    # no need to calculate anything when sphere is perfectly conducting
    if m.real > 0.0 and not np.isinf(m.real) or not np.isinf(m.imag):
        psi_nm1 = sin_x  # nm1 = n-1 = 0
        psi_n = psi_nm1 / x - cos_x

        psi_nm1_mx = np.sin(mx)  # nm1 = n-1 = 0
        psi_n_mx = psi_nm1_mx / mx - np.cos(mx)

        xi_nm1 = np_complex(psi_nm1 + 1j * cos_x)
        xi_n = np_complex(psi_n + 1j * (cos_x / x + sin_x))

        Dmx = _D_calc(np_complex(m), x, nstop + 1)
        Dx = _D_calc(np_complex(1), x, nstop + 1)

        for n in range(1, nstop + 1):
            common = (psi_n / psi_n_mx) * ((Dx[n - 1] + n / x) * xi_n - xi_nm1)

            c[n - 1] = m * common / ((m * Dmx[n - 1] + n / x) * xi_n - xi_nm1)
            d[n - 1] = common / ((Dmx[n - 1] / m + n / x) * xi_n - xi_nm1)

            psi = (2 * n + 1) * psi_n / x - psi_nm1
            psi_nm1 = psi_n
            psi_n = psi

            psi_mx = (2 * n + 1) * psi_n_mx / mx - psi_nm1_mx
            psi_nm1_mx = psi_n_mx
            psi_n_mx = psi_mx

            xi = (2 * n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi

    if n_pole != 0:
        c = c[:-1]
        d = d[:-1]
    return np.conjugate(c), np.conjugate(d)




@njit((nb_float, nb_float[::1], nb_float[::1]), cache=NB_CACHE, fastmath = NB_FASTMATH, error_model='numpy')
def _pi_tau(mu, pi, tau):
    """
    Compute the Mie scattering functions π_n and τ_n for given cosine angles.

    This function fills the pre-allocated arrays `pi` and `tau` with values
    of the Mie scattering functions for a given `mu = cos𝜃`. The function
    uses the recurrence relations for the associated Legendre polynomials
    of the first kind P_n^1. The recurrence relations ensure numerical stability
    and avoids calling scipi.special.lpmv(1, n, cos𝜃) for each n.

    `pi` and `tau` are **zero-based** arrays and therefore

    `pi[n-1]` = 𝜋_n(cos𝜃) = P_n^1(cos𝜃) / sin𝜃

    `tau[n-1]` = 𝜏_n(cos𝜃) = d/d𝜃 P_n^1(cos𝜃)`.

    Args:
        mu (float): The cosine of the scattering angle, `cos(𝜃)`.
        pi (numpy.ndarray): A pre-allocated array to store `pi_n` values.
        tau (numpy.ndarray): A pre-allocated array to store `tau_n` values.

    Returns:
        nothing.  pi and tau are modified
    """
    zero = np_float(0.)
    one = np_float(1.)
    two = np_float(2.)
    
    n_terms = len(pi)
    pi_nm2 = zero
    pi[0] = one
    
    # for i in range(n_terms-1):
    #     n = np_float(i + 1)
    #     tau[i] = n * mu * pi[i] - (n + one) * pi_nm2
    #     temp = pi[i]
    #     pi[i+1] = ((two * n + one) * mu * temp - (n + one) * pi_nm2) / n
    #     pi_nm2 = temp
        
    for i in range(n_terms-1):
        n = np_float(i + 1)
        two_n  = two * n 
        n_plus_one = n + one
        two_n_plus_one = two_n + one
        
        n_plus_one_times_pi_nm2 = n_plus_one * pi_nm2
        
        tau[i] = n * mu * pi[i] - n_plus_one_times_pi_nm2
        temp = pi[i]
        pi[i+1] = (two_n_plus_one * mu * temp - n_plus_one_times_pi_nm2) / n
        pi_nm2 = temp
        
        
# the _S1_S2_scalar is optimized for speed. Therefore, we also
# define the output arrays (S1,S2) to simplify vectorization and improve memory
# handling. For performance reasons, we make normalization a part of computiation
# This function is not meant to be used directly, instead, one uses a vectorized 
# version _S1_S2 instead.
        
@njit((nb_complex, nb_float, nb_float[:], nb_int, nb_float, nb_complex[:], nb_complex[:]), cache=NB_CACHE,
      fastmath = NB_FASTMATH, error_model='numpy')
def _S1_S2_scalar(m, x, mu, n_pole, norm, S1, S2):
    """
    Calculate the scattering amplitude functions for spheres.

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: array of angles, cos(theta), to calculate scattering amplitudes
        norm_int: integer describing type of normalization
        n_pole: return n_pole term from series (default=0 means include all terms)
        

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    a, b = _an_bn(m, x, 0)
    
    one = np_float(1.)
    two = np_float(2.)

    nangles = len(mu)
    N = len(a)
    pi = np.zeros(N,dtype = dt_float)
    tau = np.zeros(N,dtype = dt_float)
    #scale = np.empty(N,dtype = nb_float)
    n = np.arange(1, N + 1, dtype = dt_float)
    scale = (two * n + one) / ((n + one) * n)
    nstop = len(a)

    for k in range(nangles):
        _pi_tau(mu[k], pi, tau)
        
        # we will perform:
        # s1 = np.sum(scale * (pi * a + tau * b))
        # s2 = np.sum(scale * (tau * a + pi * b))
        # but, allowig numba to perform code optimization.
        
        # we do not want to write to S1 and S2 directly   
        s1 = np_complex(0.) # temporary data to improve memory handling
        s2 = np_complex(0.) # temporary data to improve memory handling  
            
        for i in range(nstop-1):
            
            # first compute scaling factors (float)
            pi_scale = scale[i] * pi[i]
            tau_scale = scale[i] * tau[i]
            
            # now compute amplitudes 
            s1 += pi_scale * a[i] # 
            s1 += tau_scale * b[i]
            s2 += pi_scale * b[i]
            s2 += tau_scale * a[i] 

        #: normalize and store results
        S1[k] = np.conjugate(s1)/norm
        S2[k] = np.conjugate(s2)/norm  


@njit((nb_complex, nb_float), cache=NB_CACHE,fastmath = NB_FASTMATH, error_model='numpy')
def _small_conducting_mie(_m, x):
    """
    Calculate the efficiencies for a small conducting spheres.

    Typically used for small conducting spheres where x < 0.1 and
    m.real == 0

    Args:
        _m: the complex index of refraction of the sphere (unused)
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    ahat1 = complex(0, 2.0 / 3.0 * (1 - 0.2 * x**2))
    ahat1 /= complex(1 - 0.5 * x**2, 2.0 / 3.0 * x**3)

    bhat1 = complex(0.0, (x**2 - 10.0) / 30.0)
    bhat1 /= complex(1 + 0.5 * x**2, -(x**3) / 3.0)
    ahat2 = complex(0.0, x**2 / 30.0)
    bhat2 = complex(0.0, -(x**2) / 45.0)

    qsca = x**4 * (
        6 * np.abs(ahat1) ** 2 + 6 * np.abs(bhat1) ** 2 + 10 * np.abs(ahat2) ** 2 + 10 * np.abs(bhat2) ** 2
    )
    qext = qsca
    g = ahat1.imag * (ahat2.imag + bhat1.imag)
    g += bhat2.imag * (5.0 / 9.0 * ahat2.imag + bhat1.imag)
    g += ahat1.real * bhat1.real
    g *= 6 * x**4 / qsca

    qback = 9 * x**4 * np.abs(ahat1 - bhat1 - 5 / 3 * (ahat2 - bhat2)) ** 2

    return qext, qsca, qback, g    

@njit((nb_complex, nb_float), cache=NB_CACHE,fastmath = NB_FASTMATH, error_model='numpy')
def _small_mie(m, x):
    """
    Calculate the efficiencies for a small sphere.

    Typically used for small spheres where x<0.1

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    m2 = m * m
    x2 = x * x

    D = m2 + 2 + (1 - 0.7 * m2) * x2
    D -= (8 * m**4 - 385 * m2 + 350) * x**4 / 1400.0
    D += 2j * (m2 - 1) * x**3 * (1 - 0.1 * x2) / 3
    ahat1 = 2j * (m2 - 1) / 3 * (1 - 0.1 * x2 + (4 * m2 + 5) * x**4 / 1400) / D

    bhat1 = 1j * x2 * (m2 - 1) / 45 * (1 + (2 * m2 - 5) / 70 * x2)
    bhat1 /= 1 - (2 * m2 - 5) / 30 * x2

    ahat2 = 1j * x2 * (m2 - 1) / 15 * (1 - x2 / 14)
    ahat2 /= 2 * m2 + 3 - (2 * m2 - 7) / 14 * x2

    T = np.abs(ahat1) ** 2 + np.abs(bhat1) ** 2 + 5 / 3 * np.abs(ahat2) ** 2
    temp = ahat2 + bhat1
    g = (ahat1 * temp.conjugate()).real / T

    qsca = 6 * x**4 * T

    if m.imag == 0:
        qext = qsca
    else:
        qext = 6 * x * (ahat1 + bhat1 + 5 * ahat2 / 3).real

    sback = 1.5 * x**3 * (ahat1 - bhat1 - 5 * ahat2 / 3)
    qback = 4 * np.abs(sback) ** 2 / x2

    return qext, qsca, qback, g


@njit(nb_float(nb_complex), cache=NB_CACHE, fastmath = NB_FASTMATH)
def abs2(a):
    return a.real**2 + a.imag**2

@njit(nb_float(nb_complex, nb_complex), cache=NB_CACHE, fastmath = NB_FASTMATH)
def prod(a,b):
    return a.real * b.real + a.imag * b.imag 
    
@njit((nb_complex, nb_float, nb_int, nb_int), cache=NB_CACHE, fastmath = NB_FASTMATH, error_model='numpy')
def _mie_scalar(m, x, n_pole, e_field):
    """
    Calculate the efficiencies for a sphere when both m and x are scalars.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        n_pole: a non-zero value returns the contribution by the n_pole multipole
        e_field: Electric (True) or Magnetic Field otherwise

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """

    # case when sphere matches its environment
    if abs(m.real - 1) <= 1e-8 and abs(m.imag) < 1e-8:
        return 0., 0., 0., 0.

    # small conducting spheres --- see Wiscombe
    if m.real == 0 and x < 0.1 and n_pole == 0:
        return _small_conducting_mie(m, x)

    if m.real > 0.0 and np.abs(m) * x < 0.1 and n_pole == 0:
        return _small_mie(m, x)

    # sometimes m=0 is used to signal perfectly conducting sphere
    if abs(m.real) < 1e-8 and abs(m.imag) < 1e-8:
        m = 1 - 10000j
            
    a, b = _an_bn(m, x, n_pole)
    
    two = np_float(2.0)
    one = np_float(1.0)

    if n_pole == 0:
        n = np.arange(1, len(a) + 1, dtype = np_float)
        cn = two * n + one
        c1n = n * (n + two) / (n + one)
        c2n = cn / n / (n + one)
        
        qext = np_float(0.)
        qsca = np_float(0.)
        _qback = np_complex(0.)
        asy1 = np_float(0.)
        asy2 = np_float(0.)
        
        x2 = x**2
        x2inv = 1/x2
        
        for i in range(len(a)):
            ai = a[i]
            bi = b[i]
            cni = cn[i]
            
            qext += cni * (ai.real + bi.real)
            if m.imag == 0:
                qsca = qext
            else:
                qsca += cni * (abs2(ai) + abs2(bi))
            _qback += (-1) ** n[i] * cni * (ai - bi)
            if i != len(a)-1:
                asy1 += c1n[i] * (prod(ai, a[i+1]) + prod(bi, b[i+1]))
                asy2 += c2n[i] * prod(ai, bi)
                
        qext = two * qext * x2inv
        qsca = two * qsca * x2inv
        qback = abs2(_qback) * x2inv
        g = np_float(4) * (asy1 + asy2) / qsca * x2inv
            
        #qext = two * np.sum(cn * (a.real + b.real)) / x**2

        #if m.imag == 0:
        #    qsca = qext
        #else:
        #    qsca = two * np.sum(cn * (np.abs(a) ** 2 + np.abs(b) ** 2)) / x**2

        #qback = np.abs(np.sum((-1) ** n * cn * (a - b))) ** 2 / x**2

        #c1n = n * (n + two) / (n + one)
        #c2n = cn / n / (n + one)
        #asy1 = c1n[:-1] * (a[:-1] * a[1:].conjugate() + b[:-1] * b[1:].conjugate()).real
        #asy2 = c2n[:-1] * (a[:-1] * b[:-1].conjugate()).real
        #g = np_float(4) * np.sum(asy1 + asy2) / qsca / x**2
    else:
        a = a[-1]
        b = b[-1]
        cn = two * n_pole + one
        c1n = n_pole * (n_pole + two) / (n_pole + one)
        if e_field == 1:
            qext = two * cn * a.real / x**2
            qsca = two * cn * np.abs(a) ** 2 / x**2
            qback = qsca / two
            g = 0.
        else:
            qext = two * cn * b.real / x**2
            qsca = two * cn * np.abs(b) ** 2 / x**2
            qback = qsca / two
            g = 0.
            
    return qext, qsca, qback, g

@njit((nb_complex, nb_float, nb_int, nb_int), cache=NB_CACHE, fastmath = NB_FASTMATH)
def _mie_scalar2(m, x, n_pole, e_field):
    """
    Calculate the efficiencies for a sphere when both m and x are scalars.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        n_pole: a non-zero value returns the contribution by the n_pole multipole
        e_field: Electric (True) or Magnetic Field otherwise

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    # case when sphere matches its environment
    if abs(m.real - 1) <= 1e-8 and abs(m.imag) < 1e-8:
        return 0., 0., 0., 0.

    # small conducting spheres --- see Wiscombe
    if m.real == 0 and x < 0.1 and n_pole == 0:
        return _small_conducting_mie(m, x)

    if m.real > 0.0 and np.abs(m) * x < 0.1 and n_pole == 0:
        return _small_mie(m, x)

    # sometimes m=0 is used to signal perfectly conducting sphere
    if abs(m.real) < 1e-8 and abs(m.imag) < 1e-8:
        m = 1 - 10000j

    a, b = _an_bn(m, x, n_pole)
    
    two = np_float(2.0)
    one = np_float(1.0)


    if n_pole == 0:
        n = np.arange(1, len(a) + 1, dtype = np_float)
        cn = two * n + one
        c1n = n * (n + two) / (n + one)
        c2n = cn / n / (n + one)
            
        qext = two * np.sum(cn * (a.real + b.real)) / x**2

        if m.imag == 0:
            qsca = qext
        else:
            qsca = two * np.sum(cn * (np.abs(a) ** 2 + np.abs(b) ** 2)) / x**2

        qback = np.abs(np.sum((-1) ** n * cn * (a - b))) ** 2 / x**2

        asy1 = c1n[:-1] * (a[:-1] * a[1:].conjugate() + b[:-1] * b[1:].conjugate()).real
        asy2 = c2n[:-1] * (a[:-1] * b[:-1].conjugate()).real
        g = np_float(4) * np.sum(asy1 + asy2) / qsca / x**2
    else:
        a = a[-1]
        b = b[-1]
        cn = two * n_pole + one
        c1n = n_pole * (n_pole + two) / (n_pole + one)
        if e_field == 1:
            qext = two * cn * a.real / x**2
            qsca = two * cn * np.abs(a) ** 2 / x**2
            qback = qsca / two
            g = 0.
        else:
            qext = two * cn * b.real / x**2
            qsca = two * cn * np.abs(b) ** 2 / x**2
            qback = qsca / two
            g = 0.
            
    return qext, qsca, qback, g

#---------------------
# Vectorized functions
#---------------------

   
# In jittted version, we use numba to automatically vectorize the scalar functions
# In case we skip numba, we rely on numpy's vectorize implementation.
# Note that numpy's vectorization is meant for reference and not for speed. 
# The resulting arrays of both implementations are identical in shape and content, 
# but numpy's version does not allow us to specify output arrays. We deal with 
# the implementation difference in the high-level functions in the core module

if USE_JIT:
    # Vectrorize using numba's guvectorize 
    
    @nb.guvectorize([(nb_complex[:], nb_float[:], nb_float[:], nb_int[:], nb_float[:], nb_complex[:], nb_complex[:])],
                 "(),(),(n),(),()->(n),(n)", cache=NB_CACHE, target = NB_TARGET, fastmath = NB_FASTMATH)
    def _S1_S2(m, x, mu, n_pole, normalization, S1, S2):
        """guvectorize version of _S1_S2_scalar"""
        _S1_S2_scalar(m[0], x[0], mu, n_pole[0],normalization[0], S1,S2) 

            
    @nb.guvectorize([(nb_complex[:], nb_float[:],  nb_int[:],  nb_int[:], nb_float[:], nb_float[:], nb_float[:],nb_float[:])],
                 "(),(),(),()->(),(),(),()", cache=NB_CACHE, target = NB_TARGET, fastmath = NB_FASTMATH)
    def _mie(m, x, n_pole, e_field, qext, qsca,qback,g):
        """Vectorized version of _mie_scalar"""
        out = _mie_scalar(m[0],x[0],n_pole[0],e_field[0])
        qext[0] = out[0]
        qsca[0] = out[1]
        qback[0] = out[2]
        g[0] = out[3]
else:
    # Vectorize using numpy's vectorize
    
    @np.vectorize(signature="(),(),(n),(),()->(n),(n)")
    def _S1_S2(m, x, mu, n_pole, normalization):
        S1 = np.empty((len(mu),), np_complex)
        S2 = np.empty((len(mu),), np_complex)
        _S1_S2_scalar(m, x, mu, n_pole, normalization, S1, S2)
        return S1, S2

    @np.vectorize(signature = "(),(),(),()->(),(),(),()")
    def _mie(m, x, n_pole, e_field):
        """Vectorized version of _mie_scalar"""
        return _mie_scalar(m,x,n_pole,e_field)
