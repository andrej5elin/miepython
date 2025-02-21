"""
Mie scattering calculations for perfect spheres.

Extensive documentation is at <https://miepython.readthedocs.io>

`miepython` is a pure Python module to calculate light scattering of
a plane wave by non-absorbing, partially-absorbing, or perfectly conducting
spheres.

The extinction efficiency, scattering efficiency, backscattering, and
scattering asymmetry for a sphere with complex index of refraction m,
diameter d, and wavelength lambda can be found by::

    import miepython as mie
    qext, qsca, qback, g = mie.ez_mie(m, d, lambda0)

The normalized scattering values for angles mu=cos(theta) are::

    import miepython as mie
    Ipar, Iper = mie.ez_intensities(m, d, lambda0, mu)

If the size parameter is known, then use::

    import miepython as mie
    mie.efficiencies(m, x)

Mie scattering amplitudes S1 and S2 (complex numbers):

    mie.S1_S2(m, x, mu)

Normalized Mie scattering intensities for angles mu=cos(theta)::

    mie.i_per(m, x, mu)
    mie.i_par(m, x, mu)
    mie.i_unpolarized(m, x, mu)
"""
import numpy as np
from miepython import _an_bn, _cn_dn, _S1_S2, _D_calc, _mie


# not really needed but included for clarity
__all__ = (
    "intensities",
    "i_par",
    "i_per",
    "i_unpolarized",
    "efficiencies_mx",
    "efficiencies",
    "S1_S2",
    "phase_matrix",
    "coefficients",
    "an_bn",
    "cn_dn",
    "_D_calc",
)


def an_bn(m, x, n_pole=0):
    """
    Compute arrays of Mie coefficients A and B for a sphere.

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
    if np.imag(m) > 0:  # ensure imaginary part of refractive index is negative
        m = np.conj(m)

    a, b = _an_bn(np.complex128(m), np.float64(x), np.int64(n_pole))

    if n_pole > 0:
        a = a[-1]
        b = b[-1]

    return a, b


def cn_dn(m, x, n_pole=0):
    """
    Calculate Mie coefficients c_n and d_n for the internal field of a sphere.

    Args:
        m (complex): Refractive index of the sphere relative to the surrounding medium.
        x (float): Size parameter of the sphere (2πr/λ).
        n_pole (int): Number of terms to calculate (n_pole).

    Returns:
        (np.ndarray, np.ndarray): Arrays of c_n and d_n coefficients.
    """
    if np.imag(m) > 0:  # ensure imaginary part of refractive index is negative
        m = np.conj(m)

    c, d = _cn_dn(np.complex128(m), np.float64(x), np.int64(n_pole))

    if n_pole > 0:
        c = c[-1]
        d = d[-1]

    return c, d


def coefficients(m, x, n_pole=0, internal=False):
    """
    Computes the Mie coefficients for a sphere.

    This function calculates the Mie coefficients for electromagnetic scattering
    by a sphere. It supports both single values and arrays for the refractive
    index and size parameter. For arrays, the lengths of `m` and `x` must match.
    If the length of `m` or `x` is zero, the function assumes scalar values.

    If `n_pole > 0`, the function returns only the 1 to nth multipole coefficients
    for each input.

    Args:
        m (complex or array-like): The complex refractive index of the sphere.
            If an array, must match the length of `x`.
        x (float or array-like): The size parameter of the sphere. If an array,
            must match the length of `m`.
        n_pole (int, optional): The specific multipole order to compute. Defaults to 0,
            which calculates all terms and returns the full arrays of coefficients.
        internal (bool, optional): If True then returns Mie coefficients needed to
            calculate fields inside the sphere.

    Returns:
        tuple:
            - a (complex or array-like): The computed Mie coefficient A_n or an array of A_n values.
            - b (complex or array-like): The computed Mie coefficient B_n or an array of B_n values.

    Notes:
        - If the imaginary part of the refractive index is positive, it is
          automatically corrected to its conjugate value to ensure a valid input.

    Examples:
        Compute coefficients for a single sphere:
        >>> m = 1.5 - 0.1j
        >>> x = 0.1
        >>> mie_coefficients(m, x)
        (array([3.33370015e-05+1.97363100e-04j, 1.77504613e-08+1.11834113e-07j,
        4.60529820e-12+2.97368373e-11j, 1.58460985e-28+1.25881287e-14j]),
        array([6.67296256e-08+2.75469444e-07j, 1.90474848e-11+7.86835968e-11j,
        3.02615454e-15+1.24937186e-14j, 1.58460985e-28+1.25881287e-14j]))

        Compute first multipoles for multiple spheres:
        >>> m = [1.5 + 0.1j, 1.4 + 0.05j]
        >>> x = [2.0, 1.8]
        >>> mie_coefficients(m, x, 1)
        (array([0.44794644+0.38665192j, 0.27813728+0.38495241j]),
        array([0.5621083 +0.25504616j, 0.20206531+0.29589842j]))
    """
    mlen = np.size(m) if np.ndim(m) > 0 else 0
    xlen = np.size(x) if np.ndim(x) > 0 else 0

    if mlen == 0 and xlen == 0:
        a, b = an_bn(m, x, n_pole)
        if internal:
            c, d = cn_dn(m, x, n_pole)
    else:
        if xlen > 0 and mlen > 0 and xlen != mlen:
            raise RuntimeError("m and x arrays must be same length")

        if n_pole == 0:
            raise RuntimeError("when m or x is an array and n_pole must be >0")

        thelen = max(xlen, mlen)
        a = np.zeros(thelen, dtype=np.complex128)
        b = np.zeros(thelen, dtype=np.complex128)
        if internal:
            c = np.zeros(thelen, dtype=np.complex128)
            d = np.zeros(thelen, dtype=np.complex128)

        for i in range(thelen):
            mm = m[i] if mlen > 0 else m
            xx = x[i] if xlen > 0 else x

            a[i], b[i] = an_bn(mm, xx, n_pole)
            if internal:
                c[i], d[i] = cn_dn(mm, xx, n_pole)

    if internal:
        return a, b, c, d
    return a, b


def efficiencies_mx(m, x, n_pole=0, field="Electric"):
    """
    Computes scattering and extinction efficiencies for a spherical particle using Mie theory.

    Supports array inputs for wavelength-dependent calculations of the refractive index
    and size parameter.

    Args:
        m (complex or array-like):
            Complex refractive index of the sphere, defined as m = n - ik,
            where n is the real part (phase velocity) and k is the
            imaginary part (absorption). Can be a scalar or an array.
        x (float or array-like):
            Size parameter of the sphere, defined as x = π d / λ,
            where d is the diameter of the sphere and λ is the
            wavelength in the surrounding medium. Can be a scalar or an array.
        n_pole (int):
            Multipole order to compute:
            - If 0 (default), computes contributions from all multipoles.
            - If non-zero, computes contributions from the specified multipole order.
        field (str):
            If "Electric" (default) If n_pole>0, then True value returns the electric
            multipole contribution otherwise the Magnetic field contribution
    Returns:
        tuple:
            qext (float or array-like):
                Extinction efficiency, representing the total attenuation of light
                (scattering plus absorption) by the particle.
            qsca (float or array-like):
                Scattering efficiency, representing the fraction of incident light
                scattered by the particle.
            qback (float or array-like):
                Backscattering efficiency, describing the fraction of incident light
                scattered in the exact backward direction (theta = 180 degrees).
            g (float or array-like):
                Asymmetry parameter, representing the average cosine of the scattering
                angle over all angles.

    Notes:
        - Ensure m and x have compatible dimensions if passed as arrays.
        - For accurate results, n_pole should be within the range of significant
          multipole contributions, typically n ~ x + 4*x**(1/3).
        - This implementation assumes spherical, homogeneous particles in a
          non-absorbing medium.

    Examples:
        Compute efficiencies for a sphere with fixed parameters:

        >>> import miepython as mie
        >>> mie.efficiencies(1.5 - 0.01j, 2.0, 0)
        (qext: 1.81, qsca: 1.72, qback: 0.27, g: 0.63)

        Compute efficiencies for wavelength-dependent refractive indices:

        >>> import miepython as mie
        >>> m_array = np.array([1.5 - 0.01j, 1.45 - 0.02j])
        >>> x_array = np.array([2.0, 2.5])
        >>> mie.efficiencies(m_array, x_array, 0)
        (qext: [1.81, 2.15], qsca: [1.72, 1.95], qback: [0.27, 0.31], g: [0.63, 0.70])
    """
    # ensure imaginary part of refractive index is negative
    if np.isscalar(m):
        m = np.conj(m) if np.imag(m) > 0 else m
    else:
        m = np.where(np.imag(m) > 0, np.conj(m), m)
    
    if field == "Electric":
        with np.errstate(all='ignore'): 
            qext, qsca, qback, g = _mie(m, x, n_pole,1)
    else:
        with np.errstate(all='ignore'): 
            qext, qsca, qback, g = _mie(m, x, n_pole,0)
    
    return qext, qsca, qback, g



def normalization_factor(m, x, norm_str):
    """
    Figure out scattering function normalization.

    Args:
        m: complex index of refraction of sphere
        x: dimensionless sphere size
        norm_str: string describing type of normalization

    Returns:
        scaling factor needed for scattering function
    """
    factor = None
    norm = norm_str.lower()

    if norm in ["bohren"]:
        factor = 1 / 2

    elif norm in ["wiscombe"]:
        factor = 1

    elif norm in ["qsca", "scattering_efficiency"]:
        factor = x * np.sqrt(np.pi)

    else:
        with np.errstate(all='ignore'): 
            qext, qsca, _, _ = _mie(m, x, 0, 1)

        if norm in ["a", "albedo"]:
            factor = x * np.sqrt(np.pi * qext)

        if norm in ["1", "one", "unity"]:
            factor = x * np.sqrt(qsca * np.pi)

        if norm in ["four_pi", "4pi"]:
            factor = x * np.sqrt(qsca / 4)

        if norm in ["qext", "extinction_efficiency"]:
            factor = x * np.sqrt(qsca * np.pi / qext)

    if factor is None:
        raise ValueError(
            "normalization must be one of 'albedo' (default), 'one'"
            "'4pi', 'qext', 'qsca', 'bohren', or 'wiscombe'"
        )
    return factor


def S1_S2(m, x, mu, norm="albedo", n_pole=0 , out = None):
    """
    Calculate the scattering amplitude functions for spheres.

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate scattering amplitudes
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means include all terms)
        out : the output arrays tuple (optional).

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
        
        
    Notes:
        This is a generalized universal function. 
        Numpy broadcasting rules apply over the first three positional 
        arguments with the following signature: '(),(),(n)->(n),(n)'
    """
    # out must be a tuple of length 2 or None.
    S1, S2 = (None,None) if out is None else out
    

    # make sure m, mu and x are arrays (even if they are scalars)
    mu = np.asarray(mu)
    x = np.asarray(x)
    m = np.asarray(m)
    n_pole = np.asarray(n_pole)
    
    # ensure imaginary part of the refractive index is negative
    if np.isscalar(m):
        m = np.conj(m) if np.imag(m) > 0 else m
    else:
        m = np.where(np.imag(m) > 0, np.conj(m), m)

    # normalization is part of S1,S2 calculation, so we compute it first
    normalization = normalization_factor(m, x, norm)
    
    # enforce 1D
    if mu.ndim == 0:
        mu = mu.reshape((1,)) 
        
    # numba has issues dealing with vectorize and _S1_S2 prints RuntimeWarnings.
    # this is an expected behavior, see:   
    # https://numba.pydata.org/numba-doc/dev/reference/fpsemantics.html
    
    # When calling a ufunc created with vectorize(), Numpy will determine 
    # whether an error occurred by examining the FPU error word. It may 
    # then print out a warning or raise an exception (such as RuntimeWarning: 
    # divide by zero encountered), depending on the current error handling settings.
    # Depending on how LLVM optimized the ufunc’s code, however, some 
    # spurious warnings or errors may appear. If you get caught by this issue, 
    # we recommend you call numpy.seterr() to change Numpy’s error handling settings, 
    # or the numpy.errstate context manager to switch them temporarily:
    
    # remove all warnings to deal with numba floating-point pitfalls
    with np.errstate(all='ignore'): 
        S1,S2 = _S1_S2(m, x, mu, n_pole, normalization, out = (S1,S2))
        
    return S1,S2
    

def phase_matrix(m, x, mu, norm="albedo", n_pole=0):
    """
    Calculate the scattering (Mueller) matrix.

    If mu has length N, then the returned matrix is 4x4xN.  If mu is a scalar
    then the matrix is 4x4

    The phase scattering matrix is computed from the scattering amplitude
    functions, according to equations 5.2.105-6 in K. N. Liou (**2002**) -
    *An Introduction to Atmospheric Radiation*, Second Edition.

    or

    Bohren and Huffman, *Absorption and Scattering of Light by Small Particles*,
    JOHN WILEY & SONS, page 112, (1983).

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), of the phase scattering matrix
        n_pole: return n_pole term from series (default=0 means include all terms)
        norm: (optional) string describing scattering function normalization

    Returns:
        p: the phase scattering matrix [sr**(-1.0)]
    """
    mu = np.atleast_1d(mu)
    s1, s2 = S1_S2(m=m, x=x, mu=mu, norm=norm, n_pole=n_pole)

    s1_star = np.conjugate(s1)
    s2_star = np.conjugate(s2)
    m1 = (s1 * s1_star).real
    m2 = (s2 * s2_star).real
    s21 = (0.5 * (s1 * s2_star + s2 * s1_star)).real
    d21 = (-0.5j * (s1 * s2_star - s2 * s1_star)).real
    phase = np.zeros(shape=(4, 4) + s1.shape)
    phase[0, 0] = 0.5 * (m2 + m1)
    phase[0, 1] = 0.5 * (m2 - m1)
    phase[1, 0] = phase[0, 1]
    phase[1, 1] = phase[0, 0]
    phase[2, 2] = s21
    phase[2, 3] = -d21
    phase[3, 2] = d21
    phase[3, 3] = s21

    return phase.squeeze()


def i_per(m, x, mu, norm="albedo", n_pole=0):
    """
    Return the scattered intensity in a plane normal to the incident light.

    This is the scattered intensity in a plane that is perpendicular to the
    field of the incident plane wave. The intensity is normalized such
    that the integral of the unpolarized intensity over 4π steradians
    is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate intensities
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means add all terms)

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, _ = S1_S2(m, x, mu, norm, n_pole)
    intensity = np.abs(s1) ** 2
    return intensity.astype("float")


def i_par(m, x, mu, norm="albedo", n_pole=0):
    """
    Return the scattered intensity in a plane parallel to the incident light.

    This is the scattered intensity in a plane that is parallel to the
    field of the incident plane wave. The intensity is normalized such
    that the integral of the unpolarized intensity over 4π steradians
    is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means add all terms)

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    _, s2 = S1_S2(m, x, mu, norm, n_pole)
    intensity = np.abs(s2) ** 2
    return intensity.astype("float")


def i_unpolarized(m, x, mu, norm="albedo", n_pole=0):
    """
    Return the unpolarized scattered intensity at specified angles.

    This is the average value for randomly polarized incident light.
    The intensity is normalized such
    that the integral of the unpolarized intensity over 4π steradians
    is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means add all terms)

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, s2 = S1_S2(m, x, mu, norm, n_pole)
    intensity = (np.abs(s1) ** 2 + np.abs(s2) ** 2) / 2
    return intensity.astype("float")


def efficiencies(m, d, lambda0, n_env=1.0):
    """
    Calculate the efficiencies of a sphere.

    Args:
        m: the complex index of refraction of the sphere [-]
        d: the diameter of the sphere                    [same units as lambda0]
        lambda0: wavelength in a vacuum                  [same units as d]
        n_env: real index of medium around sphere, optional.

    Returns:
        qext: the total extinction efficiency                  [-]
        qsca: the scattering efficiency                        [-]
        qback: the backscatter efficiency                      [-]
        g: the average cosine of the scattering phase function [-]
    """
    m_env = m / n_env
    x_env = np.pi * d / (lambda0 / n_env)
    return efficiencies_mx(m_env, x_env)


def intensities(m, d, lambda0, mu, n_env=1.0, norm="albedo", n_pole=0):
    """
    Return the scattered intensities from a sphere.

    These are the scattered intensities in a plane that is parallel (ipar) and
    perpendicular (iper) to the field of the incident plane wave.

    The scattered intensity is normalized such that the integral of the
    unpolarized intensity over 4𝜋 steradians is equal to the single scattering
    albedo.  The scattered intensity has units of inverse steradians [1/sr].

    The unpolarized scattering is the average of the two scattered intensities.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere [-]
        d: the diameter of the sphere                    [same units as lambda0]
        lambda0: wavelength in a vacuum                  [same units as d]
        mu: the cos(theta) of each direction desired     [-]
        n_env: real index of medium around sphere, optional.
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means include all terms)

    Returns:
        ipar, iper: scattered intensity in parallel and perpendicular planes [1/sr]
    """
    m_env = m / n_env
    lambda_env = lambda0 / n_env
    x_env = np.pi * d / lambda_env
    s1, s2 = S1_S2(m_env, x_env, mu, norm, n_pole)
    ipar = np.abs(s2) ** 2
    iper = np.abs(s1) ** 2
    Ipar = ipar.astype("float")
    Iper = iper.astype("float")
    return Ipar, Iper
