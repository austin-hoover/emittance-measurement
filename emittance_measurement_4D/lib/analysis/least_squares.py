"""Linear least squares solving using JAMA matrices."""
import copy
from math import sqrt
from Jama import Matrix


def sign(x):
    """Return the sign of x (+1, 0, -1)."""
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1
    
    
def transpose_list(b):
    """Convert list b to list of 1-element lists."""
    return [[elem] for elem in b]

    
def zeros(n, dtype=None):
    """Create n element column matrix filled with zeros."""
    return Matrix([[0.0] for _ in range(n)])


def norm(jama_matrix):
    """Return sum of squared elements of JAMA matrix."""
    return jama_matrix.normF()
    
    
def _sym_ortho(a, b):
    """Stable implementation of Givens rotation. 
        
    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).
    
    This code and documentation was copied from SciPy.
    
    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0:
        return sign(a), 0, abs(a)
    elif a == 0:
        return 0, sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r
 

def lsmr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, maxiter=None, 
         show=False, x0=None):
    """Iterative solver for least-squares problems.

    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    ``A`` is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. ``b`` is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).
    
    Parameters
    ----------
    A : JAMA matrix
        Matrix A in the linear system.
    b : JAMA matrix, shape (m,)
        Vector ``b`` in the linear system.
    damp : float
        Damping factor for regularized least-squares. `lsmr` solves
        the regularized least-squares problem::

         min ||(b) - (  A   )x||
             ||(0)   (damp*I) ||_2

        where damp is a scalar.  If damp is None or 0, the system
        is solved without regularization.
    atol, btol : float, optional
        Stopping tolerances. `lsmr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol.  Let ``r = b - Ax`` be the
        residual vector for the current approximate solution ``x``.
        If ``Ax = b`` seems to be consistent, ``lsmr`` terminates
        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
        Otherwise, lsmr terminates when ``norm(A^H r) <=
        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (say),
        the final ``norm(r)`` should be accurate to about 6
        digits. (The final ``x`` will usually have fewer correct digits,
        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
        or `btol` is None, a default value of 1.0e-6 will be used.
        Ideally, they should be estimates of the relative error in the
        entries of ``A`` and ``b`` respectively.  For example, if the entries
        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents
        the algorithm from doing unnecessary work beyond the
        uncertainty of the input data.
    conlim : float, optional
        `lsmr` terminates if an estimate of ``cond(A)`` exceeds
        `conlim`.  For compatible systems ``Ax = b``, conlim could be
        as large as 1.0e+12 (say).  For least-squares problems,
        `conlim` should be less than 1.0e+8. If `conlim` is None, the
        default value is 1e+8.  Maximum precision can be obtained by
        setting ``atol = btol = conlim = 0``, but the number of
        iterations may then be excessive.
    maxiter : int, optional
        `lsmr` terminates if the number of iterations reaches
        `maxiter`.  The default is ``maxiter = min(m, n)``.  For
        ill-conditioned systems, a larger value of `maxiter` may be
        needed.
    show : bool, optional
        Print iterations logs if ``show=True``.
        
    Returns
    -------
    x : ndarray of float
        Least-square solution returned.
    istop : int
        istop gives the reason for stopping::

          istop   = 0 means x=0 is a solution.  If x0 was given, then x=x0 is a
                      solution.
                  = 1 means x is an approximate solution to A*x = B,
                      according to atol and btol.
                  = 2 means x approximately solves the least-squares problem
                      according to atol.
                  = 3 means COND(A) seems to be greater than CONLIM.
                  = 4 is the same as 1 with atol = btol = eps (machine
                      precision)
                  = 5 is the same as 2 with atol = eps.
                  = 6 is the same as 3 with CONLIM = 1/eps.
                  = 7 means ITN reached maxiter before the other stopping
                      conditions were satisfied.
    itn : int
        Number of iterations used.
    normr : float
        ``norm(b-Ax)``
    normar : float
        ``norm(A^H (b - Ax))``
    norma : float
        ``norm(A)``
    conda : float
        Condition number of A.
    normx : float
        ``norm(x)``
        
    Notes
    -----
    This code and documentation was adapted from the SciPy implementation.

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders,
           "LSMR: An iterative algorithm for sparse least-squares problems",
           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
           :arxiv:`1006.0758`
    .. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/
    """
    msg = (
        'The exact solution is x = 0, or x = x0, if x0 was given   ',
        'Ax - b is small enough, given atol, btol                  ',
        'The least-squares solution is good enough, given atol     ',
        'The estimate of cond(Abar) has exceeded conlim            ',
        'Ax - b is small enough for this machine                   ',
        'The least-squares solution is good enough for this machine',
        'Cond(Abar) seems to be too large for this machine         ',
        'The iteration limit has been reached                      ',
    )
    hdg1 = '   itn      x(1)       norm r    norm Ar'
    hdg2 = ' compatible   LS      norm A   cond A'
    pfreq = 20   # print frequency (for repeating the heading)
    pcount = 0   # print counter

    m, n = A.getRowDimension(), A.getColumnDimension()

    minDim = min([m, n]) # stores number of singular values

    if maxiter is None:
        maxiter = minDim

    if show:
        print ' '
        print 'LSMR            Least-squares solution of  Ax = b\n'
        print 'The matrix A has {} rows and {} columns'.format(n, m)
        print 'damp = {}'.format(damp)
        print 'atol = {}         conlim = {}'.format(atol, conlim)
        print 'btol = {}         maxiter = {}'.format(btol, maxiter)

    u = b
    normb = norm(b)

    x = zeros(n)
    beta = normb

    if beta > 0:
        u = u.times(1 / beta)
        v = A.transpose().times(u)
        alpha = norm(v)
    else:
        v = zeros(n)
        alpha = 0

    if alpha > 0:
        v = v.times(1 / alpha)

    # Initialize variables for 1st iteration.

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = copy.deepcopy(v)
    hbar = zeros(n)

    # Initialize variables for estimation of ||r||.

    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules, normb set earlier
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        if show:
            print msg[0]
        return x, istop, itn, normr, normar, normA, condA, normx

    if show:
        print ' '
        print hdg1, hdg2
        test1 = 1
        test2 = alpha / beta
        str1 = '{:6g} {:12.5e}'.format(itn, x.get(0, 0))
        str2 = ' {:10.3e} {:10.3e}'.format(normr, normar)
        str3 = '  {:8.1e} {:8.1e}'.format(test1, test2)
        print ''.join([str1, str2, str3])

    # Main iteration loop.
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  a*v   -  alpha*u,
        #        alpha*v  =  A'*u  -  beta*v.
        u = u.times(-alpha)
        u = u.plus(A.times(v))
        beta = norm(u)

        if beta > 0:
            u = u.times(1 / beta)
            v = v.times(-beta)
            v = v.plus(A.transpose().times(u))
            alpha = norm(v)
            if alpha > 0:
                v = v.times(1 / alpha)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s*alpha
        alphabar = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.
        hbar = hbar.times(-(thetabar * rho / (rhoold * rhobarold)))
        hbar = hbar.plus(h)
        x = x.plus(hbar.times(zeta / (rho * rhobar)))
        h = h.times(-(thetanew / rho))
        h = h.plus(v)

        # Estimate of ||r||.
        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}. 
        
        # betad = betad_{k-1} here.
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = norm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = infty
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.
        if show:
            if (n <= 40) or (itn <= 10) or (itn >= maxiter - 10) or \
               (itn % 10 == 0) or (test3 <= 1.1 * ctol) or \
               (test2 <= 1.1 * atol) or (test1 <= 1.1 * rtol) or \
               (istop != 0):

                if pcount >= pfreq:
                    pcount = 0
                    print ' '
                    print hdg1, hdg2
                pcount = pcount + 1
                str1 = '{:6g} {:12.5e}'.format(itn, x.get(0, 0))
                str2 = ' {:10.3e} {:10.3e}'.format(normr, normar)
                str3 = '  {:8.1e} {:8.1e}'.format(test1, test2)
                str4 = '  {:8.1e} {:8.1e}'.format(normA, condA)
                print ''.join([str1, str2, str3, str4])

        if istop > 0:
            break

    # Print the stopping condition.
    if show:
        print ' '
        print'LSMR finished'
        print msg[istop]
        str1 = 'istop = {}    normr ={:8.1e}'.format(istop, normr)
        str2 = '    normA ={:8.1e}    normAr ={:8.1e}'.format(normA, normar)
        str3 = 'itn = {}    condA ={:8.1e}'.format(itn, condA)
        str4 = '    normx = {:8.1e}'.format(normx)
        print str1, str2
        print str3, str4

    return x, istop, itn, normr, normar, normA, condA, normx
    
    
    
def lsq_linear(A, b, solver='exact', max_iter=None, lsmr_tol=1e-12, verbose=0):
    """Solve an unbounded linear least-squares problem.
    
    Given a m-by-n design matrix A and a target vector b with m elements,
    `lsq_linear` minimizes 0.5 * ||A x - b||**2.
    
    Notes
    -----
    This documentation was adapted from SciPy.
    
    Parameters
    ----------
    A : list, shape (m, n)
        Design matrix. 
    b : list, shape (m,)
        Target vector.
    solver : {'exact', 'lsmr'}
        Method of solving unbounded least-squares problem.
            * 'exact' : Solution by QR factorization.
            * 'lsmr' : Use iterative procedure which requires only matrix-
              vector product evaluations. 
    max_iter : None or int, optional
        Maximum number of iterations before termination.
    lsmr_tol : None, float or 'auto', optional
        Tolerance parameters 'atol' and 'btol' for lsmr.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:
            * 0 : work silently (default).
            * 1 : display a termination report.
            * 2 : display progress during iterations.
    """
    A = Matrix(A)
    if type(b[0]) is not list:
        b = transpose_list(b)
    b = Matrix(b)
    if solver == 'exact':
        x = A.solve(b)
    elif solver == 'lsmr':
        show = verbose > 1
        tol = lsmr_tol
        result = lsmr(A, b, atol=tol, btol=tol, maxiter=max_iter, show=show)
        x, istop, itn, normr, normar, normA, condA, normx = result
    else:
        raise ValueError("Method must be in {'exact', 'lsmr'}")
    x = [x.get(i, 0) for i in range(x.getRowDimension())]
    return x