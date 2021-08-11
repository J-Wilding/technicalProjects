from scipy import linalg as sla
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread

class SVD:
    """The Singular Value Decomposition makes it easy to construct low-rank approximations of
        matrices. Thus it is a basis for several data compression algorithms. In this lab we'll
        learn to compute the SVD and use it to do basic image compression.
    """
    
    def compact_svd(this, A, tol=1e-6):
        """Compute the truncated SVD of A.

        Parameters:
            A ((m,n) ndarray): The matrix (of rank r) to factor.
            tol (float): The tolerance for excluding singular values.

        Returns:
            ((m,r) ndarray): The orthonormal matrix U in the SVD.
            ((r,) ndarray): The singular values of A as a 1-D array.
            ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
        """
        # First calculate the eigenvalues and eigenvectors of A^H A with the scipy linalg tool.
        eigs, V = sla.eig(A.conj().T @ A)
        
        # Next get the singular values of A, which are the sqrts of the eigenvalues of A^H A.
        σ = np.sqrt(eigs)
        
        # Now sort the singular values from greatest to least (negating since np.argsort sorts
        # least to greatest), ans sort (with array broadcasting) the columns of V accordingly.
        sort = np.argsort(-σ)
        σ = σ[sort]
        V = V[:,sort]

        # This is the compact SVD so we'll drop the singular values and respective columns from
        # the end below the given tolerance. Since σ and V are sorted this is a simple matter.
        σ_1 = [element for element in σ if element > tol]
        V_1 = V[:, :len(σ_1)]
        U_1 = A @ V_1 / σ_1
        return U_1, σ_1, V_1.conj().T

    def test_compact_svd(this):
        # Let's start with A, as such, and get the SVD from our above algorithm.
        A = np.array([[2, 5, 4], [6, 3, 0], [6, 3, 0], [2, 5, 4]])
        # A = np.random.rand(4,3)
        U, σ, Vh = this.compact_svd(A)
        print("=============================================\nMy Compact SVD", 
              "=============================================", f"U: {U}", f"Σ: {np.diag(σ)}",
              f"V^H: {Vh}", "=============================================", sep ='\n')

        # Check that U * σ * V == A
        print(f"Does the math check out?\n=============================================\nU @ Σ @"
              f"V^H == A: {np.allclose(U @ np.diag(σ) @ Vh, A)}")

        # U must be orthonormal.
        print(f"U is orthonormal (U.T @ U == I): {np.allclose(U.T @ U, np.identity(len(σ)))}")

        # Verify that the rank is correct.
        print(f"Rank is correct (Rank(A) == len(σ)): {np.linalg.matrix_rank(A) == len(σ)}")

        # Lastly check our compact SVD Algorithm against the Scipy Linalg SVD.
        U_sla, Σ_sla, Vh_sla = sla.svd(A);
        print("=============================================\nScipy Linalg Compact SVD",
              "=============================================", f"U: {U_sla}", f"Σ: {Σ_sla}",
              f"V^H: {Vh_sla}", "=============================================", sep ='\n')

    def visualize_svd(this, A):
        """ The linear transformation defined by m x n matrix A sends points from \R^n -> \R^m. So
            the SVD cleanly decomposes A into two rotations and a scaling, so, any linear
            transformation can be easily described geometrically (where V^H represents a rotation and
            Σ, a rescaling, and U, another rotation.

            Plot the effect of the SVD of A as a sequence of linear transformations on the unit
            circle and the two standard basis vectors.
        """

        # First generate a 2x200 matrix S representing a set of 200 pts on the unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        S = np.array([np.cos(theta), np.sin(theta)])
        # E represents the standard basis vectors in \R^2. We use three cols to match V^H.
        E = np.array([[1, 0, 0], [0, 0, 1]])
        # Then get the SVD-breakdown of matrix A.
        U, Sigma, Vh = sla.svd(A)

        # Plot (a) S
        ax1 = plt.subplot(221)
        ax1.plot(S[0,:], S[1,:])
        ax1.plot(E[0,:], E[1,:])
        ax1.axis("equal")
        ax1.set_xlabel(r"(a) $S$")

        # Plot (b) V.H @ S
        bs = Vh @ S
        be = Vh @ E
        ax2 = plt.subplot(222)
        ax2.plot(bs[0,:], bs[1,:])
        ax2.plot(be[0,:], be[1,:])
        ax2.axis("equal")
        ax2.set_xlabel(r"(b) $V^H S$")

        # Plot (c) Σ @ V.H @ S
        cs = np.diag(Sigma) @ bs
        ce = np.diag(Sigma) @ be
        ax3 = plt.subplot(223)
        ax3.plot(cs[0,:], cs[1,:])
        ax3.plot(ce[0,:], ce[1,:])
        ax3.axis("equal")
        ax3.set_xlabel(r"(c) $Σ V^H S$")
        ax3.set_xlim([-2, 2])
        ax3.set_ylim([-4, 4])

        # Plot (d) U @ Σ @ V.H @ S
        ds = U @ cs
        de = U @ ce
        ax4 = plt.subplot(224)
        ax4.plot(ds[0,:], ds[1,:])
        ax4.plot(de[0,:], de[1,:])
        ax4.axis("equal")
        ax4.set_xlabel(r"(d) $U Σ V^{H} S$")
        ax4.set_xlim([-4, 4])
        ax4.set_ylim([-4, 4])

        plt.show()
    
    def test_vizualization(this):
        A = np.array([[3, 1], [1, 3]])
        this.visualize_svd(A)


    def svd_approx(this, A, s):
        """We can take the compact SVD a step farther and cut out the s-most-significant singular
            values, dropping the rest. The Schmidt, Mirsky, Eckhart-Young theorem, proves that this is
            the best rank-s approximation of A wrt the 2-norm and Frobenius norm. 

            Instead of storing all mn values of A, storing the matrices U1, Σ1 and V1 only
            requires saving a total of mr + r + nr values.

            Return the truncated SVD, along with the number of bytes needed to store the
            approximation.

            Parameters:
                A ((m,n), ndarray)
                s (int): The rank of the desired approximation.
    
            Returns:
                ((m,n), ndarray) The best rank s approximation of A, A_s.
                (int) The number of entries needed to store the truncated SVD.
        """
        U, Σ, Vh = sla.svd(A)
        # Strip the r - s columns of U.
        U_1 = U[:, :s]
        # Strip the r - s singular values from Σ.
        Σ_1 = Σ[:s]
        # Strip the r-s rows of V^H.
        Vh_1 = Vh[:s, :]

        # So the approximation is the combination of the truncated SVD.
        A_s = U_1 @ np.diag(Σ_1) @ Vh_1

        # The rank of A_s should not be greater than s itself.
        if s > np.linalg.matrix_rank(A_s):
            raise ValueError(f"s > rank(A): Rank A: {np.linalg.matrix_rank(As)} S: {s}")
        
        return A_s, U_1.size + Σ_1.size + Vh_1.size

    def test_svd_approximation(this):
        A = np.random.random((20,20))
        approx = this.svd_approx(A, 3)
        print(approx)
        print(np.linalg.matrix_rank(approx[0]) == 3)


    def lowest_rank_approx(this, A, err):
        """Return the lowest rank approximation of A with error less than 'err' with respect to
            the matrix 2-norm, along with the number of bytes needed to store the approximation via
            the truncated SVD.
    
            Parameters:
                A ((m, n) ndarray)
                err (float): Desired maximum error.
    
            Returns:
                A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
                    ||A - A_s||_2 < err.
                (int) The number of entries needed to store the truncated SVD.
        """
        # Get the SVD
        U, Σ, V_h = sla.svd(A)
        size = 0
        # Iteratively whack off rows and columns respectively to approximate with lower rank
        for s in range(np.linalg.matrix_rank(A)):                                                                                                                      U1 = U[:, :s]
            Σ_1 = Σ[:s]
            Vh_1 = V_h[:s, :]
            A_s = U_1 @ np.diag(Σ_1) @ Vh_1
            if np.linalg.norm(A - A_s) < err:
                size = U1.size + sigma1.size + Vh1.size
                break
        # If it wasn't caught before then it ain't gonna happen, ie can't approx em all
        if size == 0:
            raise ValueError("A cannot be approximated within the tolerance by a matrix of lesser rank.")
        return As, int(size)
    
    
    def compress_image(this, filename, s):
        """Plot the original image found at 'filename' and the rank s approximation of the image
        found at 'filename.' State in the figure title the difference in the number of entries
        used to store the original image and the approximation.
    
        Parameters:
            filename (str): Image file path.
            s (int): Rank of new image.
        """
        raise NotImplementedError("Problem 5 Incomplete")

if __name__ == "__main__":
    svd = SVD()
    svd.test_compact_svd()
    svd.test_vizualization()
    svd.test_svd_approximation()
