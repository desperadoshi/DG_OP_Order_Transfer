import numpy as np
from sympy import symbols, integrate

def GaussLegendreRootsWeights(in_order):
    from numpy import arange,sqrt,diag,argsort
    from numpy.linalg import eig
    idx_arr=arange(1,in_order+1)
    beta_arr=idx_arr/(sqrt(4*idx_arr**2-1))
    T_mat=(diag(beta_arr,1)+diag(beta_arr,-1))
    [eigval_arr,eigvec_mat]=eig(T_mat)
    idx_sort=argsort(eigval_arr)
    roots_arr=eigval_arr[idx_sort]
    eigvec_mat=eigvec_mat[:,idx_sort]
    weights_arr=2*eigvec_mat[0,:]**2
    return roots_arr,weights_arr

def getSolutionPoints(in_p_order):
    SolPts_Vec,comp_sp_wi_arr=GaussLegendreRootsWeights(in_p_order)
    return SolPts_Vec

def getLagrangeBasis(in_p_order,in_x_arr):
    from numpy.polynomial.polynomial import Polynomial
    #########################
    # Construct Lagrange Basis Functions in the form of the Polynomial class
    # imported from numpy.
    # Output:
    #           List of Polynomial object. Each Polynomial object has 3 default
    #           parameters. The first is the coefficients, second is the domain,
    #           third is the window size. Details about the latter 2 parameters
    #           are in the definition of Polynomial class in numpy.
    PolyList_List = []
    for j in range(in_p_order+1):
        Poly_Poly = 1.0
        for k in range(in_p_order+1):
            if k == j:
                continue
            Poly_Poly *= Polynomial([-in_x_arr[k], 1.0]) / (in_x_arr[j] - in_x_arr[k])
        PolyList_List.append(Poly_Poly)
    return PolyList_List

x = symbols('x')

def calc_matrix_MN(in_p1, in_p2):
    # in_p1 and in_p2 are P order
    # Refer to Fidkowski_Master2004.pdf. Eq. 3.14
    n_nodes1 = in_p1 + 1
    n_nodes2 = in_p2 + 1
    x_arr1 = getSolutionPoints(in_p1)
    x_arr2 = getSolutionPoints(in_p2)
    Basis_List1 = getLagrangeBasis(in_p1,x_arr1)
    Basis_List2 = getLagrangeBasis(in_p2,x_arr2)
    BasisCoef_List1 = [Basis_List1[i].coef for i in range(n_nodes1)]
    BasisCoef_List2 = [Basis_List2[i].coef for i in range(n_nodes2)]
    out_mat = np.zeros((len(BasisCoef_List1),len(BasisCoef_List2)))
    for i_basis_func1 in range(len(BasisCoef_List1)):
        basis_func1_coef_arr = BasisCoef_List1[i_basis_func1]
        basis_func1 = 0
        for coef1 in range(basis_func1_coef_arr.size):
            basis_func1 += basis_func1_coef_arr[coef1] * x**coef1
        for i_basis_func2 in range(len(BasisCoef_List2)):
            basis_func2_coef_arr = BasisCoef_List2[i_basis_func2]
            basis_func2 = 0
            for coef2 in range(basis_func2_coef_arr.size):
                basis_func2 += basis_func2_coef_arr[coef2] * x**coef2
            basis_func_12_integral = integrate(basis_func1*basis_func2, (x, -1, 1))
            out_mat[i_basis_func1,i_basis_func2] = basis_func_12_integral
    return out_mat

p1_order = 2
p2_order = 2
M_mat = calc_matrix_MN(p1_order, p2_order)

p1_order = 2
p2_order = 3
N_mat = calc_matrix_MN(p1_order, p2_order)

I_mat = np.matmul(np.linalg.inv(M_mat),N_mat)
print("I_mat is")
print(I_mat)

x_arr1 = getSolutionPoints(p1_order)
x_arr2 = getSolutionPoints(p2_order)

print("Solution points of P%d are"%(p1_order))
print(x_arr1)
print("Solution points of P%d are"%(p2_order))
print(x_arr2)

u_arr1 = x_arr1**3
u_arr2 = x_arr2**3

print("Origninal solutions of P%d are"%(p1_order))
print(u_arr1)
print("Origninal solutions of P%d are"%(p2_order))
print(u_arr2)

u_arr2_1 = np.matmul(I_mat,u_arr2)
print("Restricted solutions of P%d are"%(p2_order))
print(u_arr2_1)

# Prolongate the (P-1) order solution from P solution points to (P+1) solution points.
from numpy.polynomial.polynomial import polyval
Basis_List1 = getLagrangeBasis(p1_order,x_arr1)
Basis_List2 = getLagrangeBasis(p2_order,x_arr2)
n_nodes1 = p1_order + 1
n_nodes2 = p2_order + 1
BasisCoef_List1 = [Basis_List1[i].coef for i in range(n_nodes1)]
BasisCoef_List2 = [Basis_List2[i].coef for i in range(n_nodes2)]
prolong_mat = np.zeros((n_nodes2, n_nodes1))
for i_basis_func1 in range(len(BasisCoef_List1)):
    basis_func1_coef_arr = BasisCoef_List1[i_basis_func1]
    prolong_mat[:,i_basis_func1] = polyval(x_arr2, basis_func1_coef_arr)

print("Prolongation matrix is")
print(prolong_mat)
u_arr2_1_2 = np.matmul(prolong_mat, u_arr2_1)
print("Prolongation of restricted solutions of P%d are"%(p2_order))
print(u_arr2_1_2)

# Debug P2
a = np.array([ [x_arr1[0]**2, x_arr1[0]], [x_arr1[2]**2, x_arr1[2]] ])
b = np.array([ u_arr2_1[0], u_arr2_1[2] ])
coef = np.linalg.solve(a, b)
print("Ordered coefficients of the restricted polynomial are")
print(coef)
print("The following array is expected to be the same as the prolongation of the restricted solutions of P2.")
print(coef[0]*x_arr2**2 + coef[1]*x_arr2)
