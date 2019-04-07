The restriction and prolongation operator is part of the implementation of the discontinuity sensor proposed by Persson in his paper: Persson_AIAAConf2006.

For the prolongation of the restricted solutions, the following formula is used.

$$
U^{P-1}_{i} = \sum_j \Phi_j |_{x=x_i} U^{P-1}_j
$$

It seems like the prolongation is expected to transfer the lower order solution to a higher order solution. The above does this: transfer the solution from the nodes defined by the lower order polynomial to more nodes defined by a higher order polynomial while keeping the solution accuracy to the same order.

