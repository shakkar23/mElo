// onstruct Omega matrix for given k
//
// This function constructs the 2k * 2k Omega matrix necessary for calulating
// and updating mELO ratings.
//
// @param k Integer defining the complexity of non-transitive interactions to
// model.
//
// @return a matrix
// @export
//
// @examples
// construct_omega(1)
// construct_omega(3)
/*
construct_omega < -function(k) {

    E < -diag(2 * k)
        omega < -matrix(0, ncol = 2 * k, nrow = 2 * k)

        for (i in 1 : k) {
            omega < -omega +
                E[, 2 * i - 1] % *%t(E[, 2 * i]) -
                E[, 2 * i] % *%t(E[, 2 * i - 1])
        }

    return(omega)
}
*/


// convert to c++

#include <vector>
#include <string>
#include <cmath>


#include "types.hpp"
#include "Number.hpp"
#include "construct_omega.hpp"

Matrix construct_omega() {
	Matrix E(2 * k, 2 * k);

	for (int i = 0; i < 2 * k; i++) {
		E(i,i) = 1;
	}

	Matrix matrix(2 * k, 2 * k);
	for (int i = 0; i < k; i++) {
		Row col_i_1(2 * k);
		Column col_i(2 * k);

		for (int j = 0; j < 2 * k; j++) {
			col_i[j] = E(j,2 * i);
			col_i_1[j] = E(j,2 * i + 1);
		}

		auto first = (col_i_1 * col_i).transpose();
		matrix += first;

		auto second = (col_i_1 * col_i);
		matrix -= second;
	}

	return matrix;
}