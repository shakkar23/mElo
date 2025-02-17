
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <tuple>



#include "types.hpp"
#include "Number.hpp"
#include "predict.hpp"
#include "update_c.hpp"

#include <iostream>

std::tuple<Numeric, Numeric, Column, Column> update_melo(Numeric rA, Numeric rB, Column cA, Column cB, Numeric observed_score) {
	Numeric predicted_score = predict(rA, rB, cA, cB);

	Numeric error = observed_score - predicted_score;

	Numeric K = 16;

	rA += K * error;
	rB += -K * error;
	

	for (int i = 0; i < k; i++) {
		cA[0+i*2]  += std::clamp( error * cB[1 + i*2], (Numeric)-1, (Numeric)1);
		cA[1+i*2]  += std::clamp(-error * cA[1 + i*2], (Numeric)-1, (Numeric)1);
		cB[0+i*2]  += std::clamp(-error * cB[0 + i*2], (Numeric)-1, (Numeric)1);
		cB[1+i*2]  += std::clamp( error * cA[0 + i*2], (Numeric)-1, (Numeric)1);
	}

	return { rA, rB, cA, cB };
}