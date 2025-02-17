#pragma once

#include <vector>
#include "Number.hpp"

#include <cmath>
#include <vector>
#include <span>

inline double logloss(std::vector<Numeric> prediction, std::vector<Numeric> outcome, Numeric tol = 1e-15) {
	if (outcome.size() != prediction.size()) {
		puts("Observed outcome and predicted outcome need to be equal lengths!");
		exit(1);
	}

	std::vector<Numeric> pred_capped = (prediction);
	for (int i = 0; i < pred_capped.size(); i++) {
		pred_capped[i] = std::min(1 - tol, pred_capped[i]);
		pred_capped[i] = std::max(tol, pred_capped[i]);
	}

	Numeric loss = 0;
	for (int i = 0; i < outcome.size(); i++) {
		loss += outcome[i] * log(pred_capped[i]) + (1 - outcome[i]) * log(1 - pred_capped[i]);
	}

	return -loss / outcome.size();
}

inline Numeric sigmoid(Numeric x) {
	return 1.0 / (1.0 + exp(-x));
}

template <typename T>
void softmax(std::span<T> data) {
	T sum = 0;
	for (const auto& value : data) {
		sum += exp(value);
	}
	for (auto& value : data) {
		value = exp(value) / sum;
	}
}

