#pragma once

#include <vector>
#include <stdexcept>

#include "Number.hpp"


struct Array {
	std::vector<Numeric> data;
	int height;
	int width;
};

struct Row {
	std::vector<Numeric> data;
	int width;

	// default constructor
	Row() : width(0) {}
	Row(int width) : width(width) {
		data.resize(width, 0);
	}

	// operator[] overload for easy access
	Numeric& operator[](int index) {
		return data[index];
	}

	struct Matrix operator*(const struct Column& col);
};

struct Column {
	std::vector<Numeric> data;
	int height;
	
	// default constructor
	Column() : height(0) {}
	Column(int height) : height(height) {
		data.resize(height, 0);
	}

	// operator[] overload for easy access
	Numeric& operator[](int index) {
		return data[index];
	}


	// dot
	Numeric dot(const Column& col) const {
		if (height != col.height) {
			throw std::invalid_argument("Column dimensions do not match for dot product.");
		}
		Numeric result(0);

		for (int i = 0; i < height; ++i) {
			result += data[i] * col.data[i];
		}
		return result;
	}
	// add
	Column operator+(const Column& other) const {
		if (height != other.height) {
			throw std::invalid_argument("Column dimensions do not match for addition.");
		}
		Column result;
		result.height = height;
		result.data.resize(height);
		for (int i = 0; i < height; ++i) {
			result.data[i] = data[i] + other.data[i];
		}
		return result;
	}

	// sub
	Column operator-(const Column& other) const {
		if (height != other.height) {
			throw std::invalid_argument("Column dimensions do not match for subtraction.");
		}
		Column result;
		result.height = height;
		result.data.resize(height);
		for (int i = 0; i < height; ++i) {
			result.data[i] = data[i] - other.data[i];
		}
		return result;
	}
};

// transpose row to col and vice versa
inline Row transpose(const Column& col) {
	Row row;
	row.width = col.height;
	row.data.resize(col.height);

	for (int i = 0; i < col.height; i++) {
		row.data[i] = col.data[i];
	}

	return row;
}

inline Column transpose(const Row& row) {
	Column col;
	col.height = row.width;
	col.data.resize(row.width);
	for (int i = 0; i < row.width; i++) {
		col.data[i] = row.data[i];
	}
	return col;
}


struct Matrix {
	std::vector<Numeric> data;
	int height;
	int width;
	
	Matrix() : data(), height(0), width(0) {};

	Matrix(int height, int width) : height(height), width(width) {
		data.resize(height * width, 0);
	}
	
	Numeric& operator[](int index) {
		return data[index];
	}

	Column get_col(int index) {
		Column col(height);
		for (int i = 0; i < height; i++) {
			col.data[i] = (*this)(i,index);
		}
		return col;
	}

	void set_col(int index, const Column& col) {
		for (int i = 0; i < height; i++) {
			data[i * width + index] = col.data[i];
		}
	}

	// multiply
	Matrix operator*(const Matrix& other) const {
		if (width != other.height) {
			throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
		}
		Matrix result;
		result.height = height;
		result.width = other.width;
		result.data.resize(height * other.width, 0);
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < other.width; ++j) {
				for (int k = 0; k < width; ++k) {
					result.data[i * other.width + j] += data[i * width + k] * other.data[k * other.width + j];
				}
			}
		}
		return result;
	}
	// mul column
	Column operator*(const Column& col) const {
		if(width != col.height) {
			throw std::invalid_argument("Matrix and column dimensions do not match for multiplication.");
		}
		Column result(height);

		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				result.data[i] += data[i * width + j] * col.data[j];
			}
		}
		return result;
	}

	// transpose
	inline Matrix transpose() const {
		Matrix result;
		result.height = width;
		result.width = height;
		result.data.resize(height * width);
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				result.data[j * height + i] = data[i * width + j];
			}
		}
		return result;
	}

	// add
	Matrix operator+(const Matrix& other) const {
		if (height != other.height || width != other.width) {
			throw std::invalid_argument("Matrix dimensions do not match for addition.");
		}
		Matrix result;
		result.height = height;
		result.width = width;
		result.data.resize(height * width);
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				result.data[i * width + j] = data[i * width + j] + other.data[i * width + j];
			}
		}
		return result;
	}

	// add equal
	Matrix& operator+=(const Matrix& other) {
		if (height != other.height || width != other.width) {
			throw std::invalid_argument("Matrix dimensions do not match for addition.");
		}
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				data[i * width + j] += other.data[i * width + j];
			}
		}
		return *this;
	}

	// add euqal scalar
	Matrix& operator+=(Numeric scalar) {
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				data[i * width + j] += scalar;
			}
		}
		return *this;
	}

	// subtract
	Matrix operator-(const Matrix& other) const {
		if (height != other.height || width != other.width) {
			throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
		}
		Matrix result;
		result.height = height;
		result.width = width;
		result.data.resize(height * width);
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				result.data[i * width + j] = data[i * width + j] - other.data[i * width + j];
			}
		}
		return result;
	}

	// sub equal
	Matrix& operator-=(const Matrix& other) {
		if (height != other.height || width != other.width) {
			throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
		}
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				data[i * width + j] -= other.data[i * width + j];
			}
		}
		return *this;
	}
	// sub equal scalar
	Matrix& operator-=(Numeric scalar) {
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				data[i * width + j] -= scalar;
			}
		}
		return *this;
	}

	// scalar multiplication
	Matrix operator*(Numeric scalar) const {
		Matrix result;
		result.height = height;
		result.width = width;
		result.data.resize(height * width);
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				result.data[i * width + j] = data[i * width + j] * scalar;
			}
		}
		return result;
	}

	// [,] operator
	Numeric& operator()(int y, int x) {
		return data[y * width + x];
	}
};


// mul row with col operator overload
inline Matrix Row::operator*(const Column& col) {
	// if they are now the same throw
	if (width != col.height)
		throw std::invalid_argument("Row and column dimensions do not match for multiplication.");

	Matrix result(width, col.height);

	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < col.height; ++j) {
			result(i, j) = data[i] * col.data[j];
		}
	}

	return result;
}


struct DataFrame {
	std::vector<std::string> Player_1;
	std::vector<std::string> Player_2;
	std::vector<Numeric> outcome;
};

struct mELO_rating {
	DataFrame ratings;
	Array history;
	Matrix c_mat;
	Array c_mat_history;
	Numeric p1_advantage;
	int k;
	Numeric eta_1;
	Numeric eta_2;
	std::string type;
	std::vector<Numeric> preds;
	std::vector<Numeric> outcomes;
	Numeric preds_logloss;
};
