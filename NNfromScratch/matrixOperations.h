#pragma once

#include <vector>
#include <numeric>

using namespace std;

typedef vector<double> VD;
typedef vector<vector<double> > MD;



namespace mat
{


	// Function to Transpose a Matrix
	MD transpose(const MD& m) {

		MD mat;

		for (size_t i = 0; i < m[0].size(); i++) 
		{
			mat.push_back({});

			for (size_t j = 0; j < m.size(); j++)
			{
				mat[i].push_back(m[j][i]);
			}
		}

		return mat;
	}



	// Operator Overloading to Multiply two Matrices
	MD operator*(const MD& m1, const MD& m2) noexcept {

		MD result;
		MD m3 = transpose(m2);

		for (size_t i = 0; i < m1.size(); i++) 
		{
			result.push_back({});

			for (size_t j = 0; j < m3.size(); j++) 
			{
				result[i].push_back(inner_product(m1[i].begin(), m1[i].end(), m3[j].begin(), 0.0));
			}
		}
		return result;
	}



	// Operator Overloading to Add Matrix and Vector
	MD operator+(const MD& m, const VD& v) noexcept {

		MD result;

		for (size_t j = 0; j < m.size(); j++)
		{
			result.push_back({});

			for (size_t i = 0; i < m[j].size(); i++)
			{
				result[j].push_back(m[j][i] + v[j]);
			}
		}
		return result;
	}



	// Dot Product of two vectors
	double vectorDot(const VD& v1, const VD& v2) {

		double product = 0;
		for (size_t i = 0; i < v1.size(); i++)
		{
			product += v1[i] * v2[i];
		}

		return product;

	}



	// Print Matrix
	void printMatrix(MD matrix) {
		for (size_t i = 0; i < matrix.size(); i++)
		{
			for (size_t j = 0; j < matrix[i].size(); j++)
			{
				cout << matrix[i][j] << " ";
			}
			cout << endl;
		}

		cout << "\n\n";
	}



	// Print Vector
	void printVector(VD vec) {
		for (int i = 0; i < vec.size(); i++)
		{
			cout << vec[i] << " ";
		}
		cout << "\n\n";
	}



	// Clamp lower and upper values
	double clamp(double x, const double min, const double max)
	{
		if (x < min) x = min;
		if (x > max) x = max;

		return x;
	}



}