#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

namespace Tools {

/**
* A helper method to calculate RMSE.
*/
VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth);

/**
* A helper method to calculate Jacobians.
*/
MatrixXd CalculateJacobian(const VectorXd &x);

};

#endif /* TOOLS_H_ */
