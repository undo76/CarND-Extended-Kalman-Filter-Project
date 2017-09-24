#include "kalman_filter.h"
#include <iostream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);

  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

VectorXd h(const VectorXd &x) {
  double px = x(0);
  double py = x(1);
  double vx = x(2);
  double vy = x(3);

  double rho = sqrt(px * px + py * py);

  VectorXd res(3);
  // check division by zero
  if (fabs(rho) < 0.0001) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    res << rho, atan2(py, px), 0;
  } else {
    res << rho, atan2(py, px), (px * vx + py * vy) / rho;
  }

  return res;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  MatrixXd Hj = Tools::CalculateJacobian(x_);

  const VectorXd z_pred = h(x_);

  VectorXd y = z - z_pred;

  if (y(1) > M_PI) {
    y(1) -= 2 * M_PI;
  } else if (y(1) < -M_PI) {
    y(1) += 2 * M_PI;
  }

  MatrixXd Ht = Hj.transpose();
  MatrixXd S = Hj * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);

  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * Hj) * P_;
}