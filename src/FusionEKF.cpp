#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // clang-format off
  R_laser_ << 
              0.0225, 0,
              0, 0.0225;

  R_radar_ << 
              0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ << 
              1, 0, 0, 0,
              0, 1, 0, 0;
  // clang-format on
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  // first measurement
  if (!is_initialized_) {
    previous_timestamp_ = measurement_pack.timestamp_;

    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 0, 0, 1, 1;

    switch (measurement_pack.sensor_type_) {
      case (MeasurementPackage::RADAR): {
        float rho = measurement_pack.raw_measurements_(0);
        float phi = measurement_pack.raw_measurements_(1);
        ekf_.x_ <<  // clang-format off
                    rho * cos(phi), 
                    rho * sin(phi), 
                    1, 
                    1;
                    // clang-format on
        break;
      }
      case (MeasurementPackage::LASER):
        ekf_.x_ <<  // clang-format off
                    measurement_pack.raw_measurements_(0),
                    measurement_pack.raw_measurements_(1),
                    1, 
                    1;
                    // clang-format on
        break;
    }

    ekf_.F_ = MatrixXd::Identity(4, 4);
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.P_ = MatrixXd(4, 4);

    ekf_.P_ <<  // clang-format off
                1, 0, 0, 0,
                0, 1, 0, 0, 
                0, 0, 1000, 0, 
                0, 0, 0, 1000;
                // clang-format on

    ekf_.H_ = H_laser_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  float noise_ax = 9;
  float noise_ay = 9;

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) /
             1000000.0;  // dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt / 2.;
  float dt_4 = dt_2 * dt_2 / 4.;
  float sx = noise_ax;
  float sy = noise_ay;

  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  ekf_.Q_ <<  // clang-format off
              dt_4 * sx, 0, dt_3 * sx, 0,
              0, dt_4 * sy, 0, dt_3 * sy, 
              dt_3 * sx, 0, dt_2 * sx, 0,
              0, dt_3 * sy, 0, dt_2 * sy;
              // clang-format on

  ekf_.Predict();

  // Update
  switch (measurement_pack.sensor_type_) {
    case (MeasurementPackage::RADAR):
      ekf_.R_ = R_radar_;
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);
      break;
    case (MeasurementPackage::LASER):
      ekf_.R_ = R_laser_;
      ekf_.Update(measurement_pack.raw_measurements_);
      break;
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
