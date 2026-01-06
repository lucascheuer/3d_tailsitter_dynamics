#pragma once

#include <eigen3/Eigen/Eigen>
#include <vector>

#include "OsqpEigen/OsqpEigen.h"
#include "polynomial.hpp"
#include "state.hpp"

class MinSnapTraj
{
   public:
    struct Waypoint
    {
        Waypoint() : pos(Eigen::Vector3d::Zero()), yaw(0.0) {};
        Waypoint(Eigen::Vector3d pos, double yaw)
            : pos(pos), yaw(yaw) {

              };
        Eigen::Vector3d pos;
        double yaw;
        friend std::ostream& operator<<(std::ostream& os, const Waypoint& waypoint)
        {
            os << waypoint.pos.x() << "," << waypoint.pos.y() << "," << waypoint.pos.z() << ","
               << waypoint.yaw;
            return os;
        }
    };
    MinSnapTraj();
    void SetStartDerivatives(std::vector<Waypoint> start_derivatives);
    void SetEndDerivatives(std::vector<Waypoint> end_derivatives);

    void AddWaypoint(Waypoint new_waypoint);
    void ClearWaypoints();
    bool
    Solve(double speed_weight, double start_speed, int step_limit = 100, double descent_rate = 0.1);

    void Evaluate(double time, State& state);
    double EndTime();
    bool GetWaypoint(int waypoint_num, Waypoint& to_fill);
    Waypoint* GetWaypoint(int waypoint_num);

    int GetWaypointCount() { return int(waypoints_.size()); }
    bool solved() { return solved_; }
    Eigen::VectorXd GetTimes() { return times_; }

   private:
    void AllocateInitialTimes(Eigen::VectorXd& times, double average_speed);
    void CalculateTimePowers(const Eigen::MatrixXd& times, Eigen::MatrixXd& time_powers);
    double CalculateCost(
        const Eigen::VectorXd& b_x,
        const Eigen::VectorXd& b_y,
        const Eigen::VectorXd& b_z,
        const Eigen::VectorXd& b_yaw,
        const Eigen::VectorXd& times,
        double speed_weight);
    void GenerateQ(const Eigen::MatrixXd& time_powers, Eigen::MatrixXd& Q);
    void FillH(const Eigen::VectorXd& time_powers_row, Eigen::MatrixXd& H);
    void GenerateA(const Eigen::MatrixXd& time_powers, Eigen::MatrixXd& A);
    void GenerateB(
        Eigen::VectorXd& b_x,
        Eigen::VectorXd& b_y,
        Eigen::VectorXd& b_z,
        Eigen::VectorXd& b_yaw);
    int CalculatePolyCoeffMultiplier(int coeff, int derivative_count);
    int CalculatePolyCoeffPower(int coeff, int derivative_count);
    void CalculatePolyDerivativeMultipliers(
        const int coeff_count,
        const int derivative_count,
        const Eigen::VectorXd& time_powers_row,
        Eigen::RowVectorXd& polynomial_derivative);

    bool solved_;
    bool first_time_;
    double start_time_;
    double end_time_;
    double total_time_;

    const int kPosMinDerivative = 4;
    const int kPolyOrder = 7;
    const int kCoeffCount = kPolyOrder + 1;
    // n is poly order
    // m is number of "waypoints"
    // consts per solve
    int num_segments_;
    int num_internal_joints_;
    int num_variables_;
    int num_constraints_;
    Eigen::VectorXd times_;

    OsqpEigen::Solver solver_;

    std::vector<Waypoint> start_derivatives_;
    std::vector<Waypoint> end_derivatives_;
    std::vector<Waypoint> waypoints_;
    std::vector<std::vector<Polynomial>> x_polys_;
    std::vector<std::vector<Polynomial>> y_polys_;
    std::vector<std::vector<Polynomial>> z_polys_;
    std::vector<std::vector<Polynomial>> yaw_polys_;
};