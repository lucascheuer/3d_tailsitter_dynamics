#include "min_snap_traj.hpp"

#include <iomanip>
#include <iostream>

#include "gdcpp.hpp"

MinSnapTraj::MinSnapTraj()
    : solved_(false), start_time_(0.0), end_time_(0.0), total_time_(0.0), first_time_(true)
{
    for (int derivative = 0; derivative < kPosMinDerivative; ++derivative)
    {
        start_derivatives_.emplace_back(Waypoint(Eigen::Vector3d::Zero(), 0));
        end_derivatives_.emplace_back(Waypoint(Eigen::Vector3d::Zero(), 0));
    }
}

void MinSnapTraj::SetStartDerivatives(std::vector<Waypoint> start_derivatives)
{
    if (start_derivatives.size() == kPosMinDerivative)
    {
        start_derivatives_ = start_derivatives;
    } else
    {
        std::cout << "Failed to set start derivatives. Wrong size of: " << start_derivatives.size()
                  << std::endl;
    }
}
void MinSnapTraj::SetEndDerivatives(std::vector<Waypoint> end_derivatives)
{
    if (end_derivatives.size() == kPosMinDerivative)
    {
        end_derivatives_ = end_derivatives;
    } else
    {
        std::cout << "Failed to set end derivatives. Wrong size of: " << end_derivatives.size()
                  << std::endl;
    }
};
void MinSnapTraj::AddWaypoint(Waypoint new_waypoint)
{
    waypoints_.push_back(new_waypoint);
    solved_ = false;
}

void MinSnapTraj::ClearWaypoints()
{
    waypoints_.clear();
    solved_ = false;
}

double MinSnapTraj::EndTime() { return total_time_; }

bool MinSnapTraj::GetWaypoint(int waypoint_num, Waypoint& to_fill)
{
    if (waypoint_num >= int(waypoints_.size()) || waypoint_num < 0)
    {
        return false;
    }
    to_fill.pos = waypoints_[waypoint_num].pos;
    to_fill.yaw = waypoints_[waypoint_num].yaw;
    return true;
}

MinSnapTraj::Waypoint* MinSnapTraj::GetWaypoint(int waypoint_num)
{
    if (waypoint_num >= int(waypoints_.size()) || waypoint_num < 0)
    {
        return nullptr;
    }
    return &waypoints_[waypoint_num];
    // to_fill.pos = waypoints_[waypoint_num].pos;
    // to_fill.yaw = waypoints_[waypoint_num].yaw;
}

void MinSnapTraj::Evaluate(double time, State& state)
{
    // std::cout << total_time_ << "\t" << time << std::endl;
    assert(time >= 0.0);
    assert(time <= total_time_);
    assert(solved_);
    int segment = 0;
    for (segment = 0; segment < num_segments_; ++segment)
    {
        if (time > times_(segment))
        {
            time -= times_(segment);
        } else
        {
            break;
        }
    }

    state.x = x_polys_[segment][0].Evaluate(time);
    state.vx = x_polys_[segment][1].Evaluate(time);
    state.ax = x_polys_[segment][2].Evaluate(time);
    state.jx = x_polys_[segment][3].Evaluate(time);

    state.y = y_polys_[segment][0].Evaluate(time);
    state.vy = y_polys_[segment][1].Evaluate(time);
    state.ay = y_polys_[segment][2].Evaluate(time);
    state.jy = y_polys_[segment][3].Evaluate(time);

    state.z = z_polys_[segment][0].Evaluate(time);
    state.vz = z_polys_[segment][1].Evaluate(time);
    state.az = z_polys_[segment][2].Evaluate(time);
    state.jz = z_polys_[segment][3].Evaluate(time);

    state.yaw = yaw_polys_[segment][0].Evaluate(time);
    state.vyaw = yaw_polys_[segment][1].Evaluate(time);
}

bool MinSnapTraj::Solve(
    double speed_weight,
    double start_speed,
    int step_limit,
    double descent_rate)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(5);
    const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ",");
    num_segments_ = waypoints_.size() - 1;
    num_internal_joints_ = waypoints_.size() - 2;
    num_variables_ = kCoeffCount * num_segments_;
    num_constraints_ = 2 * num_segments_ + 8 + 4 * num_internal_joints_;

    Eigen::VectorXd b_x(num_constraints_);
    Eigen::VectorXd b_y(num_constraints_);
    Eigen::VectorXd b_z(num_constraints_);
    Eigen::VectorXd b_yaw(num_constraints_);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_variables_);

    GenerateB(b_x, b_y, b_z, b_yaw);

    Eigen::VectorXd segment_time_gradients = Eigen::VectorXd::Zero(num_segments_);
    Eigen::VectorXd times = Eigen::VectorXd::Zero(num_segments_);
    Eigen::VectorXd last_times = Eigen::VectorXd::Zero(num_segments_);
    AllocateInitialTimes(times, start_speed);

    Eigen::MatrixXd time_powers;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(num_variables_, num_variables_);
    Eigen::MatrixXd A =
        Eigen::MatrixXd::Zero(2 * num_segments_ + 8 + 4 * num_internal_joints_, num_variables_);
    std::cout << "A size " << A.cols() << "," << A.rows() << "\n";
    CalculateTimePowers(times, time_powers);
    GenerateQ(time_powers, Q);
    GenerateA(time_powers, A);
    std::cout << "bx: " << b_x.transpose() << std::endl;
    std::cout << "by: " << b_y.transpose() << std::endl;
    std::cout << "bz: " << b_z.transpose() << std::endl;
    std::cout << "byaw: " << b_yaw.transpose() << std::endl;
    Eigen::SparseMatrix<double> sparseA = A.sparseView();
    Eigen::SparseMatrix<double> sparseQ = Q.sparseView();
    if (!first_time_)
    {
        solver_.data()->clearHessianMatrix();
        solver_.data()->clearLinearConstraintsMatrix();
        solver_.clearSolverVariables();
        solver_.clearSolver();

    } else
    {
        solver_.settings()->setWarmStart(true);
        solver_.settings()->setVerbosity(false);
        first_time_ = false;
    }
    solver_.data()->setNumberOfVariables(num_variables_);
    solver_.data()->setNumberOfConstraints(num_constraints_);
    if (!solver_.data()->setHessianMatrix(sparseQ))
    {
        std::cout << "Hessian not set" << std::endl;
        return false;
    }
    if (!solver_.data()->setGradient(gradient))
    {
        std::cout << "Gradient not set" << std::endl;
        return false;
    }
    if (!solver_.data()->setLinearConstraintsMatrix(sparseA))
    {
        std::cout << "Constraint matrix not set" << std::endl;
        return false;
    }
    if (!solver_.data()->setLowerBound(b_x))
    {
        std::cout << "Constraint LB not set" << std::endl;
        return false;
    }
    if (!solver_.data()->setUpperBound(b_x))
    {
        std::cout << "Constraint UB not set" << std::endl;
        return false;
    }
    if (!solver_.initSolver())
    {
        std::cout << "Not initialized" << std::endl;
        return false;
    }
    double total_time = times.sum();
    double last_cost = std::nanf("");
    // solve loop
    int ii = 0;
    double speed_descent_rate = descent_rate;
    double base_descent_rate = descent_rate;
    while (ii < step_limit)
    {
        double base_cost = CalculateCost(b_x, b_y, b_z, b_yaw, times, speed_weight);
        std::cout << ii << ": base cost: " << base_cost << std::endl;
        // std::cout << "Total time: " << times.sum() << "\ttimes: " <<
        // times.transpose().format(fmt)
        //           << std::endl;
        double h = 0.0000001;
        // loop through all segments to get gradient
        for (int segment = 0; segment < num_segments_; ++segment)
        {
            Eigen::VectorXd times_delta = Eigen::VectorXd::Zero(num_segments_);
            double subtraction = -1.0 / (num_segments_ - 1);
            times_delta.array() += subtraction;
            times_delta(segment) += 1 - subtraction;
            times_delta.array() *= h;
            double segment_adjust_cost =
                CalculateCost(b_x, b_y, b_z, b_yaw, times + times_delta, speed_weight);
            double gradient = (segment_adjust_cost - base_cost) / h;
            segment_time_gradients(segment) = gradient;
        }

        double temp_descent = descent_rate * base_cost;
        while ((times.array() < (segment_time_gradients * temp_descent).array()).sum() > 0)
        {
            temp_descent *= 0.1;
        }

        times -= segment_time_gradients * temp_descent;
        if (speed_weight == 0)
        {
            times *= total_time / times.sum();
        }

        if (abs(last_cost / base_cost - 1.0) < 0.001)
        {
            std::cout << "Reached break at " << ii << " steps" << std::endl;
            break;
        }
        last_cost = base_cost;
        last_times = times;
        ii++;
    }

    times_ = times;
    total_time_ = times_.sum();
    std::cout << times_.transpose().format(fmt) << std::endl;
    CalculateTimePowers(times_, time_powers);
    GenerateQ(time_powers, Q);
    GenerateA(time_powers, A);
    sparseA = A.sparseView();
    sparseQ = Q.sparseView();
    solver_.clearSolver();
    solver_.data()->clearHessianMatrix();
    solver_.data()->clearLinearConstraintsMatrix();
    solver_.data()->setHessianMatrix(sparseQ);
    solver_.data()->setLinearConstraintsMatrix(sparseA);
    Eigen::VectorXd constraint = Eigen::VectorXd::Zero(b_x.size());

    solver_.data()->setBounds(constraint, constraint);
    solver_.initSolver();
    solver_.updateBounds(b_x, b_x);

    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "X not solved" << std::endl;
        return false;
    }
    Eigen::VectorXd x_coeffs = solver_.getSolution();
    solver_.updateBounds(b_y, b_y);
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "Y not solved" << std::endl;
        return false;
    }
    Eigen::VectorXd y_coeffs = solver_.getSolution();

    solver_.updateBounds(b_z, b_z);
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "Z not solved" << std::endl;
        return false;
    }
    Eigen::VectorXd z_coeffs = solver_.getSolution();

    solver_.updateBounds(b_yaw, b_yaw);
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "Yaw not solved" << std::endl;
        return false;
    }
    Eigen::VectorXd yaw_coeffs = solver_.getSolution();

    x_polys_.clear();
    y_polys_.clear();
    z_polys_.clear();
    yaw_polys_.clear();
    std::cout << "Total Time: " << total_time_ << std::endl;
    for (int segment = 0; segment < num_segments_; ++segment)
    {
        std::vector<Polynomial> segment_x_polys;
        std::vector<Polynomial> segment_y_polys;
        std::vector<Polynomial> segment_z_polys;
        std::vector<Polynomial> segment_yaw_polys;
        segment_x_polys.push_back(Polynomial(x_coeffs.segment(segment * kCoeffCount, kCoeffCount)));
        std::cout << "Segment " << segment << " poly: " << segment_x_polys.back() << std::endl;

        segment_y_polys.push_back(Polynomial(y_coeffs.segment(segment * kCoeffCount, kCoeffCount)));
        segment_z_polys.push_back(Polynomial(z_coeffs.segment(segment * kCoeffCount, kCoeffCount)));
        segment_yaw_polys.push_back(
            Polynomial(yaw_coeffs.segment(segment * kCoeffCount, kCoeffCount)));
        for (int derivative = 1; derivative < 4; ++derivative)
        {
            // std::cout << "doing derivatives: " << derivative << std::endl;
            segment_x_polys.push_back(segment_x_polys.back().Derivative());
            segment_y_polys.push_back(segment_y_polys.back().Derivative());
            segment_z_polys.push_back(segment_z_polys.back().Derivative());
            segment_yaw_polys.push_back(segment_yaw_polys.back().Derivative());
        }
        x_polys_.push_back(segment_x_polys);
        y_polys_.push_back(segment_y_polys);
        z_polys_.push_back(segment_z_polys);
        yaw_polys_.push_back(segment_yaw_polys);
    }

    solved_ = true;
    return true;
}
void MinSnapTraj::AllocateInitialTimes(Eigen::VectorXd& times, double average_speed)
{
    // double total_distance = 0;
    for (int waypoint_index = 1; waypoint_index < int(waypoints_.size()); ++waypoint_index)
    {
        double distance =
            (waypoints_.at(waypoint_index).pos - waypoints_.at(waypoint_index - 1).pos).norm();
        // total_distance += distance;
        double time = distance;  // / average_speed;
        times(waypoint_index - 1) = time;
    }
    // std::cout << times.transpose().sum() << "\t";

    times *= average_speed / times.sum();
    // std::cout << times.transpose().sum() << "\n";

    total_time_ = times.sum();
}

double MinSnapTraj::CalculateCost(
    const Eigen::VectorXd& b_x,
    const Eigen::VectorXd& b_y,
    const Eigen::VectorXd& b_z,
    const Eigen::VectorXd& b_yaw,
    const Eigen::VectorXd& times,
    double speed_weight)
{
    Eigen::VectorXd upper_bound = Eigen::VectorXd::Zero(b_x.size());
    Eigen::VectorXd lower_bound = Eigen::VectorXd::Zero(b_x.size());
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_variables_);
    Eigen::MatrixXd time_powers;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(num_variables_, num_variables_);
    Eigen::MatrixXd A =
        Eigen::MatrixXd::Zero(2 * num_segments_ + 8 + 4 * num_internal_joints_, num_variables_);
    Eigen::VectorXd time_copy = times;
    CalculateTimePowers(time_copy, time_powers);
    GenerateQ(time_powers, Q);
    GenerateA(time_powers, A);
    Eigen::SparseMatrix<double> sparseA = A.sparseView();
    Eigen::SparseMatrix<double> sparseQ = Q.sparseView();
    upper_bound = b_x;
    lower_bound = b_x;
    solver_.updateHessianMatrix(sparseQ);
    solver_.updateLinearConstraintsMatrix(sparseA);
    solver_.updateBounds(b_x, b_x);
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "x not solved" << std::endl;
        return -1;
    }
    Eigen::VectorXd x_coeffs = solver_.getSolution();
    solver_.updateBounds(b_y, b_y);
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "Y not solved" << std::endl;
        return false;
    }
    Eigen::VectorXd y_coeffs = solver_.getSolution();
    upper_bound = b_z;
    lower_bound = b_z;
    solver_.updateBounds(b_z, b_z);
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "Z not solved" << std::endl;
        return false;
    }
    Eigen::VectorXd z_coeffs = solver_.getSolution();
    upper_bound = b_yaw;
    lower_bound = b_yaw;
    solver_.updateBounds(b_yaw, b_yaw);
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "Yaw not solved" << std::endl;
        return false;
    }
    Eigen::VectorXd yaw_coeffs = solver_.getSolution();
    double x_cost = x_coeffs.transpose() * Q * x_coeffs;
    double y_cost = y_coeffs.transpose() * Q * y_coeffs;
    double z_cost = z_coeffs.transpose() * Q * z_coeffs;
    double yaw_cost = yaw_coeffs.transpose() * Q * yaw_coeffs;
    double time_cost = times.sum() * speed_weight;
    double total_cost = x_cost + y_cost + z_cost + yaw_cost + time_cost;
    // std::cout << "Total " << total_cost << ", xcost: " << x_cost << ", ycost: " << y_cost
    //           << ", zcost: " << z_cost << ", yawcost: " << yaw_cost << "\n";
    return total_cost;
}

void MinSnapTraj::CalculateTimePowers(const Eigen::MatrixXd& times, Eigen::MatrixXd& time_powers)
{
    time_powers = Eigen::MatrixXd::Ones(num_segments_, kCoeffCount);

    for (int segment = 0; segment < num_segments_; ++segment)
    {
        time_powers(segment, 1) = times(segment);
        for (int power = 2; power < kCoeffCount; ++power)
        {
            time_powers(segment, power) = time_powers(segment, power - 1) * times(segment);
        }
    }
}

void MinSnapTraj::GenerateQ(const Eigen::MatrixXd& time_powers, Eigen::MatrixXd& Q)
{
    assert(Q.rows() == num_segments_ * kCoeffCount && Q.cols() == num_segments_ * kCoeffCount);
    for (int segment = 0; segment < num_segments_; ++segment)
    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(kCoeffCount, kCoeffCount);
        FillH(time_powers.row(segment), H);
        int block_location = segment * kCoeffCount;
        Q.block(block_location, block_location, kCoeffCount, kCoeffCount) = H;
    }
}

void MinSnapTraj::FillH(const Eigen::VectorXd& time_powers_row, Eigen::MatrixXd& H)
{
    Eigen::Matrix4d non_zero_h;
    non_zero_h << 1152 * time_powers_row(1), 2880 * time_powers_row(2), 5760 * time_powers_row(3),
        10080 * time_powers_row(4), 2880 * time_powers_row(2), 9600 * time_powers_row(3),
        21600 * time_powers_row(4), 40320 * time_powers_row(5), 5760 * time_powers_row(3),
        21600 * time_powers_row(4), 51840 * time_powers_row(5), 100800 * time_powers_row(6),
        10080 * time_powers_row(4), 40320 * time_powers_row(5), 100800 * time_powers_row(6),
        201600 * time_powers_row(7);

    H.bottomRightCorner(4, 4) = non_zero_h;
    // ⎡0  0  0  0      0          0          0           0     ⎤
    // ⎢                                                        ⎥
    // ⎢0  0  0  0      0          0          0           0     ⎥
    // ⎢                                                        ⎥
    // ⎢0  0  0  0      0          0          0           0     ⎥
    // ⎢                                                        ⎥
    // ⎢0  0  0  0      0          0          0           0     ⎥
    // ⎢                                                        ⎥
    // ⎢                              2           3           4 ⎥
    // ⎢0  0  0  0   1152⋅tf   2880⋅tf     5760⋅tf    10080⋅tf  ⎥
    // ⎢                                                        ⎥
    // ⎢                   2          3           4           5 ⎥
    // ⎢0  0  0  0  2880⋅tf    9600⋅tf    21600⋅tf    40320⋅tf  ⎥
    // ⎢                                                        ⎥
    // ⎢                   3           4          5            6⎥
    // ⎢0  0  0  0  5760⋅tf    21600⋅tf   51840⋅tf    100800⋅tf ⎥
    // ⎢                                                        ⎥
    // ⎢                    4          5           6           7⎥
    // ⎣0  0  0  0  10080⋅tf   40320⋅tf   100800⋅tf   201600⋅tf ⎦
}
void MinSnapTraj::GenerateA(const Eigen::MatrixXd& time_powers, Eigen::MatrixXd& A)
{
    assert(A.cols() == kCoeffCount * num_segments_);

    assert(A.rows() == 2 * num_segments_ + 8 + 4 * num_internal_joints_);
    Eigen::VectorXd ZeroTimes = Eigen::VectorXd::Zero(kCoeffCount);
    int constraint_count = 0;
    for (int segment = 0; segment < num_segments_; ++segment)
    {
        // start time
        A(constraint_count, kCoeffCount * segment) = 1;
        ++constraint_count;
        // end time
        Eigen::RowVectorXd constraint_row(kCoeffCount);
        CalculatePolyDerivativeMultipliers(
            kCoeffCount, 0, time_powers.row(segment), constraint_row);

        A.block(constraint_count, kCoeffCount * segment, 1, 8) = constraint_row;
        ++constraint_count;
    }

    // start point dervatives. No setting of target velocity and acceleration yet. In the future
    // start could be the current one.
    for (int derivative = 1; derivative <= kPosMinDerivative; ++derivative)
    {
        A(constraint_count, derivative) = 1;
        ++constraint_count;
    }
    // end point derivatives. No setting of target velocity and acceleration yet. In the future end
    // could a desired velocity
    for (int derivative = 1; derivative <= kPosMinDerivative; ++derivative)
    {
        Eigen::RowVectorXd constraint_row(kCoeffCount);
        CalculatePolyDerivativeMultipliers(
            kCoeffCount, derivative, time_powers.row(num_segments_ - 1), constraint_row);
        A.block(constraint_count, kCoeffCount * (num_segments_ - 1), 1, 8) = constraint_row;
        ++constraint_count;
    }

    // Derivative continuities
    for (int joint = 0; joint < num_internal_joints_; ++joint)
    {
        // continuity between segment = joint and segment = joint + 1
        // zeroth derivative of
        for (int derivative = 1; derivative <= kPosMinDerivative; ++derivative)
        {
            // end of first segment
            Eigen::RowVectorXd constraint_row(kCoeffCount);
            CalculatePolyDerivativeMultipliers(
                kCoeffCount, derivative, time_powers.row(joint), constraint_row);
            A.block(constraint_count, kCoeffCount * joint, 1, 8) = constraint_row;
            // start of second segment
            A(constraint_count, kCoeffCount * (joint + 1) + derivative) =
                -CalculatePolyCoeffMultiplier(derivative, derivative);
            ++constraint_count;
        }
    }
}
void MinSnapTraj::GenerateB(
    Eigen::VectorXd& b_x,
    Eigen::VectorXd& b_y,
    Eigen::VectorXd& b_z,
    Eigen::VectorXd& b_yaw)
{
    assert(b_x.size() == 2 * num_segments_ + 8 + 4 * num_internal_joints_);
    assert(b_y.size() == 2 * num_segments_ + 8 + 4 * num_internal_joints_);
    assert(b_z.size() == 2 * num_segments_ + 8 + 4 * num_internal_joints_);
    assert(b_yaw.size() == 2 * num_segments_ + 8 + 4 * num_internal_joints_);
    b_x = Eigen::VectorXd::Zero(b_x.rows());
    b_y = Eigen::VectorXd::Zero(b_y.rows());
    b_z = Eigen::VectorXd::Zero(b_z.rows());
    b_yaw = Eigen::VectorXd::Zero(b_yaw.rows());
    int constraint_count = 0;  // Each row in A is a constraint. If we just do each constraint 1 by
                               // 1 and increment this we won't get lost.

    // waypoints first
    for (int segment = 0; segment < num_segments_; ++segment)
    {
        // start time
        b_x(constraint_count) = waypoints_[segment].pos(0);
        b_y(constraint_count) = waypoints_[segment].pos(1);
        b_z(constraint_count) = waypoints_[segment].pos(2);
        b_yaw(constraint_count) = waypoints_[segment].yaw;
        ++constraint_count;
        // end time
        b_x(constraint_count) = waypoints_[segment + 1].pos(0);
        b_y(constraint_count) = waypoints_[segment + 1].pos(1);
        b_z(constraint_count) = waypoints_[segment + 1].pos(2);
        b_yaw(constraint_count) = waypoints_[segment + 1].yaw;
        ++constraint_count;
    }
    // startpoint derivatives
    std::cout << "setting b start point derivatives" << std::endl;
    for (int derivative = 1; derivative <= kPosMinDerivative; ++derivative)
    {
        b_x(constraint_count) = start_derivatives_[derivative - 1].pos.x();
        b_y(constraint_count) = start_derivatives_[derivative - 1].pos.y();
        b_z(constraint_count) = start_derivatives_[derivative - 1].pos.z();
        b_yaw(constraint_count) = start_derivatives_[derivative - 1].yaw;
        ++constraint_count;
    }
    // endpoint derivatives
    std::cout << "setting b end point derivatives" << std::endl;

    for (int derivative = 1; derivative <= kPosMinDerivative; ++derivative)
    {
        std::cout << end_derivatives_[derivative - 1].pos.x() << std::endl;
        b_x(constraint_count) = end_derivatives_[derivative - 1].pos.x();
        b_y(constraint_count) = end_derivatives_[derivative - 1].pos.y();
        b_z(constraint_count) = end_derivatives_[derivative - 1].pos.z();
        b_yaw(constraint_count) = end_derivatives_[derivative - 1].yaw;
        ++constraint_count;
    }
}

int MinSnapTraj::CalculatePolyCoeffMultiplier(int coeff, int derivative_count)
{
    int multiplier = 1;
    for (int derivative = 0; derivative < derivative_count; ++derivative)
    {
        multiplier *= coeff;
        if (multiplier == 0)
        {
            break;
        }
        --coeff;
    }  // coeff is the power that we need to raise T by.
    return multiplier;
}

int MinSnapTraj::CalculatePolyCoeffPower(int coeff, int derivative_count)
{
    int power_val = coeff - derivative_count;
    if (power_val < 0)
    {
        power_val = 0;
    }
    return power_val;
}

void MinSnapTraj::CalculatePolyDerivativeMultipliers(
    const int coeff_count,
    const int derivative_count,
    const Eigen::VectorXd& time_powers_row,
    Eigen::RowVectorXd& polynomial_derivative)
{
    assert(polynomial_derivative.cols() == coeff_count);
    for (int coeff = 0; coeff < coeff_count; ++coeff)
    {
        polynomial_derivative(coeff) =
            CalculatePolyCoeffMultiplier(coeff, derivative_count) *
            time_powers_row(CalculatePolyCoeffPower(coeff, derivative_count));
    }
}
