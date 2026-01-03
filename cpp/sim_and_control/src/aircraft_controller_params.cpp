#include "aircraft_controller_params.hpp"

AircraftControllerParameters AircraftControllerParameters::GetControllerParameters(
    std::string param_file,
    AircraftModel full_params,
    EnvironmentParameters environment_params)
{
    toml::table tbl;
    try
    {
        using namespace TomlParseHelpers;

        tbl = toml::parse_file(param_file);
        double elevon_lift_coeff_airspeed = ParseDouble(tbl, "elevon_lift_coeff_airspeed");
        double elevon_lift_coeff_thrust = ParseDouble(tbl, "elevon_lift_coeff_thrust");
        double pitch_coeff_thrust = ParseDouble(tbl, "pitch_coeff_thrust");
        double lift_coeff_thrust = ParseDouble(tbl, "lift_coeff_thrust");

        double controller_frequency = ParseDouble(tbl, "controller_frequency");
        Eigen::Matrix3d pos_gain = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d vel_gain = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d acc_gain = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d quat_gain = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d omega_gain = Eigen::Matrix3d::Zero();
        Eigen::Vector3d pos_gain_diag = ParseDoubleVector3d(tbl, "pos_gain");
        Eigen::Vector3d vel_gain_diag = ParseDoubleVector3d(tbl, "vel_gain");
        Eigen::Vector3d acc_gain_diag = ParseDoubleVector3d(tbl, "acc_gain");
        Eigen::Vector3d quat_gain_diag = ParseDoubleVector3d(tbl, "quat_gain");
        Eigen::Vector3d omega_gain_diag = ParseDoubleVector3d(tbl, "omega_gain");
        pos_gain.diagonal() = pos_gain_diag;
        vel_gain.diagonal() = vel_gain_diag;
        acc_gain.diagonal() = acc_gain_diag;
        quat_gain.diagonal() = quat_gain_diag;
        omega_gain.diagonal() = omega_gain_diag;

        AircraftControllerModel model_params(
            full_params, elevon_lift_coeff_airspeed, elevon_lift_coeff_thrust, pitch_coeff_thrust);

        AircraftControllerParameters new_params = {
            model_params = model_params,
            environment_params = environment_params,
            pos_gain = pos_gain,
            vel_gain = vel_gain,
            acc_gain = acc_gain,
            quat_gain = quat_gain,
            omega_gain = omega_gain,
            controller_frequency = controller_frequency};
        return new_params;
    } catch (const toml::parse_error& err)
    {
        std::cerr << "Parsing failed:\n" << err << "\n";
        exit(0);
    }
}