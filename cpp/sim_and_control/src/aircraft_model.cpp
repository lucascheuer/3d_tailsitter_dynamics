#include "aircraft_model.hpp"

#include "helpers.hpp"
AircraftModel::AircraftModel(std::string filename)
{
    toml::table tbl;
    try
    {
        using namespace TomlParseHelpers;
        tbl = toml::parse_file(filename);
        mass = ParseDouble(tbl, "mass");
        prop_thrust_coeff = ParseDouble(tbl, "prop_thrust_coeff");
        prop_moment_coeff = ParseDouble(tbl, "prop_moment_coeff");
        prop_max_omega = ParseDouble(tbl, "prop_max_omega");
        prop_max_omega_dot = ParseDouble(tbl, "prop_max_omega_dot");
        double propeller_diameter = ParseDouble(tbl, "propeller_diameter");
        propeller_disc_area = M_PI * pow((propeller_diameter * 0.5), 2);
        minimum_drag_coeff = ParseDouble(tbl, "minimum_drag_coeff");
        maximum_drag_coeff = ParseDouble(tbl, "maximum_drag_coeff");
        roll_rate_drag_coeff = ParseDouble(tbl, "roll_rate_drag_coeff");
        pitch_rate_drag_coeff = ParseDouble(tbl, "pitch_rate_drag_coeff");
        yaw_rate_drag_coeff = ParseDouble(tbl, "yaw_rate_drag_coeff");
        max_elevon_angle = ParseDouble(tbl, "max_elevon_angle");
        max_elevon_angle_dot = ParseDouble(tbl, "max_elevon_angle_dot");
        wingspan = ParseDouble(tbl, "wingspan");
        chord = ParseDouble(tbl, "chord");
        elevon_percentage = ParseDouble(tbl, "elevon_percentage");
        wing_surface_area = wingspan * chord;
        double depth = ParseDouble(tbl, "depth");
        delta_r = ParseDouble(tbl, "delta_r");
        linear_elevon_effectiveness = ParseDoubleVector3d(tbl, "linear_elevon_effectiveness");
        rotational_elevon_effectiveness =
            ParseDoubleVector3d(tbl, "rotational_elevon_effectiveness");
        propeller_moment_arm_left = ParseDoubleVector3d(tbl, "propeller_moment_arm_left");
        propeller_moment_arm_right = ParseDoubleVector3d(tbl, "propeller_moment_arm_right");

        moment_of_inertia(0, 0) = 1.0 / 12.0 * mass * (pow(wingspan, 2) + pow(chord, 2));
        moment_of_inertia(1, 1) = 1.0 / 12.0 * mass * (pow(depth, 2) + pow(chord, 2));
        moment_of_inertia(2, 2) = 1.0 / 12.0 * mass * (pow(depth, 2) + pow(wingspan, 2));
        moment_of_inertia_inv = moment_of_inertia.inverse();
        B(0, 0) = wingspan;
        B(1, 1) = chord;
        B(2, 2) = wingspan;
        phi_m_omega(0, 0) = roll_rate_drag_coeff;
        phi_m_omega(1, 1) = pitch_rate_drag_coeff;
        phi_m_omega(2, 2) = yaw_rate_drag_coeff;
        phi_f_v(0, 0) = minimum_drag_coeff;
        phi_f_v(1, 1) = maximum_drag_coeff;
        phi_f_v(2, 2) = M_PI * 2 + minimum_drag_coeff;

        phi_m_v(1, 2) = -(1 / chord) * delta_r * (M_PI * 2 + minimum_drag_coeff);
        phi_m_v(2, 1) = (1 / wingspan) * delta_r * maximum_drag_coeff;
        aerodynamic_moment_arm_left << -delta_r, -wingspan * 0.25, 0.0;
        aerodynamic_moment_arm_right << -delta_r, wingspan * 0.25, 0.0;
    } catch (const toml::parse_error& err)
    {
        std::cerr << "Parsing failed:\n" << err << "\n";
        mass = 0;
        moment_of_inertia = Eigen::Matrix3d::Zero();
    }
}
