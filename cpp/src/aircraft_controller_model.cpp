#include "aircraft_controller_model.hpp"

AircraftControllerModel::AircraftControllerModel(
    AircraftModel full_params,
    double elevon_lift_coeff_airspeed_input,
    double elevon_lift_coeff_thrust_input,
    double pitch_coeff_thrust_input)
{
    mass = full_params.mass;
    moment_of_inertia = full_params.moment_of_inertia;
    moment_of_inertia_inv = full_params.moment_of_inertia_inv;
    prop_thrust_coeff = full_params.prop_thrust_coeff * 1.225;
    prop_moment_coeff = full_params.prop_moment_coeff * 1.225;
    prop_max_omega = full_params.prop_max_omega;
    wing_lift_coeff = full_params.maximum_drag_coeff * 1.225 * 0.5 * full_params.wing_surface_area;
    wing_drag_coeff = full_params.minimum_drag_coeff * 1.225 * 0.5 * full_params.wing_surface_area;
    wing_lift_coeff_thrust = 0;
    wing_drag_coeff_thrust = 0;
    elevon_lift_coeff_airspeed = elevon_lift_coeff_airspeed_input;
    elevon_lift_coeff_thrust = elevon_lift_coeff_thrust_input;
    pitch_coeff_thrust = pitch_coeff_thrust_input;
    max_elevon_angle = full_params.max_elevon_angle;
    double elevon_chord = full_params.chord * full_params.elevon_percentage;
    double chord_no_elevon = full_params.chord - elevon_chord;
    thrust_l_y = abs(full_params.aerodynamic_moment_arm_left[1]);
    flap_length_x =
        abs(chord_no_elevon + 0.5 * elevon_chord - 0.25 * full_params.chord + full_params.delta_r);
    flap_length_y = abs(thrust_l_y);
}