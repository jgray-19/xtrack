// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_KICK_H
#define XTRACK_TRACK_MAGNET_KICK_H

#include "xtrack/headers/track.h"


GPUFUN
void kick_simple_single_particle(
    LocalParticle* part,
    int64_t order,
    double inv_factorial,
    const double* knl,
    const double* ksl,
    double factor,
    double kick_weight
);


GPUFUN
void track_magnet_kick_path_correction_single_particle_h0(
    LocalParticle* part, double length
) {
    if (length == 0.0) return;
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double one_plus_delta = 1.0 + LocalParticle_get_delta(part);
    const double inv_p = LocalParticle_get_rpp(part);  // == 1/(1 + delta)
    const double rv0v = 1.0 / LocalParticle_get_rvv(part);
    const double p_perp2 = px * px + py * py;
    const double one_over_pz = 1.0 / sqrt(one_plus_delta * one_plus_delta - p_perp2);
    const double d_length = length * (one_over_pz - inv_p);

    LocalParticle_add_to_x(part, px * d_length);
    LocalParticle_add_to_y(part, py * d_length);
    LocalParticle_add_to_zeta(part, rv0v * (
        0.5 * p_perp2 * length * inv_p * inv_p - one_plus_delta * d_length));
}


GPUFUN
void track_magnet_kick_single_particle(
    LocalParticle* part,
    double length,
    int64_t order,
    double inv_factorial_order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    int64_t order_rel,
    double inv_factorial_order_rel,
    GPUGLMEM const double* knl_rel,
    GPUGLMEM const double* ksl_rel,
    double rel_ref_strength,
    double const factor_knl_ksl,
    double kick_weight,
    double k0,
    double k1,
    double k2,
    double k3,
    double k0s,
    double k1s,
    double k2s,
    double k3s,
    double h,
    double hxl,
    double k0_h_correction,
    double k1_h_correction,
    uint8_t rot_frame,
    int8_t drift_model,
    uint8_t kick_is_empty
){

    double const length_of_this_kick = length * kick_weight;

    if (kick_is_empty) {
        // Every field this function would apply is exactly zero (the caller
        // checked), so the multipole kicks and the curvature corrections
        // below are all no-ops and are skipped outright.
        //
        // For mat-kick-mat-exact that leaves nothing between the two path
        // corrections, and the correction is a momentum-only map, so
        // B(l/2) B(l/2) = B(l) and a single call does the job.
        if (drift_model == 9) {
            track_magnet_kick_path_correction_single_particle_h0(
                part, length_of_this_kick);
        }
        return;
    }

    double const chi = LocalParticle_get_chi(part);
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);

    if (drift_model == 9) {
        track_magnet_kick_path_correction_single_particle_h0(
            part, 0.5 * length_of_this_kick);
    }

    double knl_main[4] = {k0, k1, k2, k3};
    double ksl_main[4] = {k0s, k1s, k2s, k3s};

    for (int index = 0; index < 4; index++) {
        knl_main[index] = knl_main[index] * length;
        ksl_main[index] = ksl_main[index] * length;
    }

    // multipolar kick
    kick_simple_single_particle(
        part,
        order,
        inv_factorial_order,
        knl,
        ksl,
        factor_knl_ksl,
        kick_weight
    );

    // multipolar kick
    kick_simple_single_particle(
        part,
        order_rel,
        inv_factorial_order_rel,
        knl_rel,
        ksl_rel,
        factor_knl_ksl * rel_ref_strength,
        kick_weight
    );

    kick_simple_single_particle(
        part,
        /* order */ 3,
        /* inv_factorial_order */ 1. / (3 * 2),
        knl_main,
        ksl_main,
        /* factor_knl_ksl */ 1,
        kick_weight
    );

    // Correct for the curvature
    double dpx = 0;
    double dpy = 0;
    double dzeta = 0;

    if (rot_frame) {
        double const hl = h * length * kick_weight + hxl * kick_weight;
        dpx += hl * (1. + LocalParticle_get_delta(part));
        double const rv0v = 1./LocalParticle_get_rvv(part);
        dzeta += -rv0v * hl * x;
    }

    double htot = h;
    if (length != 0) {
        htot += hxl / length;
    }

    // Correct for the curvature
    // k0h correction can be computed from this term in the hamiltonian
    // H = 1/2 h k0 x^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)
    double k0l_mult = 0;
    if (order >= 0) {
        k0l_mult = knl[0] * factor_knl_ksl;
    }
    if (order_rel >= 0 && knl_rel != NULL) {
        k0l_mult += knl_rel[0] * factor_knl_ksl * rel_ref_strength;
    }
    dpx += -chi * (k0_h_correction  *length + k0l_mult) * kick_weight * htot * x;

    // k1h correction can be computed from this term in the hamiltonian
    // H = 1/3 hk1 x^3 - 1/2 hk1 xy^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)
    double k1l_mult = 0;
    if (order >= 1) {
        k1l_mult = knl[1] * factor_knl_ksl;
    }
    if (order_rel >= 1 && knl_rel != NULL) {
        k1l_mult += knl_rel[1] * factor_knl_ksl * rel_ref_strength;
    }
    dpx += htot * chi * (k1_h_correction * length + k1l_mult) * kick_weight * (-x * x + 0.5 * y * y);
    dpy += htot * chi * (k1_h_correction * length  + k1l_mult) * kick_weight * x * y;

    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_py(part, dpy);
    LocalParticle_add_to_zeta(part, dzeta);

    if (drift_model == 9) {
        track_magnet_kick_path_correction_single_particle_h0(
            part, 0.5 * length_of_this_kick);
    }

}



GPUFUN
uint8_t kick_is_inactive(
    int64_t order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    double k0,
    double k1,
    double k2,
    double k3,
    double k0s,
    double k1s,
    double k2s,
    double k3s,
    double h
){
    if (h != 0) return 0;
    if (k0 != 0) return 0;
    if (k1 != 0) return 0;
    if (k2 != 0) return 0;
    if (k3 != 0) return 0;
    if (k0s != 0) return 0;
    if (k1s != 0) return 0;
    if (k2s != 0) return 0;
    if (k3s != 0) return 0;

    for (int index = order; index >= 0; index--) {
        if (knl[index] != 0) return 0;
        if (ksl[index] != 0) return 0;
    }

    return 1;

}

GPUFUN
void kick_simple_single_coordinates(
    double const x,
    double const y,
    double const chi,
    int64_t order,
    double inv_factorial,
    const double* knl,
    const double* ksl,
    double factor,
    double kick_weight,
    double *dpx,
    double *dpy
) {

    // Return if null knl/ksl pointers
    if (knl == NULL || ksl == NULL) {
        *dpx = 0.;
        *dpy = 0.;
        return;
    }

    int64_t index = order;

    double dpx_mul = chi * knl[index] * factor * inv_factorial;
    double dpy_mul = chi * ksl[index] * factor * inv_factorial;

    while( index > 0 )
    {
        double const zre = dpx_mul * x - dpy_mul * y;
        double const zim = dpx_mul * y + dpy_mul * x;

        inv_factorial *= index;
        index -= 1;

        double this_knl = chi * knl[index] * factor;
        double this_ksl = chi * ksl[index] * factor;

        dpx_mul = this_knl * inv_factorial + zre;
        dpy_mul = this_ksl * inv_factorial + zim;
    }

    dpx_mul = -dpx_mul; // rad

    *dpx = kick_weight * dpx_mul;
    *dpy = kick_weight * dpy_mul;
}


GPUFUN
void kick_simple_single_particle(
    LocalParticle* part,
    int64_t order,
    double inv_factorial,
    const double* knl,
    const double* ksl,
    double factor,
    double kick_weight
) {
    double const chi = LocalParticle_get_chi(part);
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);

    double dpx, dpy;

    kick_simple_single_coordinates(
        x,
        y,
        chi,
        order,
        inv_factorial,
        knl,
        ksl,
        factor,
        kick_weight,
        &dpx,
        &dpy);

    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_py(part, dpy);
}

GPUFUN
void evaluate_field_from_strengths(
    double const p0c,
    double const q0,
    double const x,
    double const y,
    double length,
    int64_t order,
    double inv_factorial_order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    int64_t order_rel,
    double inv_factorial_order_rel,
    GPUGLMEM const double* knl_rel,
    GPUGLMEM const double* ksl_rel,
    double rel_ref_strength,
    double const factor_knl_ksl,
    double k0,
    double k1,
    double k2,
    double k3,
    double k0s,
    double k1s,
    double k2s,
    double k3s,
    double ks,
    double dks_ds,
    double x0_solenoid,
    double y0_solenoid,
    double *Bx_T,
    double *By_T,
    double *Bz_T
){
    if (length == 0.0) {
        *Bx_T = 0.0;
        *By_T = 0.0;
        *Bz_T = 0.0;
        return;
    }

    double knl_main[4] = {k0, k1, k2, k3};
    double ksl_main[4] = {k0s, k1s, k2s, k3s};

    for (int index = 0; index < 4; index++) {
        knl_main[index] = knl_main[index] * length;
        ksl_main[index] = ksl_main[index] * length;
    }

    // multipolar kick
    double dpx_mul = 0.;
    double dpy_mul = 0.;
    kick_simple_single_coordinates(
        x,
        y,
        1., // chi
        order,
        inv_factorial_order,
        knl,
        ksl,
        factor_knl_ksl,
        1., // kick_weight
        &dpx_mul,
        &dpy_mul);

    // multipolar kick relevant for the field evaluation
    double dpx_mul_rel = 0.;
    double dpy_mul_rel = 0.;
    kick_simple_single_coordinates(
        x,
        y,
        1., // chi
        order_rel,
        inv_factorial_order_rel,
        knl_rel,
        ksl_rel,
        factor_knl_ksl * rel_ref_strength,
        1., // kick_weight
        &dpx_mul_rel,
        &dpy_mul_rel);


    // main kick
    double dpx_main=0.;
    double dpy_main=0.;
    kick_simple_single_coordinates(
        x,
        y,
        1., // chi
        3, // order
        1. / (3 * 2), //inv_factorial_order
        knl_main,
        ksl_main,
        1, // factor_knl_ksl,
        1., // kick_weight
        &dpx_main,
        &dpy_main);

    double const dpx = dpx_mul + dpx_main + dpx_mul_rel;
    double const dpy = dpy_mul + dpy_main + dpy_mul_rel;

    double const brho_0 = p0c / C_LIGHT / q0; // [T m]

    *Bx_T = dpy * brho_0 / length - 0.5 * dks_ds * brho_0 * (x - x0_solenoid); // [T]
    *By_T = -dpx * brho_0 / length - 0.5 * dks_ds * brho_0 * (y - y0_solenoid); // [T]
    *Bz_T = ks * brho_0; // [T]

}

#endif