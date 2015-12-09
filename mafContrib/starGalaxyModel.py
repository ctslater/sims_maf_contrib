
# sims_maf_contrib version

import numpy as np

class StarGalaxyModel:
    def cumulative_log_SG_ratio(self, galactic_lat, i_mag):
        """Returns the log of the ratio of stars to galaxies for
        all sources brighter than i_mag. This is only used for normalizing the stellar
        counts, and I'd like to get rid of it entirely.

        galactic_lat is in degrees."""
        A = -1.3 - 0.28*(i_mag - 25) - 0.0025*(i_mag - 25.0)**2
        B = 2.3
        C = 20.0 # Degrees
        return A + B*np.exp(-abs(galactic_lat)/C)

    def star_counts(self, galactic_lat, i_mag):
        """Returns stars per square degree per magnitude.

        Takes the cumulative_log_SG_ratio at i=21 as a normalization,
        then assumes that counts are flat per magnitude (r**-3 halo profile).
        """
        norm_magnitude = 21
        normalization_SG = self.cumulative_log_SG_ratio(galactic_lat, norm_magnitude)
        # Ngal is per square degree per magnitude
        Ngal = self.galaxy_counts(norm_magnitude)
        Nstar_norm = Ngal*10**(normalization_SG)

        # No magnitude dependence, but we want the shape of i_mag
        return Nstar_norm * (i_mag**0)

    def galaxy_counts(self, i_mag):
        """Galaxy counts per square degree per magnitude"""
        # Old cumulative version, equation C2
        #mag_func = 166000 * 10**(0.31*(i_mag - 25.0))

        Ngal = 166000*10**(0.31*(i_mag - 25.0))*np.log(10)*0.31
        return Ngal

    def galaxy_density_per_size_mag(self, theta, i_mag):
        """Returns the number of galaxies per square degree with a given size
        and a given i-band magnitude.

        Note that theta needs to extend up to ~3 arcsec, otherwise when summing over
        the size distribution, significant numbers of bright galaxies will "disappear"
        since they are beyond that size.
        """
        size_lognorm_center = -0.24*i_mag + 5.02
        size_lognorm_sigma = -0.0136*i_mag + 0.778

        coeff = 1/(theta*size_lognorm_sigma*np.sqrt(2*np.pi))
        size_func = np.exp(-(np.log(theta)-size_lognorm_center)**2/(2*size_lognorm_sigma**2))

        mag_func = self.galaxy_counts(i_mag)

        return coeff*size_func*mag_func

    def galaxy_size_mag_grid(self, theta_arr, i_mag_arr):
        """Computes a grid of galaxy densities over size and magnitude by evaluating
        galaxy_density_per_size_mag() over the input theta and i_mag arrays.
        """
        ii, thetatheta = np.meshgrid(i_mag_arr, theta_arr)
        galaxy_density_map = self.galaxy_density_per_size_mag(thetatheta, ii)
        galaxy_density_map *= abs(i_mag_arr[0] - i_mag_arr[1]) * (theta_arr[1] - theta_arr[0])
        return galaxy_density_map

    def C_from_SNR(self, SNR, theta_ratio=1.0):
        # This effective SNR scaling comes from Equation C5
        eff_SNR = SNR/theta_ratio**1.6

        fit_result = 0.4767 + 0.07909*eff_SNR - 0.00313*eff_SNR**2
        out = 1.0 * (eff_SNR > 12)
        out += fit_result * (eff_SNR <= 12) * (eff_SNR >= 1)
        out += 0.0 * (eff_SNR < 1) # For completeness

        return out

    def new_C_from_SNR(self, SNR, theta_ratio=1.0):
        """theta_ratio = alpha/alpha_g"""
        # This is a new scaling for eff_SNR
        eff_SNR = SNR/theta_ratio**2.0
        A_fixed = 95.0/100.0
        B_fixed = -1/2.5

        completeness = 1 - A_fixed*np.exp(B_fixed*eff_SNR)
        return completeness

    def SNR_from_m5(self, m5, mag, gamma=0.038):
        fluxRatio = 10**(-0.4*(m5 - mag))
        noiseSq = (0.04-gamma)*fluxRatio + gamma*fluxRatio**2
        return 1/np.sqrt(noiseSq)

    def efficiency_map(self, theta_g, i_mag, m5=24.0, theta_obs=1.0):
        theta_ratio = theta_obs / theta_g
        SNR = self.SNR_from_m5(m5, i_mag)
        efficiency_C = self.new_C_from_SNR(SNR, theta_ratio=theta_ratio)
        return efficiency_C

    def make_differential_contamination(self, i_mags, m5=25, theta_obs=0.7, galactic_lat=45):
        theta_gs = np.linspace(0.1, 3.0, 40)
        ii, thetatheta = np.meshgrid(i_mags, theta_gs)
        eff_map_arr = self.efficiency_map(thetatheta, ii, m5=m5, theta_obs=theta_obs)

        galaxy_density_map = self.galaxy_size_mag_grid(theta_gs, i_mags)
        contamination_map = (1 - eff_map_arr)*galaxy_density_map
        differential_contaminants = contamination_map.sum(axis=0)

        d_mag = abs(i_mags[1] - i_mags[0])
        Nstars = self.star_counts(galactic_lat, i_mags)*d_mag
        return differential_contaminants/Nstars

