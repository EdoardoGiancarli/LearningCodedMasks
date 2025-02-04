"""
@Title: Balancing for Sky Reconstruction
@Author: Edoardo Giancarli
@Date: 31/12/24
@Content:
    - BalancedSkyReconstruction_Peppe: Performs the cross-corelation balanced sky reconstruction.
    - BalancedSkyReconstruction_Goldwurm: Performs the cross-corelation balanced sky reconstruction.
@References:
    [1] A. Goldwurm and A. Gros, "Coded Mask Instruments for Gamma-Ray Astronomy", 2023
        (in "Handbook of X-ray and Gamma-ray Astrophysics", Springer 2023)
    [2] Notebook by Peppe (url: https://github.com/peppedilillo/masks/blob/main/notebooks/11_pcfov.ipynb)
"""

import collections.abc as c
import numpy as np
from scipy.signal import correlate


class BalancedSkyReconstruction_Peppe:
    def __init__(self, decoder, bulk):
        self.decoder = decoder
        self.bulk = bulk

        self.n, self.m = bulk.shape
        self.h, self.v = self.n - 1, self.m - 1

    def balanced_sky_reconstruction(self, detector_image):
        return self._skyrec(detector_image) - self._balancing_array(detector_image)

    def _skyrec(self, detector_image):
        skyrec = correlate(self.decoder, detector_image, mode='full')
        assert skyrec.shape == (self.n + 2*self.h, self.m + 2*self.v)
        return skyrec

    def _balancing_array(self, detector_image):
        balancing_array = correlate(self.decoder, self.bulk, mode='full')
        balancing_array *= (detector_image.sum()/self.bulk.sum())
        assert balancing_array.shape == (self.n + 2*self.h, self.m + 2*self.v)
        return balancing_array 



class BalancedSkyReconstruction_Goldwurm:
    def __init__(self, mask, decoder, bulk):
        self.mask = mask
        self.decoder = decoder
        self.bulk = bulk

        self.n, self.m = bulk.shape
        self.h, self.v = self.n - 1, self.m - 1
    
    def balanced_sky_reconstruction(self, detector_image, var=True):

        G_plus, G_minus = self._get_decoder_terms()
        B = self._get_balancing_array(G_plus, G_minus)
        N = self._get_normalization(G_plus, G_minus, B)

        gplus_cc_term = self._cc(G_plus, detector_image)
        gminus_cc_term = self._cc(G_minus, detector_image)
        S_hat = (gplus_cc_term - B*gminus_cc_term)/N

        if var:
            gplus_ccvar_term = self._cc_var(G_plus, detector_image)
            gminus_ccvar_term = self._cc_var(G_minus, detector_image)
            S_hat_var = (gplus_ccvar_term + (B**2)*gminus_ccvar_term)/(N**2)
        else:
            S_hat_var = None
        
        return S_hat, S_hat_var
    
    def _get_decoder_terms(self):
        G_plus = self.decoder.copy()
        G_plus[self.decoder < 0] = 0
        G_minus = self.decoder.copy()
        G_minus[self.decoder > 0] = 0
        assert np.all(G_plus + G_minus == self.decoder)
        return G_plus, G_minus

    def _get_balancing_array(self, G_plus, G_minus):
        num = correlate(G_plus, self.bulk, mode=self._cc_mode)
        den = correlate(G_minus, self.bulk, mode=self._cc_mode)
        assert num.shape == (self.n + 2*self.h, self.m + 2*self.v)
        assert den.shape == (self.n + 2*self.h, self.m + 2*self.v)
        return num/den

    def _get_normalization(self, G_plus, G_minus, B):
        gplus_norm = correlate(G_plus*self.mask, self.bulk, mode=self._cc_mode)
        if np.all(G_minus*self.mask != np.zeros(G_minus.shape)):                            # for URA/MURA G_minus * mask = 0
            gminus_norm = B*correlate(G_minus*self.mask, self.bulk, mode=self._cc_mode)
        else:
            gminus_norm = np.zeros(gplus_norm.shape)
        assert gplus_norm.shape == (self.n + 2*self.h, self.m + 2*self.v)
        assert gplus_norm.shape == gminus_norm.shape
        return gplus_norm - gminus_norm

    def _cc(self, g, detector_image):
        cc = correlate(g, detector_image*self.bulk, mode=self._cc_mode)
        assert cc.shape == (self.n + 2*self.h, self.m + 2*self.v)
        return cc
    
    def _cc_var(self, g, detector_image):
        cc = correlate(g**2, detector_image*(self.bulk**2), mode=self._cc_mode)
        assert cc.shape == (self.n + 2*self.h, self.m + 2*self.v)
        return cc

    @property
    def _cc_mode(self):
        return 'full'


# end