"""
@Title: Coded Mask - Sky Image Reconstruction Algorithms
@Author: Edoardo Giancarli
@Date: 7/01/25
@Content:
    - IROS: Iterative Removal of Sources algorithm (dummy).
@References:
    [1] 
    [2] 
    [3] 
"""

import numpy as np
from scipy.signal import correlate
from scipy.stats import norm
import matplotlib.pyplot as plt
import plot_module as plot


class _support:

    def shadowgram(self,
                   pos: tuple,
                   counts: int) -> np.array:
        
        s = np.zeros(self.sky_shape)
        s[*pos] = counts
        s_d_img = correlate(self.mask, s)[2*self.h : -2*self.h, 2*self.v : -2*self.v]

        assert s_d_img.shape == (self.h + 1, self.v + 1)

        return s_d_img
    
    def iros_application(self,
                         detector_image: np.array,
                         shadowgram: np.array
                         ) -> tuple[np.array, np.array, np.array]:
        
        new_detector = detector_image - shadowgram
        new_skyrec, new_skyrec_var = self.bal.balanced_sky_reconstruction(new_detector)
        new_snr = np.nan_to_num(new_skyrec/np.sqrt(new_skyrec_var + 1e-8))

        return new_detector, new_skyrec, new_snr
    
    def _select_source(self,
                       snr: np.array,
                       iteration: int,
                       show_peaks: bool) -> np.array:

        peaks_pos = np.argwhere(snr > self.threshold).T
        _n_peaks = len(peaks_pos[0])

        print(
            f"Number of outliers with SNR(σ) over {self.threshold} at iteration {iteration}: {_n_peaks}"
            )
        
        if show_peaks:
            plot.image_plot([snr],
                            [f"SkyRec Peaks with SNR $\geq$ {self.threshold}"],
                            cbarlabel=["SNR"],
                            cbarcmap=[self._cmap],
                            simulated_sources=[np.dstack((peaks_pos[0], peaks_pos[1]))[0]])
        
        loc = np.argwhere(snr == snr.max()).T

        assert snr[*loc] == snr.max()
        
        return loc, _n_peaks
    
    def _check_peaks(self, n: int) -> bool:
        check = n != 0
        return check
    
    def _record_source(self,
                       sources_log: dict,
                       pos: np.array,
                       counts: int | float) -> dict:
        
        sources_log['sources_pos'].append(pos.T[0])
        sources_log['sources_counts'].append(counts[0])

        return sources_log
    
    def _show_results(self,
                      skyrec: np.array,
                      snr: np.array,
                      iteration: int) -> None:

        # remove artifacts
        post_skyrec = skyrec.copy()
        post_skyrec[skyrec > self.vis_thres] = 0
        post_skyrec[skyrec < -self.vis_thres] = 0
        
        plot.image_plot([self.skyrec_zero, post_skyrec],
                        ["Sky Reconstruction", f"SkyRec IROS, iter. {iteration}"],
                        cbarlabel=["counts", "counts"],
                        cbarcmap=[self._cmap]*2,
                        simulated_sources=[self.source_pos]*2)

        plot.image_plot([snr, self.skyrec_zero - post_skyrec],
                        [f"IROS iter. {iteration} SNR", f"Residues: SkyRec - IROS{iteration}"],
                        cbarlabel=["counts", "counts"],
                        cbarcmap=[self._cmap]*2,
                        simulated_sources=[self.source_pos]*2)
    
    def _check_snr_norm(self,
                        iteration: int,
                        snr: np.array) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        fig.tight_layout()
        ax.hist(snr.reshape(-1), bins=50, density= True,
                color='SkyBlue', edgecolor='b', alpha=0.7)
        ax.plot(x := np.linspace(-5, 5, 1000), norm.pdf(x),
                color="OrangeRed", label="Normal distr.")
        ax.set_xlabel("SNR", fontsize=12, fontweight='bold')
        ax.set_ylabel("density", fontsize=12, fontweight='bold')
        ax.set_title(f"SkyRec SNR Statistics, iter. {iteration}",
                     fontsize=14, pad=8, fontweight='bold')
        ax.grid(visible=True, color="lightgray", linestyle="-", linewidth=0.3)
        ax.legend(loc='best')
        ax.tick_params(which='both', direction='in', width=2)
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.show()
    
    def _get_post_skyrec(self) -> np.array:
        # remove artifacts
        post_skyrec_zero = self.skyrec.copy()
        post_skyrec_zero[self.skyrec > self.vis_thres] = 0
        post_skyrec_zero[self.skyrec < -self.vis_thres] = 0
        return post_skyrec_zero
    
    @property
    def _cmap(self):
        return "viridis"



class IROS(_support):
    """Iterative Removal of Sources algorithm (dummy)."""

    def __init__(self,
                 n_iter: int,
                 snr_threshold: int | float,
                 skyrec: np.array,
                 skyrec_snr: np.array,
                 detector_image: np.array,
                 source_pos: list[tuple],
                 balancing_skyrec: object,
                 vis_thres: int):
        
        self.n = n_iter
        self.threshold = snr_threshold
        self.skyrec = skyrec
        self.snr = skyrec_snr
        self.detector_image = detector_image
        self.source_pos = source_pos
        self.vis_thres = vis_thres
        self.skyrec_zero = self._get_post_skyrec()

        self.bal = balancing_skyrec
        self.mask = self.bal.mask

        n, m = detector_image.shape
        self.h, self.v = n - 1, m - 1
        self.sky_shape = (n + 2*self.h, m + 2*self.v)

        self.sources_dataset = {'sources_pos': [],
                                'sources_counts': []}
    

    def iterate(self,
                check_snr_norm: bool = True,
                show_peaks: bool = True,
                show_results: bool = True,
                ) -> tuple[dict, np.array, np.array]:
        
        for i in range(self.n):

            if check_snr_norm:
                self._check_snr_norm(i + 1, self.snr)
            
            loc, _n_peaks = self._select_source(self.snr, i + 1, show_peaks)

            if self._check_peaks(_n_peaks):
                self.sources_dataset = self._record_source(self.sources_dataset,
                                                           loc, self.skyrec[*loc])

                shadow = self.shadowgram(loc, self.skyrec[*loc])
                self.detector_image, self.skyrec, self.snr = self.iros_application(self.detector_image,
                                                                                   shadow)
                
                if show_results:
                    self._show_results(self.skyrec, self.snr, i + 1)
            
            else:
                print("No sources detected with SNR over selected threshold...")
                break
            
        return self.sources_dataset, self.skyrec, self.snr


# end