echo "Running IROS reconstruction pipeline for Unit-1 of the LEM-X Observatory..."
echo ""

uv run IROSrec_pipeline__argparsed.py --help

uv run IROSrec_pipeline__argparsed.py \
   --run IROSbncmrk_detected_1ks_upx2_infdense_NewSGNorm_mask25 \
   --skyfield IROSDummy \
   --datadir iros_benchmark_2-50keV_mask_050_1040x17_infdet_1ks_mask25 \
   --dataset detected \
   --mask_pattern mask_NTHT_20250725.fits \
   # --up_fine 5 \

# uv run IROSrec_pipeline__argparsed.py \
#    --run GC_smoothed_rec_1ks_upx5_mask25 \
#    --skyfield GalacticCentre \
#    --datadir galctr_rxte-sax_mask_050_1040x17_2-50keV_1ks_mask25 \
#    --mask_pattern mask_NTHT_20250725.fits \
#    --up_fine 5 \
#    --smoothing \
#    --smoothing_snr_thresh 15.0 \
#    --baseline_irosrec /mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Outputs/OutGalacticCentre/galctr_rxte-sax_mask_050_1040x17_2-50keV_1ks_mask25/GC_rec_1ks_upx5_mask25/ \
#    # --energy_range 2.0 6.0 \
#    # --compose_unit \

# uv run IROSrec_pipeline__argparsed.py \
#    --run GC_rec_1ks_detected_2-6keV_smoothed_mask25 \
#    --skyfield GalacticCentre \
#    --datadir galctr_rxte-sax_mask_050_1040x17_2-50keV_1ks_mask25 \
#    --dataset detected \
#    --energy_range 2.0 6.0 \
#    --mask_pattern mask_NTHT_20250725.fits \
#    --compose_unit \
#    --smoothing \
#    --smoothing_snr_thresh 15.0 \
#    --baseline_irosrec /mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Outputs/OutGalacticCentre/galctr_rxte-sax_mask_050_1040x17_2-50keV_1ks_mask25/GC_rec_1ks_detected_2-6keV_mask25/ \

echo ""
echo "Finished IROS sky-field reconstruction(s)!"